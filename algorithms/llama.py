import numpy as np
import torch
from sklearn.utils import check_random_state
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseAlgorithm

class LlamaAlgorithm(BaseAlgorithm):
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 max_length: int = 256,
                 system_instruction: str = "You are a precise content moderation assistant. Decide if the given account text indicates a REAL or FAKE account. Answer with a single word: REAL or FAKE.",
                 n_shots: int = 6, few_shot_strategy: str = "balanced", shot_truncate: int = 224,
                 label_texts=("REAL","FAKE"),
                 label_prompt_template: str = "Text: {text}\nLabel: ",
                 device: str | None = None, quantization: str | None = None,
                 torch_dtype: str | None = "auto", hf_token: str | None = None,
                 random_state: int = 42):
        self.model_name = model_name
        self.max_length = max_length
        self.system_instruction = system_instruction
        self.n_shots = n_shots
        self.few_shot_strategy = few_shot_strategy
        self.shot_truncate = shot_truncate
        self.label_texts = label_texts
        self.label_prompt_template = label_prompt_template
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.quantization = quantization
        self.torch_dtype = torch_dtype
        self.hf_token = hf_token
        self.random_state = random_state

        self._rng = check_random_state(self.random_state)
        self._tokenizer = None
        self._model = None
        self._few_shots = []
        self._fitted = False
        self._label_to_idx = None
        self._idx_to_label = None
        self._classes_ = None
        self._has_chat_template = False
        self._label_token_ids = None

        self._load_model()

    @property
    def name(self) -> str:
        return "LLaMA"

    def _load_model(self):
        if self.torch_dtype == "auto":
            dtype = "auto"
        elif self.torch_dtype is None:
            dtype = None
        else:
            dtype = getattr(torch, self.torch_dtype)

        q_kwargs = {}
        if self.quantization in ("8bit", "4bit"):
            try:
                import bitsandbytes as _  # noqa
                q_kwargs = {"load_in_8bit": self.quantization == "8bit",
                            "load_in_4bit": self.quantization == "4bit"}
            except Exception:
                q_kwargs = {}

        auth = {"token": self.hf_token} if self.hf_token else {}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **auth)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            **q_kwargs,
            **auth
        )
        if self.device.type != "cuda":
            self._model = self._model.to(self.device)
        self._has_chat_template = hasattr(self._tokenizer, "apply_chat_template") and bool(self._tokenizer.chat_template)

    def _extract_texts(self, X):
        try:
            import pandas as pd
        except Exception:
            pd = None

        if pd is not None:
            if isinstance(X, pd.Series):
                return X.astype(str).tolist()
            if isinstance(X, pd.DataFrame):
                text_cols = [c for c in X.columns if (X[c].dtype == object) or str(X[c].dtype).startswith("string")]
                if not text_cols:
                    raise ValueError("Provide text columns or pass Series/List[str].")
                return (X[text_cols].astype(str)).agg(" ".join, axis=1).tolist()

        X = np.asarray(X)
        if X.ndim == 1:
            return [str(t) for t in X.tolist()]
        raise ValueError("X must be 1-D list/Series of texts or DataFrame with text columns.")

    def _prepare_labels(self, y):
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("LLaMA classifier supports exactly two classes.")
        self._classes_ = np.sort(classes)
        self._label_to_idx = {c: i for i, c in enumerate(self._classes_)}
        self._idx_to_label = {i: c for c, i in self._label_to_idx.items()}
        y_idx = np.array([self._label_to_idx[v] for v in y], dtype=int)
        return y_idx

    def _truncate_text(self, s: str, limit: int) -> str:
        toks = self._tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(toks) <= limit:
            return s
        return self._tokenizer.decode(toks[:limit], skip_special_tokens=True)

    def _select_few_shots(self, texts, y_idx, k: int):
        if k <= 0 or len(texts) == 0:
            return []
        idx0 = np.where(y_idx == 0)[0].tolist()
        idx1 = np.where(y_idx == 1)[0].tolist()
        self._rng.shuffle(idx0); self._rng.shuffle(idx1)
        shots = []
        take0 = k // 2
        take1 = k - take0
        if self.few_shot_strategy == "balanced":
            pool0 = idx0[:take0] if len(idx0) >= take0 else idx0
            pool1 = idx1[:take1] if len(idx1) >= take1 else idx1
            chosen = pool0 + pool1
        else:
            chosen = list(range(min(k, len(texts))))
        for i in chosen:
            lab_idx = y_idx[i]
            lab_text = self.label_texts[lab_idx]
            shots.append((self._truncate_text(texts[i], self.shot_truncate), lab_text))
        return shots

    def _build_messages(self, text: str):
        msgs = [{"role": "system", "content": self.system_instruction}]
        if self._few_shots:
            for ex_text, ex_label in self._few_shots:
                msgs.append({"role": "user", "content": self.label_prompt_template.format(text=ex_text)})
                msgs.append({"role": "assistant", "content": ex_label})
        msgs.append({"role": "user", "content": self.label_prompt_template.format(text=text) + "Choose one: REAL or FAKE.\nAnswer:"})
        return msgs

    def _build_prompt(self, text: str):
        if self._has_chat_template:
            msgs = self._build_messages(text)
            return self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        parts = [self.system_instruction.strip(), ""]
        if self._few_shots:
            parts.append("Examples:")
            for ex_text, ex_label in self._few_shots:
                parts.append(self.label_prompt_template.format(text=ex_text) + ex_label)
            parts.append("")
        parts.append("Now classify the following:")
        parts.append(self.label_prompt_template.format(text=text) + "Choose one: REAL or FAKE.\nAnswer:")
        return "\n".join(parts)

    def _prepare_label_token_ids(self):
        cands = []
        for s in self.label_texts:
            ids = self._tokenizer(" " + s, add_special_tokens=False, return_tensors=None)["input_ids"]
            if len(ids) == 0:
                ids = self._tokenizer(s, add_special_tokens=False, return_tensors=None)["input_ids"]
            cands.append(ids)
        self._label_token_ids = cands

    def _encode_for_scoring(self, prompt: str, label_ids):
        p_ids = self._tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids.to(self.device)
        l_ids = torch.tensor([label_ids], dtype=torch.long, device=self.device)
        inp = torch.cat([p_ids, l_ids], dim=1)
        return inp, p_ids.shape[1], l_ids[0]

    @torch.no_grad()
    def _candidate_logprob(self, prompt: str, label_ids) -> float:
        inp, split, label_ids = self._encode_for_scoring(prompt, label_ids)
        out = self._model(input_ids=inp)
        logits = out.logits[:, :-1, :]
        start = split - 1
        logp = 0.0
        for j in range(len(label_ids)):
            tok_id = int(label_ids[j].item())
            lp = torch.log_softmax(logits[0, start + j, :], dim=-1)[tok_id]
            logp += float(lp.item())
        return logp

    def fit(self, X, y):
        texts = self._extract_texts(X)
        y_idx = self._prepare_labels(y)
        self._few_shots = self._select_few_shots(texts, y_idx, self.n_shots)
        self._prepare_label_token_ids()
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call .fit(X,y) first.")

    @property
    def classes_(self):
        return self._classes_

    def predict_proba(self, X):
        self._check_fitted()
        texts = self._extract_texts(X)
        probs = np.zeros((len(texts), 2), dtype=float)
        for i, t in enumerate(texts):
            t_tr = self._truncate_text(t, self.max_length)
            prompt = self._build_prompt(t_tr)
            lp0 = self._candidate_logprob(prompt, self._label_token_ids[0])
            lp1 = self._candidate_logprob(prompt, self._label_token_ids[1])
            m = max(lp0, lp1)
            e0 = np.exp(lp0 - m); e1 = np.exp(lp1 - m)
            s = e0 + e1
            probs[i, 0] = e0 / s
            probs[i, 1] = e1 / s
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return np.array([self._idx_to_label[i] for i in idx])
