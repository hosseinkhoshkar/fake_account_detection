
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.utils import check_random_state
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from .base import BaseAlgorithm

class _TextDataset(Dataset):
    def __init__(self, texts, labels_idx, tokenizer, max_length: int):
        self.texts = texts
        self.labels_idx = labels_idx.astype(int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        enc["labels"] = int(self.labels_idx[i])
        return enc

class BertAlgorithm(BaseAlgorithm):
    def __init__(self, model_name: str = "distilbert-base-multilingual-cased",
                 max_length: int = 192, epochs: int = 3, batch_size: int = 16,
                 lr: float = 2e-5, weight_decay: float = 0.01, warmup_ratio: float = 0.06,
                 gradient_accumulation_steps: int = 1, class_weight: str | None = "balanced",
                 device: str | None = None, random_state: int = 42,
                 text_cols=None, text_sep: str = " [SEP] "):
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.class_weight = class_weight
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.text_cols = text_cols
        self.text_sep = text_sep

        self._rng = check_random_state(self.random_state)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self._num_labels = 2
        self._fitted = False
        self._label_to_idx = None
        self._idx_to_label = None
        self._classes_ = None
        self._model = None

    @property
    def name(self) -> str:
        return "BERT"

    def _extract_texts(self, X):
        try:
            import pandas as pd
        except Exception:
            pd = None

        if pd is not None:
            if isinstance(X, pd.Series):
                return X.astype(str).tolist()
            if isinstance(X, pd.DataFrame):
                if not self.text_cols:
                    text_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("string")]
                    if not text_cols:
                        raise ValueError("Provide `text_cols` when X is a DataFrame with no obvious text columns.")
                else:
                    text_cols = self.text_cols
                texts = (X[text_cols].astype(str)).agg(self.text_sep.join, axis=1).tolist()
                return texts

        X = np.asarray(X)
        if X.ndim == 1:
            return [str(t) for t in X.tolist()]
        raise ValueError("X must be 1-D list/Series of texts or a DataFrame with `text_cols`.")

    def _prepare_labels(self, y):
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("This classifier supports binary labels only.")
        self._classes_ = np.sort(classes)
        self._label_to_idx = {c: i for i, c in enumerate(self._classes_)}
        self._idx_to_label = {i: c for c, i in self._label_to_idx.items()}
        y_idx = np.array([self._label_to_idx[v] for v in y], dtype=int)
        return y_idx

    def fit(self, X, y):
        texts = self._extract_texts(X)
        y_idx = self._prepare_labels(y)

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self._num_labels
        ).to(self.device)

        if self.class_weight == "balanced":
            binc = np.bincount(y_idx, minlength=self._num_labels)
            cw = (binc.sum() / (self._num_labels * np.maximum(binc, 1))).astype(np.float32)
            class_weights = torch.tensor(cw, device=self.device)
        else:
            class_weights = None

        ds = _TextDataset(texts, y_idx, self._tokenizer, self.max_length)
        collator = DataCollatorWithPadding(self._tokenizer)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, collate_fn=collator)

        num_update_steps_per_epoch = max(1, len(loader) // self.gradient_accumulation_steps)
        t_total = int(num_update_steps_per_epoch * self.epochs)
        optim = AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=int(self.warmup_ratio * t_total),
            num_training_steps=t_total,
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        self._model.train()
        global_step = 0
        torch.manual_seed(self.random_state or 0)
        for epoch in range(self.epochs):
            optim.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                outputs = self._model(**batch)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

        self._fitted = True
        self._model.eval()
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("BERT model is not fitted.")

    def predict_proba(self, X):
        self._check_fitted()
        texts = self._extract_texts(X)
        ds = _TextDataset(texts, np.zeros(len(texts), dtype=int), self._tokenizer, self.max_length)
        collator = DataCollatorWithPadding(self._tokenizer)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=collator)

        probs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                logits = self._model(**batch).logits
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                probs.append(p)
        probs = np.vstack(probs) if probs else np.zeros((0, 2), dtype=float)
        return probs

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = proba.argmax(axis=1)
        return np.array([self._idx_to_label[i] for i in idx])
