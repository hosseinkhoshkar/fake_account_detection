import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils import check_random_state
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from .base import BaseAlgorithm

class _Generator(nn.Module):
    def __init__(self, noise_dim: int, class_dim: int, out_dim: int, hidden):
        super().__init__()
        dims = [noise_dim + class_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, z, y_onehot):
        x = torch.cat([z, y_onehot], dim=1)
        return self.net(x)

class _Discriminator(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden):
        super().__init__()
        dims = [in_dim] + hidden
        feats = []
        for i in range(len(dims)-1):
            feats += [nn.Linear(dims[i], dims[i+1]), nn.LeakyReLU(0.2, inplace=True)]
        self.features = nn.Sequential(*feats)
        self.src_out = nn.Linear(dims[-1], 1)
        self.cls_out = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        f = self.features(x)
        return self.src_out(f), self.cls_out(f)

class _TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.X = torch.from_numpy(X)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("ACGANAlgorithm supports binary labels only.")
        self.class_to_index = {int(c): i for i, c in enumerate(sorted(classes))}
        self.index_to_class = {i: c for c, i in self.class_to_index.items()}
        y_idx = [self.class_to_index[int(v)] for v in y]
        self.y = torch.tensor(y_idx, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ACGANAlgorithm(BaseAlgorithm):
    def __init__(self, input_dim: int, noise_dim: int = 64,
                 generator_hidden=None, discriminator_hidden=None,
                 epochs: int = 40, batch_size: int = 256, lr: float = 2e-4,
                 betas=(0.5, 0.999), device: str | None = None, random_state: int = 42,
                 class_weight: str | None = "balanced",
                 lambda_cls_D: float = 0.5, lambda_cls_G: float = 0.5):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.generator_hidden = generator_hidden or [256, 512]
        self.discriminator_hidden = discriminator_hidden or [512, 256]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.class_weight = class_weight
        self.lambda_cls_D = lambda_cls_D
        self.lambda_cls_G = lambda_cls_G

        self._rng = check_random_state(random_state)
        self._num_classes = 2
        self._G = None
        self._D = None
        self._fitted = False
        self._class_to_index = None
        self._index_to_class = None
        self._classes_ = None

       
        self._pre = None  # ColumnTransformer

    @property
    def name(self) -> str:
        return "ACGAN"

    
    def _split_columns(self, X):
        try:
            import pandas as pd
        except Exception:
            pd = None

        num_cols, cat_cols = [], []
        if pd is not None and hasattr(X, "select_dtypes"):
            for c in X.columns:
                s_num = pd.to_numeric(X[c], errors="coerce")
                ratio = s_num.notna().mean()
                if ratio >= 0.95:
                    num_cols.append(c)
                else:
                    cat_cols.append(c)
        else:
          
            num_cols = list(range(X.shape[1]))
        return num_cols, cat_cols

    def _to_numeric_block(self, A):
        try:
            import pandas as pd
            if hasattr(A, "apply"):  # DataFrame
                return A.apply(pd.to_numeric, errors="coerce")
            else:  # ndarray
                df = pd.DataFrame(A)
                return df.apply(pd.to_numeric, errors="coerce").values
        except Exception:
            return A

    def _build_preprocessor(self, X):
        num_cols, cat_cols = self._split_columns(X)

        num_pipe = Pipeline([
            ("to_numeric", FunctionTransformer(self._to_numeric_block, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
        ])

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)  # sklearn >=1.2
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)        

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ])

        transformers = []
        if len(num_cols) > 0:
            transformers.append(("num", num_pipe, num_cols))
        if len(cat_cols) > 0:
            transformers.append(("cat", cat_pipe, cat_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X, y):
        
        self._pre = self._build_preprocessor(X)
        X_num = self._pre.fit_transform(X).astype(np.float32)

        
        self.input_dim = X_num.shape[1]

        ds = _TabularDataset(X_num, y)
        self._class_to_index = ds.class_to_index
        self._index_to_class = ds.index_to_class
        self._classes_ = np.array([self._index_to_class[0], self._index_to_class[1]])

        y_idx = ds.y.numpy()
        class_counts = np.bincount(y_idx, minlength=2)
        if self.class_weight == "balanced" and class_counts.min() > 0:
            weights = 1.0 / class_counts
            sample_weights = weights[y_idx]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), sampler=sampler, drop_last=True)
        else:
            loader = DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True, drop_last=True)
        if len(loader) == 0:
            loader = DataLoader(ds, batch_size=len(ds), shuffle=True, drop_last=False)

        self._G = _Generator(self.noise_dim, self._num_classes, self.input_dim, self.generator_hidden).to(self.device)
        self._D = _Discriminator(self.input_dim, self._num_classes, self.discriminator_hidden).to(self.device)
        opt_G = optim.Adam(self._G.parameters(), lr=self.lr, betas=self.betas)
        opt_D = optim.Adam(self._D.parameters(), lr=self.lr, betas=self.betas)
        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss()

        def one_hot(labels):
            return torch.zeros((labels.size(0), self._num_classes), device=labels.device).scatter_(1, labels.unsqueeze(1), 1.0)

        torch.manual_seed(self.random_state or 0)
        self._G.train(); self._D.train()

        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                bs = xb.size(0)

                # Train D
                self._D.zero_grad()
                src_real, cls_real = self._D(xb)
                loss_src_real = bce(src_real.view(-1), torch.ones(bs, device=self.device))
                loss_cls_real = ce(cls_real, yb)

                z = torch.randn(bs, self.noise_dim, device=self.device)
                y_fake = torch.randint(0, self._num_classes, (bs,), device=self.device)
                y_fake_oh = one_hot(y_fake)
                x_fake = self._G(z, y_fake_oh).detach()
                src_fake, _ = self._D(x_fake)
                loss_src_fake = bce(src_fake.view(-1), torch.zeros(bs, device=self.device))

                loss_D = loss_src_real + loss_src_fake + self.lambda_cls_D * loss_cls_real
                loss_D.backward()
                opt_D.step()

                # Train G
                self._G.zero_grad()
                z = torch.randn(bs, self.noise_dim, device=self.device)
                y_gen = torch.randint(0, self._num_classes, (bs,), device=self.device)
                y_gen_oh = one_hot(y_gen)
                xg = self._G(z, y_gen_oh)
                src_out, cls_out = self._D(xg)
                loss_G_src = bce(src_out.view(-1), torch.ones(bs, device=self.device))
                loss_G_cls = ce(cls_out, y_gen)
                loss_G = loss_G_src + self.lambda_cls_G * loss_G_cls
                loss_G.backward()
                opt_G.step()

        self._fitted = True
        self._G.eval(); self._D.eval()
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("ACGANAlgorithm is not fitted.")

    def predict_proba(self, X):
        self._check_fitted()

        X_num = self._pre.transform(X).astype(np.float32)
        with torch.no_grad():
            xb = torch.from_numpy(X_num).to(self.device)
            src_logits, cls_logits = self._D(xb)
            src_prob_real = torch.sigmoid(src_logits.view(-1))
            cls_prob = torch.softmax(cls_logits, dim=1)
            p_fake_idx = 0.7 * cls_prob[:, 1] + 0.3 * (1.0 - src_prob_real)
            p_fake_idx = p_fake_idx.clamp(0, 1).cpu().numpy()

        p = np.zeros((X_num.shape[0], 2), dtype=float)
        label_for_idx1 = self._index_to_class[1]
        pos_fake = int(np.where(self._classes_ == label_for_idx1)[0][0])
        p[:, pos_fake] = p_fake_idx
        p[:, 1 - pos_fake] = 1.0 - p_fake_idx
        return p

    def predict(self, X):
        proba = self.predict_proba(X)
        pos_idx = 1
        yhat_bin = (proba[:, pos_idx] >= 0.5).astype(int)
        return np.where(yhat_bin == 1, self._classes_[1], self._classes_[0])
