import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler

from .base import BaseAlgorithm


class _AE(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128, 64), dropout=0.1):
        super().__init__()
        h1, h2, h3 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(h3, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class AutoencoderAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        epochs: int = 60,
        batch_size: int = 256,
        lr: float = 1e-3,
        hidden=(256, 128, 64),
        dropout: float = 0.1,
        noise_std: float = 0.01,
        train_on_normal_only: bool = True,
        early_stop_patience: int = 8,
        device: str | None = None,
        random_state: int = 42,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden = hidden
        self.dropout = dropout
        self.noise_std = noise_std
        self.train_on_normal_only = train_on_normal_only
        self.early_stop_patience = early_stop_patience
        self.random_state = random_state
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.preprocessor_ = None
        self.scaler_ = StandardScaler()
        self.model_ = None
        self.best_threshold_ = 0.0
        self.fake_label = 1
        self.normal_label = 0
        self._fitted = False

    @property
    def name(self) -> str:
        return "Autoencoder"

    def _split_columns(self, X):
        try:
            import pandas as pd
        except Exception:
            pd = None

        num_cols, cat_cols = [], []
        if pd is not None and hasattr(X, "select_dtypes"):
            for c in X.columns:
                s_num = pd.to_numeric(X[c], errors="coerce")
                if s_num.notna().mean() >= 0.95:
                    num_cols.append(c)
                else:
                    cat_cols.append(c)
        else:
            num_cols = list(range(X.shape[1]))
        return num_cols, cat_cols

    def _to_numeric_block(self, A):
        try:
            import pandas as pd
            if hasattr(A, "apply"):
                return A.apply(pd.to_numeric, errors="coerce")
            else:
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
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

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

    def _to_tensor(self, X_np):
        return torch.from_numpy(X_np.astype(np.float32)).to(self.device)

    def _recon_errors(self, model, X_np):
        model.eval()
        with torch.no_grad():
            xb = self._to_tensor(X_np)
            xhat = model(xb)
            err = ((xb - xhat) ** 2).mean(dim=1).cpu().numpy()
        return err

    def fit(self, X, y):
        vc = {lbl: (y == lbl).sum() for lbl in np.unique(y)}
        self.normal_label = max(vc, key=vc.get)
        others = [lbl for lbl in vc if lbl != self.normal_label]
        self.fake_label = others[0] if others else (1 if self.normal_label == 0 else 0)

        self.preprocessor_ = self._build_preprocessor(X)
        X_num = self.preprocessor_.fit_transform(X)
        self.scaler_.fit(X_num)
        X_scaled = self.scaler_.transform(X_num)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        if self.train_on_normal_only:
            mask = (y_tr == self.normal_label)
            X_tr_used = X_tr[mask]
        else:
            X_tr_used = X_tr

        in_dim = X_tr.shape[1]
        self.model_ = _AE(in_dim, hidden=self.hidden, dropout=self.dropout).to(self.device)
        opt = optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_tr_t = self._to_tensor(X_tr_used)
        tr_ds = torch.utils.data.TensorDataset(X_tr_t, X_tr_t)
        tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=min(self.batch_size, len(tr_ds)), shuffle=True)

        X_val_t = self._to_tensor(X_val)
        val_ds = torch.utils.data.TensorDataset(X_val_t, X_val_t)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=min(self.batch_size, len(val_ds)), shuffle=False)

        best_val = np.inf
        patience = 0
        torch.manual_seed(self.random_state or 0)

        for _ in range(self.epochs):
            self.model_.train()
            for xb, yb in tr_dl:
                opt.zero_grad()
                if self.noise_std > 0:
                    noise = torch.randn_like(xb) * self.noise_std
                    x_in = xb + noise
                else:
                    x_in = xb
                xhat = self.model_(x_in)
                loss = loss_fn(xhat, yb)
                loss.backward()
                opt.step()

            self.model_.eval()
            with torch.no_grad():
                val_losses = []
                for xb, yb in val_dl:
                    xhat = self.model_(xb)
                    val_losses.append(loss_fn(xhat, yb).item())
                cur_val = float(np.mean(val_losses)) if val_losses else np.inf

            if cur_val + 1e-8 < best_val:
                best_val = cur_val
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    break

        if 'best_state' in locals():
            self.model_.load_state_dict(best_state)

        val_err = self._recon_errors(self.model_, X_val)
        thresholds = np.unique(np.percentile(val_err, np.linspace(50, 99.9, 300)))
        best_thr, best_f1 = float(np.median(val_err)), -1.0
        for t in thresholds:
            yhat = np.where(val_err >= t, self.fake_label, self.normal_label)
            f1 = f1_score(y_val, yhat, pos_label=self.fake_label, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(t)
        self.best_threshold_ = best_thr

        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("AutoencoderAlgorithm is not fitted yet.")

    def predict(self, X):
        self._check_fitted()
        X_num = self.preprocessor_.transform(X)
        X_scaled = self.scaler_.transform(X_num)
        err = self._recon_errors(self.model_, X_scaled)
        yhat = np.where(err >= self.best_threshold_, self.fake_label, self.normal_label)
        return yhat.astype(int)
