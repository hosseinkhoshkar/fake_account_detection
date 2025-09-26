import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from lightgbm import LGBMClassifier
from .base import BaseAlgorithm

class LGBMAlgorithm(BaseAlgorithm):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.calibrated_model = None
        self.best_threshold = 0.5
        self.fake_label = 1
        self.normal_label = 0

    @property
    def name(self) -> str:
        return "LightGBM"

   
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

   
    def fit(self, X, y) -> None:
        
        vc = {lbl: (y == lbl).sum() for lbl in np.unique(y)}
        self.normal_label = max(vc, key=vc.get)
        candidates = [lbl for lbl in vc if lbl != self.normal_label]
        self.fake_label = candidates[0] if candidates else (1 if self.normal_label == 0 else 0)

        pos = int((y == self.fake_label).sum())
        neg = int((y == self.normal_label).sum())
        spw = max(1.0, neg / max(1, pos))

        # split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        pre = self._build_preprocessor(X_tr)

        
        safe_grid = [
            dict(n_estimators=400, learning_rate=0.05, num_leaves=31,  min_child_samples=20, subsample=0.9, feature_fraction=0.9),
            dict(n_estimators=600, learning_rate=0.05, num_leaves=63,  min_child_samples=20, subsample=0.9, feature_fraction=0.9),
            dict(n_estimators=800, learning_rate=0.07, num_leaves=63,  min_child_samples=30, subsample=0.8, feature_fraction=0.8),
        ]

        best_model, best_f1 = None, -1.0
        for cfg in safe_grid:
            try:
                lgb = LGBMClassifier(
                    objective="binary",
                    boosting_type="gbdt",
                    max_depth=-1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=-1,
                    scale_pos_weight=spw,
                    **cfg,
                )
                pipe = Pipeline([
                    ("pre", pre),
                    ("lgbm", lgb)
                ])
                pipe.fit(X_tr, y_tr)

               
                classes = list(pipe.named_steps["lgbm"].classes_)
                idx = classes.index(self.fake_label) if self.fake_label in classes else (1 if len(classes) > 1 else 0)

                probs = pipe.predict_proba(X_val)[:, idx]
                yhat = np.where(probs >= 0.5, self.fake_label, self.normal_label)
                f = f1_score(y_val, yhat, pos_label=self.fake_label, zero_division=0)
                if f > best_f1:
                    best_f1, best_model = f, pipe
            except Exception:
                continue

        if best_model is None:
            #fallback
            lgb = LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                max_depth=-1,
                n_estimators=600,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                feature_fraction=0.9,
                min_child_samples=20,
                reg_lambda=1.0,
                reg_alpha=0.0,
                scale_pos_weight=spw,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
            best_model = Pipeline([
                ("pre", pre),
                ("lgbm", lgb)
            ])
            best_model.fit(X_tr, y_tr)

        self.model = best_model

        
        self.calibrated_model = CalibratedClassifierCV(self.model, cv="prefit", method="sigmoid").fit(X_val, y_val)

        # threshold tuning
        thresholds = np.linspace(0.05, 0.95, 181)
        classes = list(self.calibrated_model.classes_)
        idx = classes.index(self.fake_label) if self.fake_label in classes else (1 if len(classes) > 1 else 0)
        proba_val = self.calibrated_model.predict_proba(X_val)[:, idx]

        best_thr, best_f1 = 0.5, -1.0
        for t in thresholds:
            yhat = np.where(proba_val >= t, self.fake_label, self.normal_label)
            f = f1_score(y_val, yhat, pos_label=self.fake_label, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, float(t)
        self.best_threshold = best_thr

      
        self.model.fit(X, y)
        self.calibrated_model = CalibratedClassifierCV(self.model, cv="prefit", method="sigmoid").fit(X, y)

    def predict(self, X):
        classes = list(self.calibrated_model.classes_)
        idx = classes.index(self.fake_label) if self.fake_label in classes else (1 if len(classes) > 1 else 0)
        proba = self.calibrated_model.predict_proba(X)[:, idx]
        return np.where(proba >= self.best_threshold, self.fake_label, self.normal_label).astype(int)
