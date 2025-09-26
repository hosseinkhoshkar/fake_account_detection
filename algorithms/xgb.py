import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from xgboost import XGBClassifier
from .base import BaseAlgorithm

class XGBAlgorithm(BaseAlgorithm):
    def __init__(self, random_state: int = 42, n_iter: int = 20, cv_splits: int = 5):
        self.random_state = random_state
        self.n_iter = n_iter
        self.cv_splits = cv_splits
        self.model = None
        self.calibrated_model = None
        self.best_threshold = 0.5
        self.fake_label = 1
        self.normal_label = 0

    @property
    def name(self) -> str:
        return "XGBoost"

   
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

    def _safe_cv_splits(self, y):
        uniq = np.unique(y)
        min_class = int(min((y == u).sum() for u in uniq))
        return max(2, min(self.cv_splits, min_class))

    
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
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            max_depth=6,
            reg_lambda=1.0,
            reg_alpha=0.0,
            scale_pos_weight=spw,
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=-1,
        )
        pipe = Pipeline([
            ("pre", pre),
            ("xgb", xgb)
        ])

        param_xgb = {
            "xgb__n_estimators": [300, 400, 600, 800],
            "xgb__learning_rate": [0.03, 0.05, 0.07, 0.1],
            "xgb__max_depth": [4, 6, 8],
            "xgb__min_child_weight": [1, 3],
            "xgb__subsample": [0.7, 0.9],
            "xgb__colsample_bytree": [0.6, 0.9],
            "xgb__reg_lambda": [0.5, 1.0],
            "xgb__reg_alpha": [0.0, 0.1],
        }

        cv = StratifiedKFold(
            n_splits=self._safe_cv_splits(y_tr),
            shuffle=True,
            random_state=self.random_state
        )
        rs = RandomizedSearchCV(
            pipe,
            param_xgb,
            n_iter=min(self.n_iter, 40),
            scoring=make_scorer(f1_score, pos_label=self.fake_label),
            n_jobs=-1,
            cv=cv,
            verbose=0,
            random_state=self.random_state,
            error_score="raise",
        )
        rs.fit(X_tr, y_tr)
        self.model = rs.best_estimator_

        
        self.calibrated_model = CalibratedClassifierCV(self.model, cv="prefit", method="sigmoid").fit(X_val, y_val)

        # threshold tuning
        thresholds = np.linspace(0.05, 0.95, 181)
        classes = list(self.calibrated_model.classes_)
        idx = classes.index(self.fake_label) if self.fake_label in classes else (1 if len(classes)>1 else 0)
        proba_val = self.calibrated_model.predict_proba(X_val)[:, idx]

        best_thr, best_f1 = 0.5, -1.0
        for t in thresholds:
            yhat = np.where(proba_val >= t, self.fake_label, self.normal_label)
            f = f1_score(y_val, yhat, pos_label=self.fake_label, zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, float(t)
        self.best_threshold = best_thr

        # retrain calibrated on full
        self.model.fit(X, y)
        self.calibrated_model = CalibratedClassifierCV(self.model, cv="prefit", method="sigmoid").fit(X, y)

    def predict(self, X):
        classes = list(self.calibrated_model.classes_)
        idx = classes.index(self.fake_label) if self.fake_label in classes else (1 if len(classes)>1 else 0)
        proba = self.calibrated_model.predict_proba(X)[:, idx]
        return np.where(proba >= self.best_threshold, self.fake_label, self.normal_label).astype(int)
