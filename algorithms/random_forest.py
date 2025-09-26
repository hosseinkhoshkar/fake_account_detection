import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, f1_score
from .base import BaseAlgorithm

class RandomForestAlgorithm(BaseAlgorithm):
    def __init__(self, random_state: int = 42, do_search: bool = False, n_iter: int = 20, cv_splits: int = 3):
        self.random_state = random_state
        self.do_search = do_search          
        self.n_iter = n_iter                
        self.cv_splits = cv_splits
        self.model = None
        self.best_threshold = 0.5
        self.fake_label = 1
        self.normal_label = 0

    @property
    def name(self) -> str:
        return "RandomForest"

  
    def _split_columns(self, X):
        try:
            import pandas as pd
        except Exception:
            pd = None

        num_cols, cat_cols = [], []
        if pd is not None and hasattr(X, "select_dtypes"):
            for c in X.columns:
                s = X[c]
                s_num = pd.to_numeric(s, errors="coerce")
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
                df = df.apply(pd.to_numeric, errors="coerce")
                return df.values
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

    # --- fit/predict ---
    def fit(self, X, y) -> None:
        vc = {lbl: (y == lbl).sum() for lbl in np.unique(y)}
        self.normal_label = max(vc, key=vc.get)
        candidates = [lbl for lbl in vc if lbl != self.normal_label]
        self.fake_label = candidates[0] if candidates else (1 if self.normal_label == 0 else 0)

        pre = self._build_preprocessor(X)
        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=self.random_state,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
        base = Pipeline([
            ("pre", pre),
            ("rf", rf)
        ])

        if self.do_search:
       
            param_distributions = {
                "rf__n_estimators": [300, 400, 500],
                "rf__max_depth": [None, 10, 20, 30],
                "rf__max_features": ["sqrt", "log2", 0.5],
                "rf__min_samples_split": [2, 5, 10],
                "rf__min_samples_leaf": [1, 2, 4],
                "rf__bootstrap": [True, False],
            }
            cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
            rs = RandomizedSearchCV(
                base,
                param_distributions,
                n_iter=self.n_iter,
                scoring=make_scorer(f1_score, pos_label=self.fake_label),
                n_jobs=-1,
                cv=cv,
                verbose=0,
                random_state=self.random_state,
                error_score="raise",
            )
            rs.fit(X, y)
            self.model = rs.best_estimator_
        else:
            self.model = base
            self.model.fit(X, y)

        # threshold tuning روی validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        self.model.fit(X_tr, y_tr)

        rf_est = self.model.named_steps["rf"]
        classes = list(rf_est.classes_)
        idx = classes.index(self.fake_label) if self.fake_label in classes else (1 if len(classes) > 1 else 0)

        proba_val = self.model.predict_proba(X_val)[:, idx]
        thresholds = np.linspace(0.05, 0.95, 181)
        best_thr, best_f1 = 0.5, -1.0
        for t in thresholds:
            y_hat = np.where(proba_val >= t, self.fake_label, self.normal_label)
            f1 = f1_score(y_val, y_hat, pos_label=self.fake_label, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(t)
        self.best_threshold = best_thr


        self.model.fit(X, y)

    def predict(self, X):
        rf_est = self.model.named_steps["rf"]
        classes = list(rf_est.classes_)
        idx = classes.index(self.fake_label) if self.fake_label in classes else (1 if len(classes) > 1 else 0)
        proba = self.model.predict_proba(X)[:, idx]
        return np.where(proba >= self.best_threshold, self.fake_label, self.normal_label).astype(int)
