import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from .base import BaseAlgorithm

class KNNAlgorithm(BaseAlgorithm):
    def __init__(self, do_grid: bool = False, cv_splits: int = 5, random_state: int = 42):
        self.do_grid = do_grid
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.model = None

    @property
    def name(self) -> str:
        return "KNN"

    def _split_columns(self, X):
        try:
            import pandas as pd
        except Exception:
            pd = None

        num_cols, cat_cols = [], []
        if pd is not None and isinstance(X, pd.DataFrame):
            for c in X.columns:
                s = X[c]
            
                s_num = pd.to_numeric(s, errors="coerce")
                ratio = s_num.notna().mean()
                if ratio >= 0.95: 
                    num_cols.append(c)
                else:
                    cat_cols.append(c)
        else:
           
            if isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.number):
                num_cols = list(range(X.shape[1]))
            else:
                cat_cols = list(range(X.shape[1]))
        return num_cols, cat_cols

    def _build_preprocessor(self, X):
        import pandas as pd

        num_cols, cat_cols = self._split_columns(X)

        
        to_numeric = FunctionTransformer(
            lambda df: df.apply(pd.to_numeric, errors="coerce") if isinstance(df, pd.DataFrame) else df,
            feature_names_out="one-to-one" if hasattr(FunctionTransformer, "__init__") else None
        )

        num_pipe = Pipeline([
            ("to_numeric", to_numeric),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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
        pre = self._build_preprocessor(X)
        base = Pipeline([
            ("pre", pre),
            ("knn", KNeighborsClassifier())
        ])

        if self.do_grid:
            param_grid = {
                "knn__n_neighbors": list(range(1, 18, 2)),
                "knn__weights": ["uniform", "distance"],
                "knn__p": [1, 2]
            }
            gs = GridSearchCV(base, param_grid, scoring="f1_macro", cv=self.cv_splits, n_jobs=-1)
            gs.fit(X, y)
            self.model = gs.best_estimator_
        else:
            self.model = Pipeline([
                ("pre", pre),
                ("knn", KNeighborsClassifier(n_neighbors=7, weights="distance", p=2))
            ])
            self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X).astype(int)
