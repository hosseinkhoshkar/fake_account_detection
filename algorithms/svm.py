import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from .base import BaseAlgorithm

class SVMAlgorithm(BaseAlgorithm):
    def __init__(self, do_grid: bool = False, cv_splits: int = 5, random_state: int = 42):
        self.do_grid = do_grid
        self.cv_splits = cv_splits
        self.random_state = random_state
        self.model = None

    @property
    def name(self) -> str:
        return "SVM"

    def _preprocessor(self):
        num_pipe = Pipeline([
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
        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, selector(dtype_include=np.number)),
                ("cat", cat_pipe, selector(dtype_exclude=np.number)),
            ],
            remainder="drop",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        base = Pipeline([
            ("pre", self._preprocessor()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=self.random_state))
        ])
        if self.do_grid:
            param_grid = {
                "svm__C": [0.5, 1, 5, 10],
                "svm__gamma": ["scale", 0.1, 0.01],
                "svm__class_weight": [None, "balanced"]
            }
            gs = GridSearchCV(base, param_grid, scoring="f1_macro", cv=self.cv_splits, n_jobs=-1)
            gs.fit(X, y)
            self.model = gs.best_estimator_
        else:
            self.model = base
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).astype(int)
