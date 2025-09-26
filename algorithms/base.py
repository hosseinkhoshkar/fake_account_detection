
from abc import ABC, abstractmethod
import numpy as np

class BaseAlgorithm(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        return None

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
