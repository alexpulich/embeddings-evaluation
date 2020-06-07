from abc import ABC, abstractmethod

import numpy as np


class StructuredSourceStrategy(ABC):
    """
    Base source strategy abstract class
    """

    @abstractmethod
    def apply(self, scores, pairs, structured_sources_coef) -> np.ndarray:
        pass


class NoStructuredSourceStrategy(StructuredSourceStrategy):
    """
    Strategy for plain word embeddings evaluation
    """
    def apply(self, scores, pairs, structured_sources_coef) -> np.ndarray:
        pass


class OOVStrategy(ABC):
    """
    Base strategy for OOV handling
    """

    @abstractmethod
    def handle_oov(self, embeddings, X, words):
        pass
