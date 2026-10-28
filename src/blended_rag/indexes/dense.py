"""Dense index: exact inner product over L2-normalised bge-small vectors (~23k x 384 on mini; numpy is enough)."""

import numpy as np


class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def scores(self, query_vec: np.ndarray) -> np.ndarray:
        return self.vectors @ query_vec
