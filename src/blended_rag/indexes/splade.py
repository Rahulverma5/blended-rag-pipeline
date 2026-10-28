"""Learned-sparse index (SPLADE++), standing in for IBM's ELSER `text_expansion` query.

Chunks are rows of a CSR matrix over the 30522-term BERT vocabulary; a query is a sparse vector over the same
vocabulary, and the score is their dot product, i.e. an inverted-index lookup with learned term weights.
"""

import numpy as np
import scipy.sparse as sp


class SparseIndex:
    def __init__(self, vectors: sp.csr_matrix):
        self.vectors = vectors

    def scores(self, query_vec: sp.csr_matrix) -> np.ndarray:
        return np.asarray((self.vectors @ query_vec.T).todense()).ravel()
