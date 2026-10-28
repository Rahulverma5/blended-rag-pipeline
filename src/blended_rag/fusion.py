"""Combine per-chunk score vectors from several retrievers into one ranking.

  ibm   IBM Blended-RAG / Elasticsearch: raw scores add up, with a boost on the semantic part
        (knn_best: BM25 best_fields + 10 * kNN; se_best: ELSER + BM25 best_fields, BM25 `must` match).
        Raw BM25 and SPLADE scores are unbounded, so the boost is scale-dependent; kept as the faithful baseline.
  rrf   reciprocal rank fusion, sum of 1 / (60 + rank). Scale-free, no tuning.
  wsum  weighted sum of min-max normalised scores over each retriever's top `depth`; weights tuned on dev.
"""

import numpy as np

RRF_K = 60
DEPTH = 1000


def _top(scores: np.ndarray, depth: int) -> np.ndarray:
    depth = min(depth, len(scores))
    return np.argpartition(-scores, depth - 1)[:depth]


def rrf(score_sets: dict[str, np.ndarray], weights: dict[str, float] | None = None, depth: int = DEPTH) -> np.ndarray:
    n = len(next(iter(score_sets.values())))
    fused = np.zeros(n)
    for name, s in score_sets.items():
        top = _top(s, depth)
        top = top[np.argsort(-s[top], kind="stable")]
        top = top[s[top] > 0]  # a retriever that scores a chunk 0 (no term overlap) does not vote for it
        fused[top] += (weights or {}).get(name, 1.0) / (RRF_K + 1 + np.arange(len(top)))
    return fused


def wsum(score_sets: dict[str, np.ndarray], weights: dict[str, float], depth: int = DEPTH) -> np.ndarray:
    n = len(next(iter(score_sets.values())))
    fused = np.zeros(n)
    for name, s in score_sets.items():
        top = _top(s, depth)
        lo, hi = s[top].min(), s[top].max()
        if hi > lo:
            fused[top] += weights.get(name, 0.0) * (s[top] - lo) / (hi - lo)
    return fused


def ibm(lexical: np.ndarray, semantic: dict[str, np.ndarray], boosts: dict[str, float],
        require_lexical: bool = False) -> np.ndarray:
    fused = lexical + sum(boosts.get(k, 1.0) * v for k, v in semantic.items())
    if require_lexical:  # se_best puts BM25 in `must`: chunks with no query term are dropped
        fused = np.where(lexical > 0, fused, -np.inf)
    return fused


def fuse(method: str, score_sets: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    if len(score_sets) == 1:
        return next(iter(score_sets.values()))
    if method == "rrf":
        return rrf(score_sets, weights)
    if method == "wsum":
        return wsum(score_sets, weights or {k: 1.0 for k in score_sets})
    if method == "ibm":
        semantic = {k: v for k, v in score_sets.items() if k != "bm25"}
        return ibm(score_sets["bm25"], semantic, weights or {"dense": 10.0, "splade": 1.0},
                   require_lexical="splade" in semantic)
    raise ValueError(f"unknown fusion method {method!r}")
