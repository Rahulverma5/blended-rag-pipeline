"""Retrieval metrics.

  * char P@k / R@k   LegalBench-RAG's metric: character overlap between the top-k retrieved spans and the
                     ground-truth spans (upstream legalbenchrag/run_benchmark.py). One fix: retrieved spans are
                     merged into a union first; upstream sums overlaps per chunk, so overlapping chunks are
                     double-counted and recall can exceed 1.
  * doc-hit@k        is any of the top-k chunks from the right document (isolates routing errors)
  * MRR, nDCG@10     a chunk is relevant if it overlaps any ground-truth span (binary relevance)
  * aggregation      macro over the 4 sub-datasets, 0.25 each, as in the paper
  * bootstrap        95% CI of the macro score, resampling queries within each sub-dataset;
                     paired bootstrap for "is config A better than config B"
"""

from collections import defaultdict

import numpy as np

from .data import DATASETS, Query, Snippet

KS = (1, 2, 4, 8, 16, 32, 64)


def merge_spans(spans: list[Snippet]) -> list[Snippet]:
    """Union of spans, per file, as sorted non-overlapping intervals."""
    by_file = defaultdict(list)
    for s in spans:
        by_file[s.file_path].append((s.start, s.end))
    merged = []
    for path, ivs in sorted(by_file.items()):
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged.append(Snippet(path, cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append(Snippet(path, cur_s, cur_e))
    return merged


def _overlap(a: Snippet, b: Snippet) -> int:
    if a.file_path != b.file_path:
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def precision_recall(retrieved: list[Snippet], gold: list[Snippet]) -> tuple[float, float]:
    retrieved, gold = merge_spans(retrieved), merge_spans(gold)
    hit = sum(_overlap(r, g) for r in retrieved for g in gold)
    total_ret = sum(r.end - r.start for r in retrieved)
    total_gold = sum(g.end - g.start for g in gold)
    return (hit / total_ret if total_ret else 0.0), (hit / total_gold if total_gold else 0.0)


def _relevant(chunk: Snippet, gold: list[Snippet]) -> bool:
    return any(_overlap(chunk, g) > 0 for g in gold)


def per_query(ranked: list[Snippet], q: Query, ks=KS) -> dict[str, float]:
    """All metrics for one query, given its ranked retrieved spans (best first)."""
    out = {}
    for k in ks:
        out[f"P@{k}"], out[f"R@{k}"] = precision_recall(ranked[:k], q.snippets)
    gold_docs = {s.file_path for s in q.snippets}
    for k in (1, 8):
        out[f"doc@{k}"] = float(any(r.file_path in gold_docs for r in ranked[:k]))
    rel = [_relevant(r, q.snippets) for r in ranked]
    first = next((i for i, x in enumerate(rel) if x), None)
    out["MRR"] = 0.0 if first is None else 1.0 / (first + 1)
    dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(rel[:10]) if x)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(q.snippets))))
    out["nDCG@10"] = dcg / idcg
    return out


def macro(scores: list[dict[str, float]], queries: list[Query]) -> dict[str, dict[str, float]]:
    """Mean per sub-dataset, plus ALL = unweighted mean of the sub-dataset means (0.25 each)."""
    keys = list(scores[0])
    table = {}
    for d in DATASETS:
        rows = [s for s, q in zip(scores, queries) if q.dataset == d]
        if rows:
            table[d] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    table["ALL"] = {k: float(np.mean([table[d][k] for d in table])) for k in keys}
    return table


def _strata(queries: list[Query]) -> list[np.ndarray]:
    return [np.array([i for i, q in enumerate(queries) if q.dataset == d]) for d in DATASETS
            if any(q.dataset == d for q in queries)]


def bootstrap_ci(values: np.ndarray, queries: list[Query], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the macro mean of a per-query metric, stratified by sub-dataset."""
    rng = np.random.default_rng(seed)
    strata = _strata(queries)
    stats = [np.mean([values[rng.choice(ix, len(ix))].mean() for ix in strata]) for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, queries: list[Query], n: int = 2000, seed: int = 0) -> float:
    """One-sided p-value for "macro mean of a > b": share of resamples where a - b <= 0."""
    rng = np.random.default_rng(seed)
    diff = a - b
    strata = _strata(queries)
    stats = np.array([np.mean([diff[rng.choice(ix, len(ix))].mean() for ix in strata]) for _ in range(n)])
    return float((stats <= 0).mean())
