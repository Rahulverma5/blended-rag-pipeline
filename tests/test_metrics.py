import numpy as np
import pytest

from blended_rag import metrics
from blended_rag.data import Query, Snippet


def S(start, end, path="a.txt"):
    return Snippet(path, start, end)


def test_merge_spans_unions_overlaps_per_file():
    merged = metrics.merge_spans([S(0, 10), S(5, 20), S(30, 40), S(0, 5, "b.txt")])
    assert merged == [S(0, 20), S(30, 40), S(0, 5, "b.txt")]


def test_exact_hit_is_perfect():
    assert metrics.precision_recall([S(100, 200)], [S(100, 200)]) == (1.0, 1.0)


def test_partial_overlap():
    p, r = metrics.precision_recall([S(0, 100)], [S(50, 250)])
    assert p == pytest.approx(0.5) and r == pytest.approx(0.25)


def test_overlapping_chunks_are_not_double_counted():
    # Upstream would score recall 1.5 here: both chunks cover the whole gold span.
    p, r = metrics.precision_recall([S(0, 100), S(0, 100)], [S(0, 100)])
    assert (p, r) == (1.0, 1.0)


def test_wrong_file_scores_zero_and_empty_retrieval_is_safe():
    assert metrics.precision_recall([S(0, 100, "b.txt")], [S(0, 100)]) == (0.0, 0.0)
    assert metrics.precision_recall([], [S(0, 100)]) == (0.0, 0.0)


def test_multiple_gold_spans():
    p, r = metrics.precision_recall([S(0, 50)], [S(0, 50), S(100, 150)])
    assert p == 1.0 and r == pytest.approx(0.5)


def test_per_query_rank_metrics():
    q = Query("q", "?", "cuad", [S(100, 200)])
    ranked = [S(0, 50), S(0, 50, "b.txt"), S(150, 300)]
    m = metrics.per_query(ranked, q)
    assert m["MRR"] == pytest.approx(1 / 3)
    assert m["doc@1"] == 1.0 and m["R@1"] == 0.0
    assert m["nDCG@10"] == pytest.approx(1 / np.log2(4))
    assert m["R@4"] == pytest.approx(0.5)


def test_macro_weights_datasets_equally():
    qs = [Query("1", "", "cuad", [S(0, 1)]), Query("2", "", "cuad", [S(0, 1)]), Query("3", "", "maud", [S(0, 1)])]
    table = metrics.macro([{"x": 1.0}, {"x": 1.0}, {"x": 0.0}], qs)
    assert table["cuad"]["x"] == 1.0 and table["ALL"]["x"] == 0.5


def test_bootstrap_ci_brackets_mean_and_paired_test_detects_gap():
    qs = [Query(str(i), "", "cuad" if i % 2 else "maud", [S(0, 1)]) for i in range(200)]
    rng = np.random.default_rng(1)
    a = rng.random(200)
    lo, hi = metrics.bootstrap_ci(a, qs)
    assert lo < a.mean() < hi
    assert metrics.paired_bootstrap(a + 0.1, a, qs) == 0.0
    assert metrics.paired_bootstrap(a, a + 0.1, qs) == 1.0
