import numpy as np
import pytest

from blended_rag import fusion


def test_rrf_rewards_agreement_over_one_strong_vote():
    a = np.array([3.0, 2.0, 1.0, 0.0])   # ranks chunk 0 first
    b = np.array([0.0, 2.0, 3.0, 1.0])   # ranks chunk 2 first
    fused = fusion.rrf({"a": a, "b": b})
    assert np.argmax(fused) in (1, 2) and fused[1] > fused[3]


def test_rrf_ignores_zero_scores_and_is_scale_free():
    a = np.array([5.0, 0.0, 1.0])
    assert fusion.rrf({"a": a})[1] == 0.0
    assert np.allclose(fusion.rrf({"a": a, "b": a * 1000}), fusion.rrf({"a": a, "b": a}))


def test_wsum_normalises_scales_and_weights():
    bm25 = np.array([30.0, 10.0, 20.0])
    dense = np.array([0.1, 0.9, 0.5])
    assert np.argmax(fusion.wsum({"bm25": bm25, "dense": dense}, {"bm25": 1.0, "dense": 0.0})) == 0
    assert np.argmax(fusion.wsum({"bm25": bm25, "dense": dense}, {"bm25": 0.0, "dense": 1.0})) == 1


def test_wsum_handles_constant_scores():
    assert np.all(fusion.wsum({"a": np.ones(4)}, {"a": 1.0}) == 0)


def test_single_retriever_passes_through():
    s = np.array([1.0, 3.0])
    assert fusion.fuse("rrf", {"bm25": s}) is s


def test_ibm_sum_boosts_dense_and_splade_requires_lexical_match():
    bm25 = np.array([2.0, 0.0, 1.0])
    fused = fusion.fuse("ibm", {"bm25": bm25, "dense": np.array([0.1, 0.9, 0.3])})
    assert fused == pytest.approx([3.0, 9.0, 4.0])
    fused = fusion.fuse("ibm", {"bm25": bm25, "splade": np.array([1.0, 50.0, 1.0])})
    assert fused[1] == -np.inf and fused[0] == 3.0
