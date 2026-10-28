import numpy as np
import scipy.sparse as sp

from blended_rag import encode
from blended_rag.chunking import Chunk

CHUNKS = [Chunk("a.txt", 0, 5, "Section 1 Term.", "Acme NDA"), Chunk("a.txt", 6, 11, "", "Acme NDA")]
CORPUS = {"a.txt": "hello world"}


def test_vectors_round_trip_and_fingerprint_guards_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(encode, "VECTORS", tmp_path)
    fp = encode.fingerprint(encode.chunk_texts(CHUNKS, CORPUS, "plain"), encode.DENSE_MODEL)
    dense = np.random.default_rng(0).random((2, 4)).astype(np.float32)
    sparse = sp.csr_matrix(np.array([[0, 1.5, 0], [2.25, 0, 0]], dtype=np.float32))
    encode.save("x", "dense", dense, fp)
    encode.save("x", "sparse", sparse, fp)
    assert np.allclose(encode.load("x", "dense", fp), dense, atol=1e-3)
    assert (encode.load("x", "sparse", fp) != sparse).nnz == 0
    other = encode.fingerprint(encode.chunk_texts(CHUNKS, CORPUS, "contextual"), encode.DENSE_MODEL)
    assert encode.load("x", "dense", other) is None


def test_contextual_text_prefixes_title_and_section():
    plain, ctx = encode.chunk_texts(CHUNKS, CORPUS, "plain"), encode.chunk_texts(CHUNKS, CORPUS, "contextual")
    assert plain == ["hello", "world"]
    assert ctx == ["Acme NDA | Section 1 Term.\nhello", "Acme NDA\nworld"]
