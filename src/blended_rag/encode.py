"""Encode chunks with the dense and SPLADE models; cache vectors with a fingerprint.

Two backends compute the same function (checked by `parity` in kaggle/encode.py):
  fastembed  ONNX on CPU. Used for queries (always) and for chunks when the cache is missing.
  torch      transformers on GPU. Used on Kaggle to encode chunks; ~100x faster than this laptop's CPU.
  dense   = bge-small-en-v1.5, CLS pooling, L2-normalised, max 512 tokens
  sparse  = SPLADE++ (prithivida/Splade_PP_en_v1), max over tokens of log(1 + relu(MLM logits))

Text variants:
  plain       chunk text only
  contextual  "title | section\n" + chunk text, so a chunk knows which contract and clause it belongs to
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .chunking import Chunk

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge's retrieval instruction
MAX_LEN = 512
ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "artifacts" / "vectors"
MODELS = ROOT / "data" / "models"


def chunk_texts(chunks: list[Chunk], corpus: dict[str, str], variant: str) -> list[str]:
    texts = [corpus[c.doc][c.start:c.end] for c in chunks]
    if variant == "plain":
        return texts
    if variant == "contextual":
        return [f"{c.title} | {c.section}\n{t}" if c.section else f"{c.title}\n{t}" for c, t in zip(chunks, texts)]
    raise ValueError(f"unknown text variant {variant!r}")


def fingerprint(texts: list[str], model: str) -> str:
    """Hash of the exact encoder inputs, so a change to chunking, titles or the text variant invalidates the cache."""
    h = hashlib.sha256(f"{model}|{MAX_LEN}".encode())
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _path(name: str, kind: str) -> Path:
    return VECTORS / f"{name}.{kind}.{'npy' if kind == 'dense' else 'npz'}"


def save(name: str, kind: str, vectors, fp: str) -> None:
    VECTORS.mkdir(parents=True, exist_ok=True)
    if kind == "dense":
        np.save(_path(name, kind), vectors.astype(np.float16))
    else:
        # scipy can't store float16; vocab ids fit uint16 (30522), so this is ~4 bytes per non-zero.
        m = vectors.tocsr()
        np.savez_compressed(_path(name, kind), indptr=m.indptr.astype(np.int64), indices=m.indices.astype(np.uint16),
                            data=m.data.astype(np.float16), shape=np.array(m.shape))
    meta = VECTORS / "fingerprints.json"
    fps = json.loads(meta.read_text()) if meta.exists() else {}
    fps[f"{name}.{kind}"] = fp
    meta.write_text(json.dumps(fps, indent=1, sort_keys=True))


def has(name: str, kind: str, fp: str) -> bool:
    meta = VECTORS / "fingerprints.json"
    return meta.exists() and json.loads(meta.read_text()).get(f"{name}.{kind}") == fp and _path(name, kind).exists()


def load(name: str, kind: str, fp: str):
    """Cached vectors, or None if missing or built from different chunks / model / variant."""
    if not has(name, kind, fp):
        return None
    if kind == "dense":
        return np.load(_path(name, kind)).astype(np.float32)
    z = np.load(_path(name, kind))
    return sp.csr_matrix((z["data"].astype(np.float32), z["indices"].astype(np.int32), z["indptr"]),
                         shape=tuple(z["shape"]))


# ---------------------------------------------------------------- fastembed (CPU, ONNX)

def _sparse_rows(embs, dim: int) -> sp.csr_matrix:
    rows, cols, vals = [], [], []
    for i, e in enumerate(embs):
        rows.extend([i] * len(e.indices))
        cols.extend(e.indices)
        vals.extend(e.values)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(embs), dim), dtype=np.float32)


SPARSE_DIM = 30522  # bert-base-uncased vocabulary


class FastEmbedEncoder:
    def __init__(self, threads: int | None = None):
        from fastembed import SparseTextEmbedding, TextEmbedding

        self.dense = TextEmbedding(DENSE_MODEL, cache_dir=str(MODELS), threads=threads)
        self.sparse = SparseTextEmbedding(SPARSE_MODEL, cache_dir=str(MODELS), threads=threads)

    def encode_dense(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        return np.stack(list(self.dense.embed(texts, batch_size=batch_size))).astype(np.float32)

    def encode_sparse(self, texts: list[str], batch_size: int = 1) -> sp.csr_matrix:
        # batch_size 1: the MLM head emits batch x tokens x 30522 floats; batch 16 was OOM-killed at 3.4 GB here.
        return _sparse_rows(list(self.sparse.embed(texts, batch_size=batch_size)), SPARSE_DIM)

    def query_dense(self, queries: list[str]) -> np.ndarray:
        return self.encode_dense([QUERY_PREFIX + q for q in queries])

    def query_sparse(self, queries: list[str]) -> sp.csr_matrix:
        # query_embed batches 256 at a time (256 x tokens x 30522 floats): that swapped for an hour, then OOM.
        return _sparse_rows(list(self.sparse.embed(queries, batch_size=4)), SPARSE_DIM)


# ---------------------------------------------------------------- torch (Kaggle GPU)

class TorchEncoder:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dense_tok = AutoTokenizer.from_pretrained(DENSE_MODEL)
        self.dense_model = AutoModel.from_pretrained(DENSE_MODEL).to(self.device).eval()
        self.sparse_tok = AutoTokenizer.from_pretrained(SPARSE_MODEL)
        self.sparse_model = AutoModelForMaskedLM.from_pretrained(SPARSE_MODEL).to(self.device).eval()

    def _batches(self, tok, texts, batch_size):
        # Sort by length so each batch pads little; yield original indices to restore order.
        order = np.argsort([len(t) for t in texts])
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            yield idx, {k: v.to(self.device) for k, v in enc.items()}

    def encode_dense(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = np.zeros((len(texts), self.dense_model.config.hidden_size), dtype=np.float32)
        with self.torch.inference_mode():
            for idx, enc in self._batches(self.dense_tok, texts, batch_size):
                cls = self.dense_model(**enc).last_hidden_state[:, 0]
                out[idx] = self.torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy()
        return out

    def encode_sparse(self, texts: list[str], batch_size: int = 64) -> sp.csr_matrix:
        parts, order = [], []
        with self.torch.inference_mode():
            for idx, enc in self._batches(self.sparse_tok, texts, batch_size):
                logits = self.sparse_model(**enc).logits
                w = self.torch.log1p(self.torch.relu(logits)) * enc["attention_mask"].unsqueeze(-1)
                parts.append(sp.csr_matrix(w.max(dim=1).values.float().cpu().numpy()))
                order.extend(idx)
        mat = sp.vstack(parts).tocsr()
        inv = np.empty(len(order), dtype=int)
        inv[np.array(order)] = np.arange(len(order))
        return mat[inv]
