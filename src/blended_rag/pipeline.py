"""The blended retriever: route to a contract, score chunks with BM25 / dense / SPLADE, fuse, return cited spans.

  Resources   everything that depends only on the chunking + text variant: chunks, BM25 fields, vectors, router.
              Shared across configs, so an ablation over templates / fusion / routing encodes nothing twice.
  Config      one retrieval recipe (which retrievers, which IBM template, which fusion, routing on/off).
  BlendedRetriever = Resources + Config, with .search(query, k).

Chunk vectors come from artifacts/vectors/ when their fingerprint matches (committed for the default config,
or produced on Kaggle for the ablation); otherwise they are encoded here on CPU with fastembed and cached.
"""

import json
from dataclasses import dataclass, field

import numpy as np

from . import encode, fusion, queries
from .chunking import Chunk, chunk_corpus
from .data import Snippet
from .indexes.bm25 import BM25Fields
from .indexes.dense import DenseIndex
from .indexes.splade import SparseIndex
from .routing import DocRouter

QUERY_CACHE = encode.ROOT / "artifacts" / "cache"


@dataclass(frozen=True)
class Config:
    name: str
    chunking: str = "structure"
    max_chars: int = 500
    variant: str = "contextual"         # text fed to dense / SPLADE (see encode.chunk_texts)
    bm25: str | None = "most_fields"    # IBM template, or None to drop BM25
    fields: tuple[str, ...] = ("text", "title")
    dense: bool = True
    splade: bool = True
    fusion: str = "rrf"
    weights: tuple[tuple[str, float], ...] = ()
    route: bool = True

    @property
    def vectors(self) -> str:
        return f"{self.chunking}{self.max_chars}-{self.variant}"

    @property
    def retrievers(self) -> list[str]:
        return [r for r, on in (("bm25", self.bm25), ("dense", self.dense), ("splade", self.splade)) if on]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    text: str

    @property
    def snippet(self) -> Snippet:
        return Snippet(self.chunk.doc, self.chunk.start, self.chunk.end)

    def as_dict(self) -> dict:
        c = self.chunk
        return {"doc": c.doc, "span": [c.start, c.end], "section": c.section, "score": round(self.score, 4),
                "text": self.text}


@dataclass
class Resources:
    corpus: dict[str, str]
    chunking: str
    max_chars: int
    variant: str
    rebuild: bool = False
    tag: str = ""  # set for a corpus subset, so its vectors never overwrite the full split's cache
    _cache: dict = field(default_factory=dict)

    def __post_init__(self):
        self.chunks = chunk_corpus(self.corpus, self.chunking, self.max_chars)
        docs = sorted(self.corpus)
        self.doc_ids = {d: i for i, d in enumerate(docs)}
        self.chunk_doc = np.array([self.doc_ids[c.doc] for c in self.chunks], dtype=int)

    def _get(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def encoder(self) -> encode.FastEmbedEncoder:
        return self._get("encoder", encode.FastEmbedEncoder)

    @property
    def bm25(self) -> BM25Fields:
        return self._get("bm25", lambda: BM25Fields(self.chunks, self.corpus))

    @property
    def router(self) -> DocRouter:
        return self._get("router", lambda: DocRouter(self.corpus))

    @property
    def name(self) -> str:
        return f"{self.chunking}{self.max_chars}-{self.variant}" + (f"@{self.tag}" if self.tag else "")

    def _vectors(self, kind: str):
        name = self.name
        model = encode.DENSE_MODEL if kind == "dense" else encode.SPARSE_MODEL
        texts = encode.chunk_texts(self.chunks, self.corpus, self.variant)
        fp = encode.fingerprint(texts, model)
        vecs = None if self.rebuild else encode.load(name, kind, fp)
        if vecs is None:
            print(f"encoding {len(texts)} chunks ({name}, {kind}) on CPU; slow on old laptops, see kaggle/README.md")
            vecs = self.encoder.encode_dense(texts) if kind == "dense" else self.encoder.encode_sparse(texts)
            encode.save(name, kind, vecs, fp)
        return vecs

    @property
    def dense(self) -> DenseIndex:
        return self._get("dense", lambda: DenseIndex(self._vectors("dense")))

    @property
    def splade(self) -> SparseIndex:
        return self._get("splade", lambda: SparseIndex(self._vectors("sparse")))

    def score_sets(self, query: str, qdense=None, qsparse=None, need=("bm25", "dense", "splade")) -> dict:
        """Raw scores of every chunk: BM25 per field, dense, SPLADE (only those in `need`)."""
        out = {}
        if "bm25" in need:
            out["bm25_fields"] = self.bm25.scores(query)
        if "dense" in need:
            out["dense"] = self.dense.scores(self.encoder.query_dense([query])[0] if qdense is None else qdense)
        if "splade" in need:
            out["splade"] = self.splade.scores(self.encoder.query_sparse([query]) if qsparse is None else qsparse)
        return out

    def route_mask(self, query: str) -> np.ndarray:
        routed = [self.doc_ids[d] for d in self.router.route(query)]
        return np.isin(self.chunk_doc, routed)


def combine(cfg: Config, sets: dict, mask: np.ndarray | None = None) -> np.ndarray:
    """Fused chunk scores for one config, from precomputed score sets."""
    parts = {}
    if cfg.bm25:
        parts["bm25"] = queries.apply(cfg.bm25, sets["bm25_fields"], cfg.fields)
    for r in ("dense", "splade"):
        if getattr(cfg, r):
            parts[r] = sets[r]
    fused = fusion.fuse(cfg.fusion, parts, dict(cfg.weights) or None).astype(np.float64)
    if cfg.route and mask is not None:
        fused = np.where(mask, fused, -np.inf)
    return fused


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, int(np.isfinite(scores).sum()))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


def query_vectors(res: Resources, texts: list[str], need: list[str]) -> dict:
    """Encode all queries once (cached on disk): ~1 min for the 776 mini queries on CPU."""
    QUERY_CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, model in (("dense", encode.DENSE_MODEL), ("splade", encode.SPARSE_MODEL)):
        if kind not in need:
            continue
        path = QUERY_CACHE / f"queries.{kind}.{encode.fingerprint(texts, model)}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            out[kind] = z["v"] if kind == "dense" else _csr(z)
            continue
        if kind == "dense":
            v = res.encoder.query_dense(texts)
            np.savez(path, v=v)
        else:
            v = res.encoder.query_sparse(texts)
            np.savez(path, data=v.data, indices=v.indices, indptr=v.indptr, shape=np.array(v.shape))
        out[kind] = v
    return out


def _csr(z):
    import scipy.sparse as sp

    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


class BlendedRetriever:
    def __init__(self, corpus: dict[str, str], cfg: Config, resources: Resources | None = None,
                 rebuild: bool = False):
        self.cfg = cfg
        self.res = resources or Resources(corpus, cfg.chunking, cfg.max_chars, cfg.variant, rebuild)

    def search(self, query: str, k: int = 8) -> list[Hit]:
        sets = self.res.score_sets(query, need=self.cfg.retrievers)
        mask = self.res.route_mask(query) if self.cfg.route else None
        scores = combine(self.cfg, sets, mask)
        return [Hit(self.res.chunks[i], float(scores[i]), self.res.corpus[self.res.chunks[i].doc][
            self.res.chunks[i].start:self.res.chunks[i].end]) for i in top_k(scores, k)]


def dumps(hits: list[Hit]) -> str:
    return json.dumps([h.as_dict() for h in hits], indent=2)
