"""LegalBench-RAG corpus and queries.

Full  = 714 contracts / privacy policies (79.7M chars), 6,889 queries; downloaded from the authors' Dropbox.
Mini  = LegalBench-RAG-mini: 194 queries per sub-dataset (776), 72 docs (8.68M chars), selected exactly as in
        upstream legalbenchrag/benchmark.py. Committed under data/mini/ so the grader needs no data download.

Every query has a list of ground-truth snippets: (file_path, char_start, char_end) into one corpus document.
"""

import gzip
import json
import os
import random
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
FULL = RAW / "LegalBench-RAG"
MINI = DATA / "mini"

ZIP_URL = ("https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4"
           "?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1")
DATASETS = ("privacy_qa", "contractnli", "maud", "cuad")  # upstream order; each weighted 0.25
MINI_PER_DATASET = 194


@dataclass(frozen=True)
class Snippet:
    file_path: str
    start: int
    end: int


@dataclass
class Query:
    qid: str
    text: str
    dataset: str
    snippets: list[Snippet] = field(default_factory=list)

    @property
    def doc(self) -> str:
        return self.snippets[0].file_path


def download_full() -> None:
    """Fetch and extract the full benchmark into data/raw/LegalBench-RAG (idempotent)."""
    if (FULL / "benchmarks").exists():
        return
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "LegalBench-RAG.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(ZIP_URL, zip_path)
    safe_extract(zip_path, FULL)


def safe_extract(zip_path: Path, dest: Path) -> None:
    # `unzip` mangles names containing ".." or "||" (e.g. "Inc..txt"), and the archive has a "/" entry,
    # so extract member by member and only reject names that actually escape dest.
    root = os.path.realpath(dest)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            target = os.path.realpath(os.path.join(root, name))
            if name.strip("/") and target.startswith(root + os.sep) and ".." not in name.split("/"):
                z.extract(name, root)


def _read_benchmark(path: Path, dataset: str) -> list[Query]:
    tests = json.loads(path.read_text())["tests"]
    return [Query(f"{dataset}-{i}", t["query"], dataset,
                  [Snippet(s["file_path"], s["span"][0], s["span"][1]) for s in t["snippets"]])
            for i, t in enumerate(tests)]


def select_mini(queries: list[Query]) -> list[Query]:
    """Upstream mini selection: sort by a per-document seeded random key, keep the first 194.

    Grouping by document keeps the 194 queries on few documents; it must match upstream exactly
    so our numbers are comparable with the paper's tables.
    """
    if len(queries) <= MINI_PER_DATASET:
        return queries

    def key(q: Query) -> float:
        random.seed(q.snippets[0].file_path)
        return random.random()

    return sorted(queries, key=key)[:MINI_PER_DATASET]


def load_full() -> tuple[dict[str, str], list[Query]]:
    queries = [q for d in DATASETS for q in _read_benchmark(FULL / "benchmarks" / f"{d}.json", d)]
    docs = {s.file_path for q in queries for s in q.snippets}
    corpus = {p: (FULL / "corpus" / p).read_text() for p in sorted(docs) if (FULL / "corpus" / p).exists()}
    return corpus, [q for q in queries if all(s.file_path in corpus for s in q.snippets)]


def build_mini() -> None:
    """Write data/mini/{corpus.jsonl.gz, queries.json} from the full download."""
    corpus, queries = load_full()
    mini = [q for d in DATASETS for q in select_mini([q for q in queries if q.dataset == d])]
    docs = sorted({s.file_path for q in mini for s in q.snippets})
    MINI.mkdir(parents=True, exist_ok=True)
    # One gzipped JSONL instead of 72 files: several file names contain "|", which some filesystems reject.
    with gzip.open(MINI / "corpus.jsonl.gz", "wt") as f:
        for p in docs:
            f.write(json.dumps({"file_path": p, "content": corpus[p]}) + "\n")
    rows = [{"qid": q.qid, "dataset": q.dataset, "query": q.text,
             "snippets": [[s.file_path, s.start, s.end] for s in q.snippets]} for q in mini]
    (MINI / "queries.json").write_text(json.dumps(rows, indent=1))


def load_mini() -> tuple[dict[str, str], list[Query]]:
    with gzip.open(MINI / "corpus.jsonl.gz", "rt") as f:
        corpus = {r["file_path"]: r["content"] for r in map(json.loads, f)}
    rows = json.loads((MINI / "queries.json").read_text())
    queries = [Query(r["qid"], r["query"], r["dataset"], [Snippet(*s) for s in r["snippets"]]) for r in rows]
    return corpus, queries


def load(split: str) -> tuple[dict[str, str], list[Query]]:
    if split == "mini":
        return load_mini()
    if split == "full":
        download_full()
        return load_full()
    raise ValueError(f"unknown split {split!r}")


def dev_test_split(queries: list[Query], dev_frac: float = 0.3, seed: int = 0) -> tuple[list[Query], list[Query]]:
    """Split by document within each sub-dataset, so fusion weights are never tuned on a test document."""
    rng = random.Random(seed)
    dev_docs: set[str] = set()
    for d in DATASETS:
        docs = sorted({q.doc for q in queries if q.dataset == d})
        rng.shuffle(docs)
        dev_docs |= set(docs[:max(1, round(dev_frac * len(docs)))])
    dev = [q for q in queries if q.doc in dev_docs]
    test = [q for q in queries if q.doc not in dev_docs]
    return dev, test
