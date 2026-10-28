"""Ablation over retrievers / IBM templates / fusion / routing / chunking on LegalBench-RAG-mini.

Protocol (no tuning on test documents):
  dev   30% of documents per sub-dataset (235 queries): pick wsum weights and the default config
  test  the other 70% (541 queries): every number in the ablation table
  all   the 776 mini queries: the default config only, for comparison with the paper's tables
Headline metrics: R@8 (8 chunks ~ 4k chars, a realistic LLM context) and the paper's R@64; 95% bootstrap CIs.
"""

import itertools
import json
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np

from . import data, encode, metrics
from .pipeline import Config, Resources, combine, query_vectors, top_k

RESULTS = encode.ROOT / "artifacts" / "results"
K = 64

_B, _D, _S = dict(dense=False, splade=False), dict(bm25=None, splade=False), dict(bm25=None, dense=False)
CONFIGS = {c.name: c for c in [
    # single retrievers, IBM templates (no routing)
    Config("bm25-match", bm25="match", route=False, **_B),
    Config("bm25-best", bm25="best_fields", route=False, **_B),
    Config("bm25-most", bm25="most_fields", route=False, **_B),
    Config("bm25-cross", bm25="cross_fields", route=False, **_B),
    Config("bm25-phrase", bm25="phrase", route=False, **_B),
    Config("bm25-best+section", bm25="best_fields", fields=("text", "title", "section"), route=False, **_B),
    Config("dense", route=False, **_D),
    Config("splade", route=False, **_S),
    # blends as in the IBM paper (raw score sum), then rank / normalised fusion
    Config("ibm-knn-best", bm25="best_fields", splade=False, fusion="ibm", route=False),
    Config("ibm-se-best", bm25="best_fields", dense=False, fusion="ibm", route=False),
    Config("rrf-bm25+dense", splade=False, route=False),
    Config("rrf-bm25+splade", dense=False, route=False),
    Config("rrf-all", route=False),
    Config("wsum-all", fusion="wsum", route=False),  # weights filled in from dev
    # + document routing
    Config("bm25-match+route", bm25="match", **_B),
    Config("dense+route", **_D),
    Config("rrf-all+route"),
    Config("wsum-all+route", fusion="wsum"),
    # leave-one-out: what each retriever adds to the routed blend
    Config("rrf-all+route-bm25", bm25=None),  # = the default: BM25 routes, dense + SPLADE rank clauses
    Config("rrf-all+route-dense", dense=False),
    Config("rrf-all+route-splade", splade=False),
    # chunking / text variant
    Config("rrf-all+route@naive500-plain", chunking="naive", variant="plain"),
    Config("rrf-all+route@structure500-plain", variant="plain"),
    Config("rrf-all+route@structure800", max_chars=800),
    Config("dense@naive500-plain", chunking="naive", variant="plain", route=False, **_D),  # ~ paper's baseline
]}
# Chosen on dev (R@8 45.5 vs 43.6 for all three retrievers); on test +2.7 R@8, paired bootstrap p = 0.002.
DEFAULT = "rrf-all+route-bm25"
# One row per claim in the write-up: baselines, IBM template, blending, routing, per-retriever leave-one-out,
# chunking / paper baseline.
CORE = ["bm25-match", "bm25-most", "dense", "splade", "ibm-se-best", "rrf-all", "rrf-all+route", "dense+route",
        "rrf-all+route-bm25", "rrf-all+route-dense", "rrf-all+route-splade",
        "dense@naive500-plain", "rrf-all+route@naive500-plain"]
WEIGHT_GRID = (0.0, 0.5, 1.0, 2.0)


def _groups(cfgs: list[Config]) -> dict[tuple, list[Config]]:
    g = defaultdict(list)
    for c in cfgs:
        g[(c.chunking, c.max_chars, c.variant)].append(c)
    return g


def _has_vectors(res: Resources, need: set[str]) -> bool:
    texts = encode.chunk_texts(res.chunks, res.corpus, res.variant)
    return all(encode.has(res.name, "dense" if kind == "dense" else "sparse", encode.fingerprint(texts, model))
               for kind, model in (("dense", encode.DENSE_MODEL), ("splade", encode.SPARSE_MODEL)) if kind in need)


def run(cfgs: list[Config], corpus: dict[str, str], qs: list[data.Query], allow_encode: bool = False,
        tag: str = "") -> dict[str, list[dict]]:
    """Per-query metrics for every config; score sets are computed once per query and chunk set."""
    out = {}
    for (chunking, max_chars, variant), group in _groups(cfgs).items():
        res = Resources(corpus, chunking, max_chars, variant, tag=tag)
        need = {r for c in group for r in c.retrievers}
        if not allow_encode and not _has_vectors(res, need - {"bm25"}):
            print(f"skip {res.name}: no cached vectors (run kaggle/encode.py, or pass --allow-encode)")
            continue
        qv = query_vectors(res, [q.text for q in qs], sorted(need - {"bm25"}))
        rows = defaultdict(list)
        t = time.time()
        for i, q in enumerate(qs):
            sets = res.score_sets(q.text, qv["dense"][i] if "dense" in qv else None,
                                  qv["splade"][i] if "splade" in qv else None, need=need)
            mask = res.route_mask(q.text) if any(c.route for c in group) else None
            for c in group:
                idx = top_k(combine(c, sets, mask), K)
                ranked = [data.Snippet(res.chunks[j].doc, res.chunks[j].start, res.chunks[j].end) for j in idx]
                rows[c.name].append(metrics.per_query(ranked, q))
        ms = (time.time() - t) / len(qs) * 1000
        print(f"{res.name}: {len(group)} configs x {len(qs)} queries, {ms:.0f} ms/query for the whole group")
        out.update(rows)
    return out


def tune_weights(corpus, dev: list[data.Query], base: Config, tag: str = "",
                 allow_encode: bool = False) -> tuple[tuple[str, float], ...]:
    """Grid-search wsum weights (BM25, dense, SPLADE) on dev documents, maximising R@8."""
    grid = [w for w in itertools.product(WEIGHT_GRID, repeat=3) if any(w)]
    cfgs = [replace(base, name=f"w{w}", fusion="wsum", weights=tuple(zip(("bm25", "dense", "splade"), w)))
            for w in grid]
    scores = run(cfgs, corpus, dev, allow_encode, tag)
    if not scores:
        return ()
    best = max(scores, key=lambda n: metrics.macro(scores[n], dev)["ALL"]["R@8"])
    return next(c.weights for c in cfgs if c.name == best)


def summarise(scores: dict[str, list[dict]], qs: list[data.Query]) -> dict:
    table = {}
    for name, rows in scores.items():
        m = metrics.macro(rows, qs)
        entry = {"ALL": m["ALL"], "by_dataset": {d: m[d] for d in data.DATASETS if d in m}}
        for k in ("R@8", "R@64"):
            entry[f"{k}_ci"] = metrics.bootstrap_ci(np.array([r[k] for r in rows]), qs)
        table[name] = entry
    return table


def markdown(table: dict, title: str) -> str:
    cols = ("P@1", "R@1", "R@8", "R@64", "doc@1", "MRR", "nDCG@10")
    lines = [f"### {title}", "", "| config | " + " | ".join(cols) + " | R@8 95% CI | R@64 by dataset (priv/nli/maud/cuad) |",
             "|---" * (len(cols) + 3) + "|"]
    for name, e in table.items():
        a = e["ALL"]
        lo, hi = e["R@8_ci"]
        by = "/".join(f"{100 * e['by_dataset'][d]['R@64']:.0f}" for d in data.DATASETS if d in e["by_dataset"])
        lines.append(f"| {name} | " + " | ".join(f"{100 * a[c]:.1f}" for c in cols) +
                     f" | {100 * lo:.1f}–{100 * hi:.1f} | {by} |")
    return "\n".join(lines) + "\n"


def evaluate(names: list[str], split: str = "mini", allow_encode: bool = False, datasets=None,
             label: str = "custom") -> dict:
    corpus, qs = data.load(split)
    tag = ""
    if datasets:  # development subset: smaller corpus, separate vector cache
        qs = [q for q in qs if q.dataset in datasets]
        corpus = {d: t for d, t in corpus.items() if d.split("/")[0] in datasets}
        tag = "+".join(datasets)
    dev, test = data.dev_test_split(qs)
    cfgs = [CONFIGS[n] for n in names]
    if any(c.fusion == "wsum" for c in cfgs):
        w = tune_weights(corpus, dev, CONFIGS["rrf-all"], tag, allow_encode)
        print(f"wsum weights tuned on dev: {dict(w)}")
        cfgs = [replace(c, weights=w) if c.fusion == "wsum" else c for c in cfgs]
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {}
    for part, part_qs in (("test", test), ("all", qs)):
        part_cfgs = cfgs if part == "test" else [c for c in cfgs if c.name == DEFAULT]
        if not part_cfgs:
            continue
        table = summarise(run(part_cfgs, corpus, part_qs, allow_encode, tag), part_qs)
        report[part] = table
        print(markdown(table, f"{split}{'/' + tag if tag else ''} {part} ({len(part_qs)} queries)"))
    out = RESULTS / f"{split}{'@' + tag if tag else ''}.{label}.json"  # e.g. mini.core.json, mini.default.json
    out.write_text(json.dumps(report, indent=1))
    out.with_suffix(".md").write_text("\n".join(markdown(t, f"{split} {p}") for p, t in report.items()))
    print(f"wrote {out.relative_to(encode.ROOT)} and .md")
    return report
