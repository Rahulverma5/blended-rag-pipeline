"""CLI entry point. Run `python src/main.py --help`."""

import argparse
import json
import shutil
import subprocess
import time

import numpy as np

from blended_rag import data, encode
from blended_rag.evaluate import CONFIGS, DEFAULT


def _retriever(args):
    from blended_rag.pipeline import BlendedRetriever

    corpus, _ = data.load(args.split)
    return BlendedRetriever(corpus, CONFIGS[args.config], rebuild=getattr(args, "rebuild", False))


def cmd_download(args: argparse.Namespace) -> None:
    encode.FastEmbedEncoder()
    print(f"ONNX models (bge-small, SPLADE++) in {encode.MODELS}")
    if args.full:
        data.download_full()
        print(f"full LegalBench-RAG in {data.FULL}")


def cmd_index(args: argparse.Namespace) -> None:
    t = time.time()
    r = _retriever(args)
    for kind in r.cfg.retrievers:
        getattr(r.res, kind)  # builds BM25, loads (or encodes) dense / SPLADE vectors
    if r.cfg.route:
        r.res.router
    print(f"{args.split}: {len(r.res.corpus)} docs, {len(r.res.chunks)} chunks ({r.res.name}), "
          f"retrievers {r.cfg.retrievers}, ready in {time.time() - t:.1f}s")


def cmd_search(args: argparse.Namespace) -> None:
    from blended_rag.pipeline import dumps

    print(dumps(_retriever(args).search(args.query, args.k)))


ASK_PROMPT = """Answer the question about the contract using ONLY the numbered excerpts. Cite excerpts like [2].
If the excerpts do not contain the answer, say so.

Question: {query}

{context}"""


def cmd_ask(args: argparse.Namespace) -> None:
    hits = _retriever(args).search(args.query, args.k)
    context = "\n\n".join(f"[{i + 1}] ({h.chunk.doc}, chars {h.chunk.start}-{h.chunk.end}, {h.chunk.section})\n"
                          f"{h.text}" for i, h in enumerate(hits))
    if args.llm == "claude" and shutil.which("claude"):
        out = subprocess.run(["claude", "-p", ASK_PROMPT.format(query=args.query, context=context)],
                             capture_output=True, text=True, timeout=120)
        print(out.stdout.strip() or out.stderr.strip())
        print("\n--- sources ---")
    elif args.llm == "claude":
        print("(`claude` CLI not found: showing the retrieved evidence only)\n")
    print(context)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from blended_rag.evaluate import evaluate

    from blended_rag.evaluate import CORE

    presets = {"all": list(CONFIGS), "core": CORE, "default": [DEFAULT]}
    names = [n for c in args.configs for n in presets.get(c, [c])]
    evaluate(names, args.split, args.allow_encode, args.datasets, label="+".join(args.configs))


def cmd_benchmark(args: argparse.Namespace) -> None:
    import resource

    corpus, qs = data.load(args.split)
    t = time.time()
    r = _retriever(args)
    r.search(qs[0].text)  # warm-up: loads models and vectors
    build = time.time() - t
    lat = []
    for q in qs[:args.n]:
        t = time.perf_counter()
        r.search(q.text, 8)
        lat.append((time.perf_counter() - t) * 1000)
    vec_mb = sum(p.stat().st_size for p in encode.VECTORS.glob(f"{r.res.name}.*")) / 1e6
    print(json.dumps({"config": args.config, "chunks": len(r.res.chunks), "load_s": round(build, 1),
                      "query_ms_p50": round(float(np.percentile(lat, 50)), 1),
                      "query_ms_p95": round(float(np.percentile(lat, 95)), 1),
                      "vectors_mb_on_disk": round(vec_mb, 1),
                      "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Blended RAG for legal contract retrieval (LegalBench-RAG)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch the ONNX models (~650 MB); --full also fetches the 714-doc corpus")
    p.add_argument("--full", action="store_true")
    p.set_defaults(func=cmd_download)

    def common(p, default_k=None):
        p.add_argument("--split", default="mini", choices=["mini", "full"])
        p.add_argument("--config", default=DEFAULT, choices=list(CONFIGS))
        if default_k:
            p.add_argument("-k", type=int, default=default_k, help="number of chunks to return")

    p = sub.add_parser("index", help="build BM25 + router, load cached chunk vectors (or encode them on CPU)")
    common(p)
    p.add_argument("--rebuild", action="store_true", help="re-encode chunks even if cached vectors match")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="top-k cited spans for a question, as JSON")
    p.add_argument("query")
    common(p, 5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="retrieve evidence; with --llm claude, also a cited answer via `claude -p`")
    p.add_argument("query")
    common(p, 5)
    p.add_argument("--llm", default="none", choices=["none", "claude"])
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("evaluate", help="ablation table on the dev/test split (artifacts/results/)")
    p.add_argument("--split", default="mini", choices=["mini", "full"])
    p.add_argument("--configs", nargs="+", default=["default"], help="config names, or presets: default, core, all")
    p.add_argument("--datasets", nargs="+", choices=list(data.DATASETS), help="evaluate on a subset only")
    p.add_argument("--allow-encode", action="store_true", help="encode missing chunk vectors on CPU (slow)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("benchmark", help="CPU latency / memory / index size for one config")
    common(p)
    p.add_argument("-n", type=int, default=100, help="number of queries to time")
    p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
