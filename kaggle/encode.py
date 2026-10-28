"""Encode LegalBench-RAG-mini chunks on a Kaggle GPU (see kaggle/README.md).

    python kaggle/encode.py --out /kaggle/working

Writes <out>/vectors/{name}.{dense.npy,sparse.npz}, fingerprints.json and parity.json, then <out>/vectors.zip.
Copy the unzipped files to artifacts/vectors/ in the repo; `index` picks them up by fingerprint.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blended_rag import chunking, data, encode  # noqa: E402

CONFIGS = ("naive:500", "structure:500", "structure:800")
VARIANTS = ("plain", "contextual")


def parity(gpu: encode.TorchEncoder, texts: list[str]) -> dict:
    """Same chunks through torch (here) and fastembed ONNX (what the grader uses for queries)."""
    try:
        cpu = encode.FastEmbedEncoder()
    except ImportError:
        return {"skipped": "fastembed not installed"}
    d_gpu, d_cpu = gpu.encode_dense(texts), cpu.encode_dense(texts)
    s_gpu, s_cpu = gpu.encode_sparse(texts).toarray(), cpu.encode_sparse(texts).toarray()
    s_cos = (s_gpu * s_cpu).sum(1) / (np.linalg.norm(s_gpu, axis=1) * np.linalg.norm(s_cpu, axis=1))
    return {"dense_cos_min": float((d_gpu * d_cpu).sum(1).min()), "sparse_cos_min": float(s_cos.min())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/kaggle/working")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--smoke", action="store_true", help="20 chunks, small batches: check the script on a CPU")
    args = p.parse_args()
    batch = {"dense": 4, "sparse": 2} if args.smoke else {"dense": 256, "sparse": 64}

    out = Path(args.out)
    encode.VECTORS = out / "vectors"
    encode.MODELS = out / "models"  # /kaggle/input is read-only
    corpus, _ = data.load_mini()
    enc = encode.TorchEncoder()
    print(f"device: {enc.device}")
    log = {}
    for cfg in args.configs:
        strategy, max_chars = cfg.split(":")
        chunks = chunking.chunk_corpus(corpus, strategy, int(max_chars))[:20 if args.smoke else None]
        for variant in args.variants:
            name = f"{strategy}{max_chars}-{variant}"
            texts = encode.chunk_texts(chunks, corpus, variant)
            for kind, fn, model in (("dense", enc.encode_dense, encode.DENSE_MODEL),
                                    ("sparse", enc.encode_sparse, encode.SPARSE_MODEL)):
                t = time.time()
                vecs = fn(texts, batch_size=batch[kind])
                encode.save(name, kind, vecs, encode.fingerprint(texts, model))
                log[f"{name}.{kind}"] = {"chunks": len(chunks), "seconds": round(time.time() - t, 1)}
                print(name, kind, log[f"{name}.{kind}"], flush=True)
        if cfg == args.configs[0]:
            log["parity"] = parity(enc, encode.chunk_texts(chunks, corpus, "contextual")[:8 if args.smoke else 32])
            print("parity", log["parity"], flush=True)
    (encode.VECTORS / "encode_log.json").write_text(json.dumps(log, indent=1))
    shutil.make_archive(str(out / "vectors"), "zip", encode.VECTORS)
    print(f"wrote {out / 'vectors.zip'}")


if __name__ == "__main__":
    main()
