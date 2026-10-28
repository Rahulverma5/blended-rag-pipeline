# Problem 2.2: Blended RAG for contract review

Retrieve the exact clause that answers a question about a specific contract, such as "Where is the No-Shop clause in the BridgeBio / Eidos merger agreement?" The system follows IBM's [Blended RAG](https://arxiv.org/abs/2404.07220): BM25, dense and learned-sparse retrievers, multi-field query templates and fusion. It is rebuilt in-process with no Elasticsearch and runs on a CPU. It is evaluated on [LegalBench-RAG](https://arxiv.org/abs/2408.10343), which covers NDAs, M&A agreements, commercial contracts and privacy policies, with character-level ground-truth spans.

The write-up is in [docs/](docs/).

## Run

```bash
pip install -r requirements.txt
python src/main.py download                  # two ONNX models, ~650 MB
python src/main.py index                     # loads committed vectors, ~2 s
python src/main.py evaluate --configs default
python src/main.py search "Consider the Acquisition Agreement between Parent 'BridgeBio Pharma, Inc.' and Target 'Eidos Therapeutics, Inc.'; Where is the No-Shop Clause" -k 3
python src/main.py ask "<question>" --llm claude   # optional: cited answer via `claude -p`
pytest -q
```

Everything runs on a CPU with no API key. The mini corpus and the default config's chunk vectors are committed.

## Pipeline

1. **Chunk.** Contracts are cut at section headings into spans of at most 500 characters, with no overlap. Each chunk keeps its exact character offsets for citation.
2. **Route.** BM25 scores whole documents to pick the contract the question names. A document's score is the rarest query word that appears in its name, plus BM25 over its preamble. It picks the right contract for 99.6% of queries.
3. **Rank** chunks inside the routed contract with two retrievers fused by reciprocal rank fusion:
   - `bge-small-en-v1.5` dense vectors;
   - `SPLADE++` sparse vectors, standing in for IBM's ELSER.

   Both encode `title | section` + chunk text.
4. **Return** the top-k spans with document, offsets and section heading.

## Results

Evaluated on LegalBench-RAG-mini (776 queries, 72 documents, 8.7M characters), split by document into dev (30%) and test (70%). The default config was chosen on dev. P = precision, R = recall, computed on character overlap; ALL is the mean of the four sub-datasets.

| test, 541 queries | P@1 | R@8 | R@64 | doc@1 |
|---|---|---|---|---|
| BM25, `match` | 2.5 | 10.1 | 30.8 | 76.1 |
| BM25, IBM `most_fields` [text, title] | 2.9 | 16.2 | 61.1 | 86.0 |
| dense | 11.4 | 35.6 | 75.3 | 95.0 |
| SPLADE | 13.3 | 30.1 | 75.4 | 94.3 |
| IBM blend (SPLADE + BM25 best_fields, score sum) | 8.3 | 29.9 | 75.0 | 94.1 |
| RRF: BM25 + dense + SPLADE | 9.7 | 33.9 | 79.0 | 95.2 |
| + routing | 10.8 | 38.0 | 82.5 | 99.4 |
| **+ routing, dense + SPLADE (default)** | **14.4** | **40.6** | **82.0** | **99.4** |

On all 776 mini queries the default scores P@1 13.5, R@64 81.9. The paper's best (OpenAI `text-embedding-3-large`) is P@1 6.4 and R@64 62.2 with RCTS chunking, or 76.4 with naive chunking.

- **CPU latency:** 165 ms p50 / 198 ms p95 per query on an i5-10210U, including query encoding.
- **Memory and disk:** 0.95 GB peak RAM, 26 MB of vectors.

The full ablation, with leave-one-out, chunking, 95% bootstrap CIs and per-dataset numbers, is in [artifacts/results/mini.core.md](artifacts/results/mini.core.md).

## Layout

| Path | Contents |
|---|---|
| `src/blended_rag/data.py` | download, exact mini split, dev/test split |
| `chunking.py` | structure-aware chunker |
| `indexes/` | BM25 fields, dense, SPLADE |
| `queries.py` | IBM query templates |
| `fusion.py` | RRF, weighted sum, IBM score sum |
| `routing.py` | document router |
| `pipeline.py` | retriever |
| `metrics.py` | LegalBench-RAG metrics |
| `evaluate.py` | ablation |
| `kaggle/` | GPU encoding of chunk vectors (Kaggle T4); a parity check confirms it matches the CPU encoder |
| `tests/` | 35 tests; offline end to end on a fixture corpus |

## Data and credits

- **LegalBench-RAG** (Pipitone & Houir Alami, 2024), built from:
  - CUAD, MAUD and ContractNLI (CC BY 4.0);
  - PrivacyQA.
- `data/mini/` holds the mini subset, selected exactly as in the authors' `benchmark.py`.
- Method from **Blended RAG** (Sawarkar et al., IBM, 2024).
