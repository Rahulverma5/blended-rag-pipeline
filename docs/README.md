# Blended RAG for Contract Review: Write-up

**In short:** we rebuilt IBM's Blended RAG as a CPU-only, two-stage retriever: find the contract, then find the clause. It runs on LegalBench-RAG-mini, 776 legal questions with character-exact answers. It retrieves 81.9% of the answer text in the top 64 chunks, against 62.2% for the benchmark authors' OpenAI-embedding baseline, at 165 ms per query on a laptop CPU. How to run it is in the [root README](../README.md).

## 1. Problem

**Real task: contract review.** A lawyer or analyst asks a question about one specific agreement, for example "Where is the No-Shop clause in the BridgeBio / Eidos merger agreement?" The system must return the exact passage, with a citation, so a person or an LLM can rely on it.

**Why it's hard:**
- **The corpus is full of near-duplicates.** Every NDA has a confidentiality clause, and every merger agreement has a No-Shop clause.
- **Matching needs both exact and loose wording.** Questions mix exact legal terms ("No-Shop", "Change of Control") with paraphrases ("can they walk away early?").
- **Documents are long.** A MAUD merger agreement averages about 350k characters, while the answer is usually about 330 characters.

**Brainstorming:**
- **Plain dense RAG fails.** It finds the right *kind* of clause, but often in the wrong contract. So the problem splits into two questions: (1) *which contract?* and (2) *which clause in it?*
- **Why blend:** Blended RAG mixes exact-word search (BM25), meaning search (dense vectors) and learned keyword expansion (SPLADE), because each catches questions the others miss.
- **Why rebuild:** IBM's code depends on Elasticsearch and its licensed ELSER model, which cannot run in a grader or on a laptop. We kept the method and rebuilt it in-process with open models.

## 2. Data

**Benchmark: LegalBench-RAG** (Pipitone & Houir Alami, 2024). It has four sub-datasets: CUAD (commercial contracts), MAUD (M&A agreements), ContractNLI (NDAs) and PrivacyQA (privacy policies). In total that is 714 documents, 79.7M characters and 6.9k questions. Each answer is given as character spans in a document, so scoring is exact and needs no LLM judge.

**Subset used: LegalBench-RAG-mini.** It has 194 questions per sub-dataset (776 total) over 72 documents (8.68M characters). We reproduced it exactly from the authors' selection code, and the counts match the paper, so our numbers compare directly with theirs.

**Split:** dev (30%, 235 questions) and test (70%, 541 questions), **grouped by document**. No test contract is ever used for tuning.

**Two observations shaped the design:**
- **Every question names its contract:** "Consider the Acquisition Agreement between Parent X and Target Y; …". This motivated the routing stage.
- **File names are messy:** `VELCO%20NDA%20rev0`, `ExcelerateStandardNDAFormat`. We normalise them into readable titles.

## 3. Model architecture

```
question ─► ROUTE: which contract?  ─► RANK: which clause?                ─► top-k cited spans
            BM25 over contract names     dense (bge-small)  ─┐               (doc, char offsets,
            + preambles                  SPLADE++ (sparse)  ─┴─ RRF fusion    section heading)
```

**Chunking.** Contracts are cut at section headings ("ARTICLE IV", "Section 7.2 …") into pieces of at most 500 characters, with no overlap. Each chunk keeps exact character offsets for citation. Before encoding, each chunk is prefixed with its context, `contract title | section heading`, so a clause "knows" which contract and section it belongs to.

**Routing.** Each document is scored as:
- **name score:** the rarest question word that appears in the contract's name;
- **plus** BM25 over the name and the first 2,000 characters (where parties are named).

Taking the *rarest* word, rather than summing all words, lets one distinctive name ("Kenway") beat generic words ("non-disclosure agreement") that match every NDA. Only the top contract(s) go on to ranking. The router picks the right contract for 99.6% of questions.

**Ranking.** Inside the routed contract, two retrievers score every chunk:
- **dense:** `bge-small-en-v1.5` cosine similarity;
- **sparse:** `SPLADE++` term-weight dot product.

Their rankings are merged by reciprocal rank fusion (RRF): each chunk scores Σ 1/(60 + rank). RRF is scale-free, so the two retrievers' incompatible scores need no calibration.

**Where BM25 went.** BM25 with IBM's multi-field query templates is used for routing, not for ranking clauses. The evaluation in section 5 shows why.

## 4. Training (setup and parameters)

**No model is trained or fine-tuned.** This is deliberate and follows the Blended RAG paper: pretrained retrievers are used zero-shot, so the system transfers to any contract corpus without labelled data. "Training" here means fixing the parameters below and selecting one configuration on the dev split.

| Component | Setting |
|---|---|
| Dense model | `BAAI/bge-small-en-v1.5` (33M params), CLS pooling, max 512 tokens, bge query instruction |
| Sparse model | `prithivida/Splade_PP_en_v1` (~110M params), max 512 tokens |
| BM25 | `bm25s`, English stopwords, Snowball stemmer; fields: text, title, section |
| IBM templates | match / best_fields (tie_breaker 0.3) / most_fields / cross_fields / phrase |
| Fusion | RRF, k = 60, top 1,000 per retriever |
| Router | name idf + preamble BM25 (fixed 1:1), margin 0.5, at most 3 contracts |
| Chunks | heading-aware, at most 500 characters, pieces under 150 characters merged, no overlap |

- **Model selection:** 13 configurations were compared. The default (routing + dense + SPLADE) had the best dev recall in the top 8 chunks, R@8 = 45.5, against 43.6 for blending all three retrievers. It was then confirmed on test.
- **Compute:** chunk vectors were encoded on a free Kaggle T4, about 6 minutes per chunking. A parity check shows the GPU and CPU encoders agree (cosine ≥ 0.99999). Everything else, including encoding questions, runs on a 4-core laptop CPU.

## 5. Offline evaluation

**Metrics.** We use the benchmark's own metrics:
- **Recall@k (R@k):** the share of the answer's characters found in the top-k chunks.
- **Precision@k (P@k):** the share of retrieved characters that belong to the answer.

Both are averaged equally over the four sub-datasets. **R@8 is our practical metric:** 8 chunks is about 1k tokens, a realistic LLM context. R@64 is reported to compare with the paper. We fixed one scoring bug: the upstream code double-counts overlapping retrieved spans, so we merge them first. 95% bootstrap confidence intervals and paired tests are in [mini.core.md](../artifacts/results/mini.core.md).

**Ablation** (test set, 541 questions; R = recall, P = precision, %):

| Configuration | P@1 | R@8 | R@64 |
|---|---|---|---|
| BM25, plain match | 2.5 | 10.1 | 30.8 |
| BM25, IBM `most_fields` (text + title) | 2.9 | 16.2 | 61.1 |
| Dense only | 11.4 | 35.6 | 75.3 |
| SPLADE only | 13.3 | 30.1 | 75.4 |
| IBM blend: SPLADE + BM25, score sum | 8.3 | 29.9 | 75.0 |
| RRF: BM25 + dense + SPLADE | 9.7 | 33.9 | 79.0 |
| … + routing | 10.8 | 38.0 | 82.5 |
| **… + routing, without BM25 (default)** | **14.4** | **40.6** | **82.0** |
| … + routing, without dense | 8.8 | 33.2 | 80.1 |
| Dense only, naive 500-character chunks, no context | 3.3 | 17.8 | 43.0 |

**Against the paper.** On all 776 mini questions the default scores P@1 13.5 and R@64 81.9. The paper's baseline (OpenAI `text-embedding-3-large`) scores P@1 6.4 and R@64 62.2 with recursive chunking, or R@64 76.4 with naive chunks.

**What the ablation shows:**
- **Templates matter:** IBM's title-aware template doubles BM25 recall (R@64 30.8 → 61.1), because matching the contract's name is most of the problem.
- **Chunk context is the biggest single gain:** heading-aware chunks with title and section context lift dense R@64 from 43.0 to 75.3.
- **Routing helps:** it adds +4.1 R@8 to the blend and fixes "right clause, wrong contract".
- **BM25 hurts ranking once routing is on:** removing it gains +2.7 R@8 (paired bootstrap p = 0.002). Its value was finding the contract, which the router now does. The exception is MAUD, where it still helps (R@64 52.6 vs 48.1).
- **Dense is the most valuable ranker:** removing it costs 4.8 R@8.
- **SPLADE adds little:** it matches dense alone on R@8 (40.6 vs 40.9, within CI) but improves P@1 (14.4 vs 13.2) and R@64 (82.0 vs 80.6).
- **IBM's raw score sum underperforms:** it scores below SPLADE alone, because unnormalised BM25 scores swamp the blend.

**Cost:**
- **Latency:** 165 ms median / 198 ms p95 per question on an i5-10210U CPU, including encoding the question.
- **Footprint:** 0.95 GB RAM; 26 MB of vectors.

## 6. Trade-offs

- **Small open models vs a hosted embedding API.** Our models are free, private, offline and run on a CPU, but they are weaker one-to-one. Routing and context-prefixed chunks close the gap and more.
- **Hard routing.** Restricting the search to the routed contract raises precision, but a routing error is unrecoverable. At 99.6% accuracy this is acceptable, and up to 3 contracts are kept when scores are close.
- **SPLADE.** It costs about 3× dense's encoding time (283 s vs 88 s on the T4) for +1.2 P@1 and +1.4 R@64. We keep it for precision; dropping it is a valid cheaper setting.
- **Dropping BM25 from ranking.** It gives a better average, but MAUD loses 4.5 R@64. A per-domain blend is possible but would need more tuning data.
- **Small chunks without overlap.** These give higher precision and cleaner citations, but long clauses get split, which lowers recall at a fixed k.
- **RRF.** It needs no tuning and is robust to score scales, but it ignores how confident each retriever is.
- **Committed vectors.** They make the grader run fast (about 6 minutes, with no GPU), at the cost of 26 MB in git.

## 7. Limitations

- **Queries name their contract.** This is a benchmark artifact that makes routing easy. A real product needs a contract picker or a "which contract?" follow-up when routing is unsure.
- **MAUD is still weak** (R@64 48–49%): merger agreements are very long, and their clauses span multiple chunks. A next step is hierarchical retrieval (section first, then clause) or larger chunks for MAUD.
- **Only the mini split was evaluated.** The full 80M-character corpus was not indexed. Cost grows linearly: encoding is per chunk, and search is a matrix product.
- **Retrieval only.** Answer quality (`ask --llm claude`) is not scored, and no cross-encoder reranker was tested.
- **One distribution.** Dev and test come from the same 72 documents (split by document). Results on other contract types are untested.

## 8. Folder structure

```
run.yaml                 exact grader commands (setup, run, test)
src/main.py              CLI: download | index | search | ask | evaluate | benchmark
src/blended_rag/
  data.py                LegalBench-RAG loading, exact mini split, dev/test split by document
  chunking.py            heading-aware chunker with exact character offsets
  indexes/               bm25.py (multi-field BM25), dense.py, splade.py
  queries.py             IBM query templates (match, best/most/cross_fields, phrase)
  routing.py             contract router
  fusion.py              RRF, weighted sum, IBM score sum
  encode.py              CPU (ONNX) and GPU (torch) encoders, fingerprinted vector cache
  pipeline.py            BlendedRetriever: route → score → fuse → cited spans
  metrics.py             character-level P/R@k, MRR, nDCG, bootstrap CIs
  evaluate.py            ablation configs and results tables
kaggle/                  GPU encoding script + how-to
data/mini/               committed corpus (72 docs) and 776 questions
artifacts/vectors/       committed chunk vectors for the default config
artifacts/results/       ablation tables (mini.core.md) and CPU benchmark
tests/                   35 tests, including an offline end-to-end run
```
