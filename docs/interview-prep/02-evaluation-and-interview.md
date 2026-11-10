# T7 · Blended RAG — Part 2: evaluation, results, the 5-minute talk and Q&A

> Part 1 ([01-system-and-math.md](01-system-and-math.md)) explains every component. This part explains **how we measured it**, **what the numbers mean**, and **what to say**.

---

## 1. Metrics, with the math

### 1.1 Character-level Recall@k and Precision@k (the benchmark's own metric)
For one question, let $G$ = the set of **gold characters** (the union of the gold spans) and $R_k$ = the set of characters in the **top-k retrieved chunks**:

$$\text{R@}k = \frac{|R_k \cap G|}{|G|} \qquad \text{P@}k = \frac{|R_k \cap G|}{|R_k|}$$

- **Recall@k**: what fraction of the answer text did we find?
- **Precision@k**: what fraction of what we returned is answer text?

**Worked example.** The gold span is `[1000, 1330)` (330 chars):
- retrieve one chunk `[900, 1400)` (500 chars): overlap = 330 → **R@1 = 1.0, P@1 = 330/500 = 0.66**;
- retrieve a chunk `[1200, 1700)`: overlap = 130 → R@1 = 0.39, P@1 = 0.26.

### 1.2 The upstream scoring bug we fixed
The authors' code computes overlap **per retrieved chunk and adds them up**. If two retrieved chunks overlap each other, the same gold characters are counted twice, so recall can exceed 1. We **merge retrieved spans into a union first** ($R_k$ is a set of characters). Our numbers can therefore only be **stricter**, never inflated. Our chunks don't overlap anyway, but the fix makes comparisons fair.

### 1.3 Macro-average over the four sub-datasets
$$\text{ALL} = \tfrac14\big(\text{PrivacyQA} + \text{ContractNLI} + \text{MAUD} + \text{CUAD}\big)$$
Each sub-dataset weighs 25% regardless of size, exactly as in the paper. This prevents one easy dataset from hiding a hard one.

### 1.4 Why P@1 looks low (and what the ceiling is)
P@1 = 13.5% sounds bad until you see the arithmetic: a 500-char chunk containing a 330-char answer can reach at most 330/500 = 66%. Across the dataset, the oracle precision of our chunks (retrieving exactly the chunks that touch the answer) is about **70%**. So **13.5 vs the paper's 6.4 is 2× the baseline, against a ~70% ceiling**. P is low by construction for any chunk-based retriever.

### 1.5 Why R@8 is our main metric and R@64 is reported
- **R@8:** 8 chunks × 500 chars ≈ 4,000 chars ≈ **1,000 tokens**, a realistic LLM context. It reflects what an answer generator would actually see.
- **R@64:** the paper's headline cutoff, reported only to compare with it.

### 1.6 Rank metrics (where the first relevant chunk is)
A chunk is "relevant" if it overlaps any gold span.
- **doc@k:** is any of the top-k chunks from the right **contract**? (It isolates routing errors.)
- **MRR (mean reciprocal rank):** $\frac{1}{|Q|}\sum_q \frac{1}{\text{rank of first relevant chunk}}$. First relevant at rank 3 → 1/3.
- **nDCG@10:** rewards relevant chunks near the top, with log discounting:
$$\text{DCG@10} = \sum_{i=1}^{10} \frac{\text{rel}_i}{\log_2(i+1)}, \qquad \text{nDCG@10} = \frac{\text{DCG@10}}{\text{IDCG@10}}$$
IDCG is the best possible DCG (all relevant chunks at the top).

---

## 2. Statistics: how sure are we?

### 2.1 Bootstrap confidence intervals
We have 541 test questions, but a different sample of questions would give slightly different numbers. **Bootstrap:**
1. Within each sub-dataset, sample its questions **with replacement** (same size). This is the "stratified" part, which keeps the 25% weighting fair.
2. Compute the macro R@8 on this resample.
3. Repeat **2,000** times. The middle 95% of the results (2.5th to 97.5th percentile) is the **95% CI**.

Example: default R@8 = 40.6, CI **37.6–43.6** (about ±3 points).

### 2.2 Paired bootstrap test (is config A really better than B?)
Both configs answer the **same** questions, so compare them **question by question**:
1. For each question, $d_q$ = R@8(A) − R@8(B).
2. Bootstrap-resample questions (stratified) 2,000 times and compute the macro mean of $d$ each time.
3. **p-value** = the share of resamples where the mean difference ≤ 0.

Pairing removes the noise from "some questions are just hard", so it is much more sensitive than comparing two separate CIs.

**Key result:** dropping BM25 from the ranking: +2.7 R@8, **p = 0.002**, i.e. in only 0.2% of resamples was it not better.

---

## 3. Protocol: how decisions were made (no test leakage)
1. **Dev (235 questions, 30% of documents):** every choice. 13 configurations were compared, and the default was picked by **dev R@8**.
2. **Test (541 questions, other 70% of documents):** report only.
3. **All 776:** the default config only, to compare with the paper (whose numbers use all mini questions).

| Dev split (235) | R@8 | R@64 |
|---|---|---|
| route + BM25 + dense + SPLADE | 43.6 | 81.7 |
| route + dense only | 43.8 | 80.1 |
| **route + dense + SPLADE (chosen)** | **45.5** | **82.5** |
| route + BM25 + SPLADE (no dense) | 40.0 | 81.1 |
| route + BM25 + dense (no SPLADE) | 41.1 | 79.9 |

Test then **confirmed** the choice (+2.7 R@8 over the all-three blend, p = 0.002).

---

## 4. Results: the ablation table, read row by row

Test set, 541 questions, % (P = precision, R = recall, char-level; "Contract @1" = doc@1).

| # | Configuration | P@1 | R@8 | R@64 | Contract @1 | What this row teaches |
|---|---|---|---|---|---|---|
| 1 | BM25, plain `match` (text only) | 2.5 | 10.1 | 30.8 | 76.1 | keyword search alone is weak on legal text |
| 2 | BM25, IBM `most_fields` (text + title) | 2.9 | 16.2 | 61.1 | 86.0 | **IBM's templates double recall**: the title field finds the contract |
| 3 | Dense, naive chunks, no context | 3.3 | 17.8 | 43.0 | 77.3 | a typical "plain RAG" setup |
| 4 | Dense (heading chunks + context prefix) | 11.4 | 35.6 | 75.3 | 95.0 | **context prefix = biggest single gain** (43.0 → 75.3) |
| 5 | SPLADE | 13.3 | 30.1 | 75.4 | 94.3 | best single-retriever precision |
| 6 | IBM blend: SPLADE + BM25, raw score sum | 8.3 | 29.9 | 75.0 | 94.1 | raw-score sum is **worse than SPLADE alone**: scale mismatch |
| 7 | RRF: BM25 + dense + SPLADE | 9.7 | 33.9 | 79.0 | 95.2 | blending raises recall above every single retriever |
| 8 | … + routing | 10.8 | 38.0 | 82.5 | 99.4 | routing: +4.1 R@8, wrong-contract errors almost gone |
| 9 | **… + routing, without BM25 (default)** | **14.4** | **40.6** | **82.0** | **99.4** | BM25 hurts the *ranking* once routing exists |
| 10 | … + routing, without dense | 8.8 | 33.2 | 80.1 | 99.4 | dense is the most valuable ranker (−4.8 R@8) |
| 11 | … + routing, without SPLADE | 8.8 | 35.5 | 82.0 | 99.4 | SPLADE adds +2.5 R@8 to BM25 + dense |
| 12 | route + dense only | 13.2 | 40.9 | 80.6 | 99.4 | ties the default on R@8 (within CI); SPLADE's gain is P@1 (+1.2), R@64 (+1.4) |

### 4.1 Leave-one-out: what each retriever contributes
Leave-one-out = remove one component from the full system and measure the drop.

| Removed from route + BM25 + dense + SPLADE (R@8 38.0) | R@8 | Change | Reading |
|---|---|---|---|
| − BM25 | 40.6 | **+2.6** (+2.7 before rounding; p = 0.002 vs the full blend) | BM25 is noise in the ranking after routing |
| − SPLADE | 35.5 | −2.5 | SPLADE helps when BM25 is also present |
| − dense | 33.2 | **−4.8** | dense is the backbone |

**Why BM25 hurts after routing (a good interview answer).** Inside one routed contract, almost every chunk contains the contract's vocabulary. BM25's strength, matching rare exact words such as the party name, no longer separates chunks, because they all share it. Its weak clause-level ranking (R@8 16.2 on its own) then pulls good dense and SPLADE results down in the fusion. Its value **moved** to the router.

**The exception:** MAUD (long merger agreements) R@64 is 52.6 with BM25 vs 48.1 without. Exact clause names ("No-Shop", "Material Adverse Effect") help in very long documents. That is a trade-off, listed in §7.

### 4.2 Per-sub-dataset R@64 (test)
| Config | PrivacyQA | ContractNLI | MAUD | CUAD |
|---|---|---|---|---|
| BM25 plain | 36 | 53 | 4 | 30 |
| Dense | 87 | 91 | 45 | 79 |
| RRF all + route | 95 | 100 | 53 | 82 |
| **Default** | **95** | **100** | **48** | **85** |

MAUD is the hardest: documents average ~350k characters, and clauses span many chunks.

### 4.3 Against the paper (all 776 mini questions)
| | P@1 | R@64 |
|---|---|---|
| **Ours (default)** | **13.5** | **81.9** |
| Paper: OpenAI `text-embedding-3-large`, recursive (RCTS) chunks | 6.4 | 62.2 |
| Paper: OpenAI embeddings, naive chunks | — | 76.4 |

**How to present it honestly:**
- **Same comparison basis:** the same questions, the same metric definition, and stricter scoring on our side (merged spans).
- **The gain comes from design, not bigger models:** the paper's baseline has **no routing and no context prefix**. Our embedder (33M params) is far smaller than OpenAI's.
- **One caveat:** routing uses the benchmark's habit of naming the contract.

### 4.4 Cost
| | Value |
|---|---|
| Latency per question (i5-10210U CPU, incl. encoding) | **165 ms p50 · 198 ms p95** |
| Peak RAM | 0.95 GB |
| Vectors on disk | 26 MB |
| Index load | ~2–5 s |
| Chunk encoding (once, Kaggle T4) | ~6 min per chunking |

**Scaling estimate (full 80M-char corpus, ~10× mini):** ~230k chunks.
- **Encoding:** ~1 hour on a T4.
- **Dense matrix:** 230k × 384 × 2 B ≈ 177 MB, so we'd add an HNSW/FAISS index.
- **SPLADE:** stays an inverted index.
- **Cost:** linear in the number of chunks.

---

## 5. Trade-offs (what we chose and what it costs)

| Decision | Benefit | Cost / risk |
|---|---|---|
| Small open models on CPU instead of a hosted embedding API | free, private (contracts never leave the machine), offline, 165 ms | weaker one-to-one; compensated by routing + context |
| Hard routing (mask) | removes "wrong contract" errors; higher precision | a routing miss is unrecoverable (0.4–0.5% of questions); mitigated by keeping up to 3 close contracts |
| RRF instead of tuned weights | no calibration, nothing to overfit on 235 dev questions | ignores score margins |
| Keep SPLADE | +1.2 P@1, +1.4 R@64 over dense alone | ~3× dense's encoding time (283 s vs 88 s on a T4) |
| Drop BM25 from ranking | +2.7 R@8 overall | MAUD −4.5 R@64 |
| 500-char chunks, no overlap | precise citations, no wasted top-k | long clauses get split → lower recall at fixed k |
| Commit vectors (26 MB) | grader runs in ~6 min, no GPU | binary data in git |

---

## 6. Limitations and next steps

| Limitation | Next step |
|---|---|
| **Every question names its contract** (benchmark artifact) | a contract picker in the UI, or a "which contract?" follow-up when the router's top-2 scores are close |
| **MAUD still weak** (R@64 48%) | hierarchical retrieval (rank sections first, then clauses inside), or bigger chunks for very long docs |
| **Only the mini split** (8.7M chars) | index the full 80M chars; add HNSW for dense |
| **Retrieval only**; answers not scored | generate answers with an LLM and score faithfulness and citation accuracy |
| **No reranker** | cross-encoder rerank of the top 50 (the paper found Cohere's reranker hurt, so it needs testing) |
| **One distribution** | test on other contract types or other jurisdictions |

---

## 7. The 5-minute presentation script (≈ 650 words, timed)

Use the 7-slide deck (https://claude.ai/artifact/9NLWy1TwW7cesVb8G5kD7c) or the two SVGs.

**[0:00–0:30] Hook (cover slide)**
"For Task 7 I built a Blended RAG retriever for contract review. A lawyer asks about one specific contract, and the system returns the exact clause with character offsets. On LegalBench-RAG it finds 82% of the answer text in the top 64 chunks, against 62% for the benchmark's OpenAI-embedding baseline, on a laptop CPU at 165 milliseconds per question."

**[0:30–1:15] Problem and brainstorm (problem slide)**
"Contracts are near-duplicates: every NDA has a confidentiality clause. A plain dense RAG finds the right *kind* of clause but often in the wrong contract. So I split the problem in two: first *which contract*, then *which clause*. For the clause step I used IBM's Blended RAG idea: keyword, dense and learned-sparse retrievers combined. IBM's code needs Elasticsearch and a licensed ELSER model, so I rebuilt it in-process: BM25, bge-small and SPLADE++."

**[1:15–1:45] Data (data slide)**
"LegalBench-RAG has four legal sub-datasets with gold answers as character spans, so scoring is exact. I reproduced the paper's mini split exactly, 776 questions over 72 documents, so the numbers are comparable. I split dev and test by document, so no test contract ever influenced a choice."

**[1:45–2:45] Architecture (architecture slide)**
"Offline, I cut contracts at section headings into chunks of at most 500 characters with exact offsets, and prefix each chunk with its contract title and section before encoding. That prefix was the biggest single gain: dense recall at 64 went from 43 to 75. Online, the router scores each contract by the rarest question word in its name plus BM25 over its preamble. Max-pooling matters: a sum let generic words like 'non-disclosure agreement' outvote one distinctive name like 'Kenway', and fixing that took routing from 78 to 99.5 percent. Inside the routed contract, dense and SPLADE rank the chunks and reciprocal rank fusion merges the two rankings without calibrating scores."

**[2:45–3:45] Results (results slide)**
"The ablation tells the story. IBM's title-aware template doubles BM25 recall. The context prefix doubles dense recall. Blending and then routing add about four more points of recall at 8. The surprising finding: once routing exists, removing BM25 from the ranking *improves* recall at 8 by 2.7 points, with a paired-bootstrap p of 0.002. BM25's value was finding the contract, and the router now does that. The default was chosen on the dev split and confirmed on test."

**[3:45–4:30] Trade-offs and limitations (last slide)**
"Small open models keep contracts private and cost nothing. Routing is hard, so a miss is unrecoverable, but it's 99.5% accurate and keeps up to three close candidates. The honest caveat: benchmark questions always name the contract; in a product I'd add a contract picker or a clarifying question. MAUD merger agreements are still the weak spot, and the next step is hierarchical retrieval."

**[4:30–5:00] Close**
"It runs from a fresh clone in about six minutes on a CPU, with 35 tests. Happy to go deeper on any part."

---

## 8. Q&A bank (technical, with short spoken answers)

**Retrieval fundamentals**

1. **Explain BM25.** "It sums, over query terms, the term's rarity (idf) times a saturating function of how often it appears in the chunk, normalised by chunk length. k₁ = 1.5 controls saturation, b = 0.75 the length penalty."
2. **Why does dense retrieval fail on party names?** "Rare proper nouns are split into word-pieces with weak embeddings, and a 384-d vector compresses the whole chunk, so a single name barely moves it. Keyword methods match it exactly."
3. **How does SPLADE differ from BM25?** "Both output a sparse bag of words, but SPLADE's weights come from a transformer's MLM head: log(1 + ReLU(logit)), max-pooled over tokens. It weights words by context and adds expansion terms that aren't in the text."
4. **What's the difference between CLS and mean pooling?** "CLS takes the first token's output vector; mean pooling averages all token vectors. bge was trained with CLS, so we use CLS for it."
5. **Why normalise vectors?** "Then the dot product equals cosine similarity, and scores are bounded in [−1, 1]."
6. **Why no ANN index?** "There are 22,739 chunks; an exact matrix-vector product takes under a millisecond. ANN pays off at millions of vectors, and I'd add HNSW for the full corpus."

**Design choices**

7. **Why RRF and not a weighted sum?** "The scores live on different scales. RRF uses ranks, needs no calibration and has one constant (k = 60) from the literature. A weighted sum adds weights to tune on a 235-question dev set."
8. **Why k = 60?** "It's the value from the original RRF paper (Cormack et al., 2009). It flattens the curve so a #1 vote doesn't dominate. I didn't tune it."
9. **Why did IBM's raw score sum underperform?** "BM25 scores (0–30) and SPLADE scores add on different scales, so the bigger one dominates. It scored 75.0 R@64, below SPLADE alone at 75.4."
10. **Why max-pooling in the router?** "A contract reference is a name: one distinctive word identifies it. Summing lets many generic words outvote it. Max of idf over name words fixed ContractNLI routing from 37% to 100%."
11. **What if the question doesn't name a contract?** "Then the router's top scores are close. I'd detect that from the score margin and ask a clarifying question, or search all contracts without the mask, which is the 79.0 R@64 blend."
12. **Why 500 characters and no overlap?** "The median answer is about 330 characters. Precision is charged per character, so overlap wastes top-k on duplicates. Cutting at headings and sentence ends avoids most bad cuts."
13. **Why prefix chunks with title and section?** "A clause like '(a) The Company shall not…' has no idea which contract or clause it belongs to. The prefix puts that into the vector. Dense R@64 went from 43 to 75."
14. **Why did BM25 hurt after routing?** "Inside one contract, every chunk shares its vocabulary, so BM25's exact-name advantage disappears and its weak clause ranking adds noise to the fusion. Its value moved into the router."

**Evaluation and statistics**

15. **Is the comparison with the paper fair?** "Same questions, same metric, and stricter scoring on my side. The difference: their baseline has no routing or context prefix, and that's where most of my gain comes from, not from better embeddings."
16. **Explain the paired bootstrap.** "Per question I take the difference between two configs, resample questions 2,000 times stratified by sub-dataset, and report the share of resamples where the mean difference is ≤ 0."
17. **Why split by document?** "Questions about the same contract are correlated. Splitting by question would let tuning see test contracts."
18. **Why is P@1 so low?** "It's character precision: a 500-character chunk with a 330-character answer can't exceed 66%, and the oracle ceiling is about 70%. 13.5 is 2× the paper's 6.4."
19. **Could you have overfit?** "The defaults are fixed (RRF k = 60, router weights 1:1, margin 0.5). The only selection was one config out of 13, on dev, confirmed on test with p = 0.002."

**Engineering**

20. **How did you run it with no GPU?** "Chunk vectors were encoded once on a free Kaggle T4. The grader loads them by fingerprint and only encodes questions on the CPU with ONNX."
21. **How do you know GPU and CPU vectors match?** "A parity test encodes the same chunks both ways: cosine ≥ 0.99999 for dense and sparse."
22. **What bug taught you the most?** "SPLADE query encoding: the library batched 256 questions, and the MLM output is batch × tokens × 30,522 floats. It swapped for an hour and was killed. Batch size 4 fixed it: 25 seconds at 1 GB."
23. **How would you deploy it?**
    - **Serving:** a stateless CPU service; vectors loaded at start-up; dense via HNSW at scale; SPLADE via an inverted index.
    - **Updates:** re-encode only new contracts.
    - **Monitoring:** routing margin (to catch ambiguous questions), latency p95, and the citation-click rate.
24. **How would you add answer generation?** "Pass the top 8 spans, numbered, to an LLM with an instruction to cite. Evaluate faithfulness (is every claim supported by a cited span?) and citation precision."
25. **What would you do with one more week?** "Hierarchical retrieval for MAUD, a cross-encoder reranker test, the full 80M-character corpus, and an answer-level evaluation."

---

## 9. Numbers card (memorise)

| | |
|---|---|
| Mini split | 776 questions · 72 docs · 8.68M chars · 4 sub-datasets × 194 |
| Dev / test | 235 / 541, split by document |
| Chunks | 22,739 (structure-aware, ≤ 500 chars) vs 17,403 naive |
| Headline (776) | R@64 **81.9** vs 62.2 · P@1 **13.5** vs 6.4 |
| Test default | R@8 **40.6** [37.6–43.6] · R@64 82.0 · P@1 14.4 |
| Biggest gains | context prefix: dense R@64 43.0 → 75.3 · IBM template: BM25 R@64 30.8 → 61.1 |
| Routing | 0.78 → 0.995 top-1 (max-pooled idf) · +4.1 R@8 |
| BM25 out of ranking | +2.7 R@8, p = 0.002 · MAUD −4.5 R@64 |
| Cost | 165 / 198 ms · 0.95 GB · 26 MB · ~6 min grader run |
| Models | bge-small-en-v1.5 (33M, 384-d) · SPLADE++ (BERT-base, 30,522-d) |
