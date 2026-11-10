# T7 · Blended RAG for Contract Review — Part 1: the system, step by step, with the math

> Read this first. Part 2 ([02-evaluation-and-interview.md](02-evaluation-and-interview.md)) covers metrics, statistics, results, the 5-minute script and the Q&A bank.
>
> Diagrams: [offline.svg](offline.svg) (build once per corpus) · [online.svg](online.svg) (one question at query time).

---

## 0. The whole project in one page

**The task we picked.** Contract review. A lawyer asks a question about **one specific contract**, for example:

> "Consider the Acquisition Agreement between Parent 'BridgeBio Pharma, Inc.' and Target 'Eidos Therapeutics, Inc.'; Where is the No-Shop Clause?"

The system must return **the exact passage** (character offsets inside the contract) so the lawyer, or an LLM, can cite it.

**What the brief asked.** "Use the provided Blended RAG pipeline (IBM) on a real-world task with complex text retrieval."

**The story in five lines (problem → brainstorming → solution):**
1. **Problem:** contracts look alike. Every NDA has a confidentiality clause, and every merger agreement has a No-Shop clause. A plain search finds the right *kind* of clause but often in the **wrong contract**.
2. **Brainstorm:** split the problem into two questions: (a) *which contract?* then (b) *which clause inside it?*
3. **Solution, part (a):** a cheap **router** that scores whole contracts by name, using rare words.
4. **Solution, part (b):** IBM's **Blended RAG** idea, which is several different retrievers combined, rebuilt **without Elasticsearch** so it runs on a laptop CPU.
5. **Result:** 81.9% of the answer text is in the top 64 chunks, against 62.2% for the benchmark authors' OpenAI-embedding baseline, at 165 ms per question on a CPU.

**The final system (online):**
```
question ──► ROUTE: pick ≤3 contracts ──► SCORE chunks of those contracts with
             (rare-name-word + BM25)        dense (bge-small) and SPLADE++
                                         ──► FUSE the two rankings with RRF ──► top-k cited spans
```

**Why "Blended"?** IBM's paper ("Blended RAG", Sawarkar et al., 2024) showed that **mixing** keyword search (BM25), meaning search (dense vectors) and learned sparse search (ELSER), plus smart query templates, beats any single retriever. We kept that idea and replaced the parts that need Elasticsearch.

---

## 1. Why IBM's code could not be used as-is

| IBM component | Problem | What we used instead |
|---|---|---|
| Elasticsearch server | needs a running cluster; the grader runs `run.yaml` on a plain machine | in-process Python libraries (`bm25s`, numpy, scipy) |
| ELSER v1 (Elastic's learned sparse model) | needs a paid/trial Elastic ML node and several GB of RAM | **SPLADE++** (`prithivida/Splade_PP_en_v1`), the open model ELSER is based on |
| Dense kNN in Elasticsearch | needs the cluster | **bge-small-en-v1.5** vectors + an exact numpy dot product |
| FlanT5-XXL (11B) answer generator | will not run on free hardware | retrieval-first; optional `ask --llm claude` |
| Hard-coded Elasticsearch URL, `os.chdir`, per-row indexing | research scripts, not a library | a modular, tested package (35 tests) |

**Interview line:** "I kept IBM's method (three retrievers, multi-field query templates, fusion) and rebuilt the infrastructure in-process, because the grader runs on a plain CPU machine."

---

## 2. Data: LegalBench-RAG

### 2.1 What it is
LegalBench-RAG (Pipitone & Houir Alami, 2024) is a retrieval benchmark built from four legal datasets:

| Sub-dataset | What the documents are | In the mini split |
|---|---|---|
| **PrivacyQA** | privacy policies of apps (Fiverr, 23andMe, …) | 7 docs · 194 questions |
| **ContractNLI** | non-disclosure agreements (NDAs) | 18 docs · 194 questions |
| **MAUD** | merger & acquisition agreements (very long, ~350k chars each) | 18 docs · 194 questions |
| **CUAD** | commercial contracts (licence, supply, agency, …) | 29 docs · 194 questions |
| **Total** | | **72 docs · 8,682,104 characters · 776 questions** |

The full benchmark is 714 documents, 79.7M characters and about 6.9k questions.

### 2.2 Why it is a good benchmark for this task
- **The answers are character spans.** Each question has one or more gold snippets `(file, start, end)`, so we can score retrieval **exactly**, with no LLM judge and no human grading.
- **It is genuinely hard.** Long documents, legal jargon, near-duplicate contracts.
- **There is a published baseline** to beat: the authors ran OpenAI `text-embedding-3-large` with two chunking methods.

### 2.3 The "mini" split, reproduced exactly
The paper reports numbers on **LegalBench-RAG-mini**: 194 questions per sub-dataset. The download does not contain the mini file, so we reproduced the authors' selection code (`benchmark.py`) exactly:

```
for each sub-dataset:
    sort questions by  key(q) = random(seed = q's document path)     # same doc → same key
    keep the first 194
```
Seeding the random generator with the **document path** gives every question of one document the same key, so the 194 questions come from a **few whole documents**. The counts match the paper exactly (72 docs, 8,682,104 chars), which proves the reproduction and makes our numbers **directly comparable** to theirs.

### 2.4 Dev / test split **by document**
- **Dev:** 30% of the documents in each sub-dataset → 235 questions. Used for every design choice.
- **Test:** the other 70% → 541 questions. Used only to report numbers.

**Why by document and not by question?** Questions about the same contract share vocabulary and structure. If the same contract appeared in both dev and test, tuning on dev would leak information about test. Grouping by document means **no test contract ever influenced a choice**.

### 2.5 The observation that shaped the whole design
Every question starts with **"Consider the <contract description>; <question>"**. So every question *names its contract*. That suggested the two-stage design (route to the contract, then find the clause). It is also a **benchmark artifact**: real users might not name the contract. We say this openly (Part 2, limitations).

---

## 3. Chunking: cutting contracts into retrievable pieces

### 3.1 Why chunk at all
Retrievers compare a question with **pieces** of text. A whole merger agreement (~350k characters) is too long for an encoder (512-token limit) and would match everything. The answer is usually short: the **median gold span is about 330 characters**. So we cut documents into pieces of at most **500 characters**.

### 3.2 Two chunkers, compared
| Chunker | How | Chunks on mini |
|---|---|---|
| **Naive** (paper's baseline) | fixed 500-character windows, no overlap | 17,403 |
| **Structure-aware** (ours) | cut at headings, pack whole lines/sentences up to 500 chars | 22,739 |

**Structure-aware algorithm:**
1. **Find headings** with regular expressions: `ARTICLE IV GENERAL UNDERTAKING`, `Section 7.2 Non-Compete.`, or an ALL-CAPS line such as `MUTUAL NON-DISCLOSURE AGREEMENT`.
2. **Never cross a heading:** each section is chunked on its own.
3. **Split a long section recursively** at the coarsest boundary that works: first at line breaks, then sentence ends (`.` `;` `:`), then spaces, then a hard cut. Then **greedily re-pack** neighbouring pieces while the total stays ≤ 500 characters.
4. **Merge tiny pieces.** A bare heading like `ARTICLE XII` or a sentence tail under 150 characters is merged into its neighbour, if the result still fits. These were about 13% of the pieces and are useless on their own.
5. **Trim whitespace** and keep exact offsets `(doc, start, end)`.

Each chunk also gets two **fields**:
- `section`: the heading(s) inside the chunk, or the heading it continues;
- `title`: a readable document name from the file name. URL-decoding and camelCase splitting turn `VELCO%20NDA%20rev0` into "VELCO NDA rev0" and `ExcelerateStandardNDAFormat` into "Excelerate Standard NDA Format".

### 3.3 Why **no overlap** (a math argument)
The metric charges precision for **every retrieved character** (§Part 2). If chunks overlapped by 50%, the top-8 results would contain many duplicate characters: you pay for them in precision and they bring no new recall. Gold spans are short (~330 chars), so overlap's usual benefit (an answer cut in half by a boundary) is small. Cutting at headings and sentence ends prevents most bad cuts anyway.

### 3.4 How we knew the structure-aware chunker was better: the oracle test
Before any retriever existed, we asked: *if a perfect retriever returned exactly the chunks that overlap the answer, how precise would the result be?*

$$\text{oracle precision} = \frac{\text{answer characters inside those chunks}}{\text{total characters of those chunks}}$$

| Chunker | Oracle precision | Oracle recall |
|---|---|---|
| Naive 500 | 0.49 | 0.999 |
| Structure-aware 500 | **0.70** | 0.998 |

Chunks that follow the document's structure **line up with the answers**, so less waste. This also gives the **ceiling** for P@1 (Part 2 §1.4).

### 3.5 The context prefix (the biggest single gain in the project)
Before encoding, every chunk is prefixed with its contract title and section:

```
Eidos Therapeutics, Inc. BridgeBio Pharma, Inc. | Section 6.3 No Solicitation.
(a) The Company shall not, and shall cause its Subsidiaries not to, directly or indirectly …
```

**Why:** a chunk like "(a) The Company shall not … solicit" says nothing about *which* contract or *which* clause it belongs to. With the prefix, the encoder's vector contains "BridgeBio", "Eidos" and "No Solicitation", so it matches the question.

**Effect:** dense R@64 went from **43.0 to 75.3** (naive plain chunks → structure chunks with context). That is the largest single improvement in the whole ablation. The same idea appears in industry as "contextual retrieval".

---

## 4. The three retrievers (the "blend")

Each retriever answers: *how relevant is chunk c to question q?* They make **different mistakes**, which is why blending helps.

### 4.1 BM25: keyword matching done right

**Intuition.** Count how often the question's words appear in the chunk, but:
- **rare words matter more** ("No-Shop" beats "the");
- **repeating a word has diminishing returns** (the 10th "Company" adds little);
- **long chunks are penalised a little** (they contain more words by chance).

**Formula** (Lucene variant, as implemented in `bm25s` with its defaults k₁ = 1.5, b = 0.75):

$$\text{BM25}(q, c) = \sum_{t \in q} \text{idf}(t)\cdot \frac{f(t,c)\,(k_1+1)}{f(t,c) + k_1\left(1 - b + b\,\frac{|c|}{\text{avgdl}}\right)}$$

$$\text{idf}(t) = \ln\!\left(1 + \frac{N - n_t + 0.5}{n_t + 0.5}\right)$$

- $f(t,c)$: how many times term $t$ appears in chunk $c$;
- $|c|$: chunk length in terms; avgdl: the average chunk length;
- $N$: number of chunks; $n_t$: number of chunks containing $t$;
- $k_1$ controls saturation (how fast repeated words stop adding); $b$ controls the length penalty.

**Tiny example.** N = 22,739 chunks. "solicit" appears in 120 chunks, "company" in 9,000:
- idf(solicit) = ln(1 + 22,619.5/120.5) ≈ ln(188.7) ≈ **5.24**
- idf(company) = ln(1 + 13,739.5/9,000.5) ≈ ln(2.53) ≈ **0.93**

One "solicit" counts about 5.6× more than one "company".

**Pre-processing:** lower-casing, English **stopword** removal, and **Snowball stemming** ("terminating", "terminated" → "termin"), so different forms of a word match.

**BM25 is great at:** exact legal terms ("No-Shop", "Change of Control", party names).
**BM25 is bad at:** paraphrase ("can they walk away early?" vs "termination for convenience").

### 4.2 IBM's multi-field query templates
IBM did not just run BM25 on the text. They searched **several fields** (`content`, `title`) with Elasticsearch `multi_match` query types. We ported them as formulas over per-field BM25 scores $s_f$:

| Template (Elasticsearch name) | Our formula | Meaning |
|---|---|---|
| `match` | $s_\text{text}$ | text only |
| `best_fields` (tie_breaker 0.3) | $\max_f s_f + 0.3 \sum_{f \neq \text{best}} s_f$ | best field wins, others help a little |
| `most_fields` | $\sum_f s_f$ | every matching field adds up |
| `cross_fields` | BM25 on one concatenated field "title + section + text" | treat all fields as one bag of words |
| `phrase` | BM25 on **word bigrams** ("no_shop", "change_control") | rewards exact phrases |

We indexed five BM25 fields: `text`, `title`, `section`, `all` (concatenation) and `phrase` (bigrams of title + text).

**Why the title field is so powerful here:** the question names the contract, and the `title` field holds the contract name. Adding the title field gives every chunk of the **right contract** a bonus, a soft form of routing.

**Effect:** BM25 R@64 went from **30.8 (plain `match`) to 61.1 (`most_fields`)**. On NQ, IBM's paper saw a similar jump from match to best_fields (25 → 85 top-5 accuracy).

### 4.3 Dense retrieval: meaning matching (bge-small-en-v1.5)

**Intuition.** A neural network reads a text and outputs a point in a 384-dimensional space. Texts with similar **meaning** land close together, even with no shared words.

**How bge-small works (bi-encoder):**
1. The tokenizer splits text into word-pieces (max 512 tokens).
2. A 12-layer BERT-style transformer (33M parameters) produces one vector per token.
3. **CLS pooling:** take the vector of the special first token `[CLS]` as the sentence vector $h \in \mathbb{R}^{384}$.
4. **L2-normalise:** $v = h / \lVert h\rVert$, so every vector has length 1.

**Score = cosine similarity = dot product** (because lengths are 1):

$$s_\text{dense}(q, c) = \cos(v_q, v_c) = v_q \cdot v_c = \sum_{i=1}^{384} v_{q,i}\, v_{c,i} \in [-1, 1]$$

**Query instruction.** bge models are trained to receive questions with the prefix *"Represent this sentence for searching relevant passages: "*. We add it to questions only, not to chunks. This is called **asymmetric** retrieval: short question vs long passage.

**Search.** Mini has 22,739 chunks, so the score of every chunk is one matrix–vector product: $S = V_\text{chunks}\, v_q$, a 22,739 × 384 matrix (≈ 8.7M multiply-adds, under a millisecond). **No approximate nearest-neighbour index** (HNSW / FAISS) is needed at this size. We would add one for millions of chunks.

**Dense is great at:** paraphrase and synonyms.
**Dense is bad at:** exact rare strings (a party name, a section number), and it can be "vaguely on topic".

### 4.4 SPLADE++: learned sparse retrieval (our replacement for ELSER)

**Intuition.** BM25 uses the words **literally present**. SPLADE uses a transformer to decide (a) **how important** each word in the chunk is and (b) **which other words should also be added** ("expansion"). The output is still a list of words with weights, so it is as fast as keyword search but smarter.

**How it works:**
1. Run the text through a BERT model with its **masked-language-model (MLM) head**. For every token position $i$ and every vocabulary word $j$ (30,522 words), the head gives a logit $\ell_{ij}$: "how much does position $i$ suggest word $j$?"
2. Turn this into one weight per vocabulary word with **max pooling** over positions:

$$w_j = \max_{i}\ \log\big(1 + \text{ReLU}(\ell_{ij})\big)$$

- **ReLU** removes negative evidence;
- **log(1 + ·)** dampens very large values, similar in spirit to BM25's saturation;
- **max over positions** means "the strongest evidence anywhere in the text".

3. Most $w_j$ are exactly 0, so the vector is **sparse**: about 100 non-zero words out of 30,522 for a 500-char chunk.

**Score** = dot product of the two sparse vectors (only shared non-zero words count):

$$s_\text{SPLADE}(q, c) = \sum_{j} w_j(q)\, w_j(c)$$

**Example of expansion.** A chunk "Either party may cancel this Agreement upon ninety days' notice" may get non-zero weights for "termination", "terminate" and "end", words that **are not in the text**. A question "can they terminate early?" then matches.

**Storage.** We keep chunk vectors as a sparse CSR matrix (rows = chunks, columns = vocabulary). Scoring all chunks is one sparse matrix–vector product, the same operation an **inverted index** does.

**SPLADE vs ELSER:** ELSER is Elastic's productised model built on the same SPLADE idea. SPLADE++ is open and runs anywhere.

### 4.5 Why three retrievers (complementary errors)
| Question type | BM25 | Dense | SPLADE |
|---|---|---|---|
| exact term: "Where is the No-Shop clause?" | ✔ strong | ~ | ✔ |
| paraphrase: "can the buyer walk away?" | ✘ | ✔ strong | ~ |
| party name: "Kenway's NDA" | ✔ | ✘ weak | ✔ |
| mixed | ~ | ~ | ✔ in between |

When retrievers err on **different** questions, combining them lifts recall. On test, the RRF blend of all three reached R@64 **79.0**, above every single retriever (dense 75.3, SPLADE 75.4, BM25 61.1).

---

## 5. Routing: "which contract is this question about?"

### 5.1 Why a router
Without routing, a question about Kenway's NDA retrieves confidentiality clauses from **all 18 NDAs**: right clause type, wrong contract. A router picks the contract first. Then we rank chunks **only inside it** (a hard filter called a "mask").

### 5.2 First attempt and why it failed
First router: BM25 over (title + first 2,000 characters) plus fuzzy string matching of the title. Result: **78% top-1**, but only **37% on ContractNLI**.

**Debugging one failure** ("Consider Kenway's Non-Disclosure Agreement; Does the document indicate …"):
- the right doc ("Kenway NDA Form Blank") scored 6.3;
- a generic template "Mutual Non Disclosure Agreement …" scored 6.8;
- reason: generic question words ("does", "receiving", "consider", "non", "disclosure") matched the **template NDA's** preamble. Their **sum** beat the one distinctive word "kenway" (BM25 contribution 3.1 vs ≤ 0.7 for any generic word).

### 5.3 The fix: max-pooling over rare name words
A document reference works like a **name**: one distinctive word identifies it. So instead of **summing** word scores, take the **maximum**:

$$\text{name}(d) = \max_{t \,\in\, Q \,\cap\, \text{Name}(d)} \text{idf}(t), \qquad \text{idf}(t) = \ln\!\left(1 + \frac{D - n_t + 0.5}{n_t + 0.5}\right)$$

- $Q$: the question's tokens (stemmed, stopwords removed); Name(d): tokens of the document's name;
- $D$ = 72 documents; $n_t$ = how many documents contain $t$ (in name + first 2,000 chars).

Add a **context** score so contracts that name parties in their preamble also match:

$$\text{ctx}(d) = \frac{\text{BM25}\big(q,\ 3\times\text{name} + \text{first 2,000 chars}\big)}{\max_{d'} \text{BM25}(q, d')} \in [0,1]$$

$$\boxed{\text{score}(d) = \text{name}(d) + \text{ctx}(d)}$$

**Selection:** keep every document within **margin 0.5** of the best score, **at most 3** (covers near-ties like amendments).

**Two cleaning details that mattered:**
- the name is matched **both camel-split and as written**: "AG Projects" *and* "AGProjects", because the question wrote "AGProjects'";
- file names are URL-decoded (`%20` → space).

**Result:** routing top-1 **0.78 → 0.995** on mini (ContractNLI 0.37 → 1.0), at **0.3 ms per question**. Weights are fixed at 1:1, not tuned.

**Interview line:** "Summing word scores lets many generic words outvote one distinctive name. A reference to a contract behaves like a name, so I max-pool the idf of name words, which made routing 99.5% accurate."

### 5.4 Where BM25 ended up
BM25 still powers routing (the context score) and the IBM template baseline. But adding BM25 to the **clause ranking** after routing *hurt* (Part 2): its job, finding the contract, is done by the router, and its clause ranking is weak (R@8 16.2).

---

## 6. Fusion: combining rankings

### 6.1 The problem
Dense gives cosines in [−1, 1]. SPLADE gives unbounded positive numbers (often 10–40). BM25 gives 0–30+. **Adding raw scores** lets the retriever with the biggest numbers dominate.

### 6.2 IBM's way (raw score sum), kept as a baseline
IBM's Elasticsearch queries add raw scores:
- `knn_best` = BM25(best_fields) + **10** × kNN score;
- `se_best` = ELSER score + BM25(best_fields), keeping only chunks that match BM25 (Elasticsearch `must`).

Our port (`ibm-se-best`) scored R@64 **75.0**, *below* SPLADE alone (75.4): the unnormalised BM25 scores drown the SPLADE signal.

### 6.3 Reciprocal Rank Fusion (RRF), our choice
Use **ranks**, not scores:

$$\text{RRF}(c) = \sum_{r \,\in\, \{\text{dense},\ \text{SPLADE}\}} \frac{1}{k + \text{rank}_r(c)}, \qquad k = 60$$

- $\text{rank}_r(c)$ = position of chunk $c$ in retriever $r$'s list (1 = best); we consider each retriever's top 1,000;
- a retriever that scores a chunk 0 (no shared term) gives it no vote;
- $k = 60$ (from Cormack et al., 2009) flattens the curve, so being #1 vs #5 matters but not hugely.

**Worked example:**

| Chunk | Dense rank | SPLADE rank | RRF |
|---|---|---|---|
| A | 1 | 30 | 1/61 + 1/90 = 0.01639 + 0.01111 = **0.0275** |
| B | 3 | 2 | 1/63 + 1/62 = 0.01587 + 0.01613 = **0.0320** |

B wins: **both retrievers agree it is good**, even though neither ranks it first. Agreement beats one strong vote. That is the behaviour you want when retrievers make different mistakes.

**Why RRF:** no calibration, no tuning, robust to score scales. **Cost:** it ignores *how much* better one chunk is (score margins). A tuned weighted sum of min-max-normalised scores (`wsum`) is also implemented; we did not make it the default, to avoid tuning more knobs on a small dev set.

---

## 7. Engineering: making it run on a laptop and in the grader

### 7.1 Compute plan
- **Laptop:** i5-10210U (4 cores, AVX2), ~2 GB of free RAM, no GPU.
- **SPLADE on the laptop:** at batch size 16 it was **killed for running out of memory at 3.4 GB**. The MLM head outputs `batch × tokens × 30,522` floats, e.g. 16 × 512 × 30,522 × 4 bytes ≈ **1 GB for one tensor**, plus intermediates.
- **Projected CPU time:** ~45 minutes per chunking. So chunk encoding ran **once on a free Kaggle T4 GPU** (structure chunks: dense 88 s, SPLADE 283 s).

### 7.2 GPU vs CPU parity
Chunks are encoded with PyTorch on the GPU. Questions are encoded at query time with **fastembed (ONNX) on the CPU**. If the two disagreed, scores would be wrong. A parity check on the same chunks gave **cosine ≥ 0.99999** for both models.

### 7.3 Storage math
- Dense: 22,739 × 384 × 2 bytes (fp16) ≈ **17.5 MB**.
- SPLADE: CSR with **uint16** vocabulary ids (30,522 < 65,536) and **fp16** weights, compressed ≈ **8 MB** (scipy cannot store fp16, so we save the CSR arrays ourselves).
- Total ≈ 26 MB, committed to git, so the grader needs no GPU and no encoding.

### 7.4 Fingerprints (safety)
Each vector file is stored with a **SHA-256 hash of the exact texts that were encoded** (plus model name and max length). If chunking, titles or the context prefix change, the hash changes and the cache is ignored. **Stale vectors can never be loaded silently.**

**Story:** a title-cleaning fix changed the context prefix. An earlier fingerprint over chunk offsets alone would have reused the outdated vectors. Hashing the texts closed that hole.

### 7.5 A real bug worth telling: the hour-long OOM
fastembed's `query_embed` batches **256 questions at a time**. With SPLADE that is 256 × tokens × 30,522 floats, many GB. The process swapped for an hour, then was killed. **Fix:** encode questions with batch size 4. After the fix, the full evaluation took **25 s at 1 GB**.

**Lesson:** for models with a vocabulary-sized output, memory scales with **batch × length × vocabulary**, so batch size is a memory knob, not just a speed knob.

### 7.6 The grader path
`run.yaml`: `pip install` → `download` (two ONNX models, ~650 MB) → `index` (loads committed vectors, ~2 s) → `evaluate --configs default` → `search "…"` → `pytest` (35 tests).

From a fresh clone it runs in about **6 minutes on a CPU**, with no API key.

---

## 8. End-to-end walkthrough of one question

**Question:** "Consider the Acquisition Agreement between Parent 'BridgeBio Pharma, Inc.' and Target 'Eidos Therapeutics, Inc.'; Where is the No-Shop Clause"

1. **Encode the question** (CPU, ONNX):
   - bge-small with the instruction prefix → a 384-d unit vector;
   - SPLADE++ → about 30–60 weighted vocabulary terms ("shop", "solicit", "bridgebio", …).
2. **Route:**
   - tokens like "bridgebio" and "eidos" occur in only one document name, so their idf is high (with D = 72 and n = 1: ln(1 + 71.5/1.5) ≈ **3.9**);
   - name score ≈ 3.9, plus context ≈ 1.0 → the Eidos/BridgeBio agreement wins by a wide margin;
   - selected: 1 contract.
3. **Mask:** only that contract's chunks (a few hundred) can be ranked.
4. **Score:** dense cosine and SPLADE dot product for each of those chunks.
5. **Fuse:** RRF over the two rankings.
6. **Return the top-k:** e.g. `{"doc": "maud/Eidos Therapeutics, Inc._BridgeBio Pharma, Inc..txt", "span": [190878, 191031], "section": "Article IX", "text": "…No Change in Parent Recommendation…"}`.
7. **Optional:** `ask --llm claude` sends the numbered spans to an LLM, which answers with citations [1], [2].

**Time:** 165 ms p50 per question, most of it spent encoding the question with two transformers on the CPU.

---

## 9. Glossary (simple definitions)

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation: find relevant text first, then let an LLM answer from it |
| **Chunk** | a piece of a document (here ≤ 500 characters) that is retrieved as a unit |
| **Span / offsets** | `[start, end)` character positions inside a document |
| **BM25** | classic keyword relevance score (term frequency × rarity, with saturation and a length penalty) |
| **idf** | inverse document frequency: how rare a word is → how informative |
| **Bi-encoder / dense retrieval** | encode question and chunk separately into vectors; compare with cosine |
| **CLS pooling** | use the first special token's output as the sentence vector |
| **SPLADE** | a transformer that outputs a sparse, weighted, expanded bag of words |
| **MLM head** | the part of BERT that predicts vocabulary words; SPLADE reuses it to weight words |
| **Inverted index** | word → list of chunks containing it; makes sparse search fast |
| **ELSER** | Elastic's learned sparse model (SPLADE-like), used by IBM |
| **RRF** | Reciprocal Rank Fusion: combine rankings by summing 1/(60 + rank) |
| **Routing** | choose the target document(s) before searching inside them |
| **Mask** | set the scores of chunks outside the routed contracts to −∞ |
| **ONNX / fastembed** | portable model format and a small CPU inference library (no PyTorch) |
| **fp16 / uint16** | 2-byte float / 2-byte unsigned integer storage |
| **Fingerprint** | a hash of the exact inputs; a mismatch means the cache is stale |
