# Interview prep: T7 · Blended RAG for contract review

Study notes for the AIntropy interview (30 min: 5 min per project + Q&A, as technical as possible).

| File | What's in it |
|---|---|
| [01-system-and-math.md](01-system-and-math.md) | every component in simple English: intuition → formula → worked example (data, chunking, BM25, IBM templates, dense, SPLADE, routing, RRF, engineering) |
| [02-evaluation-and-interview.md](02-evaluation-and-interview.md) | metric math, statistics, the ablation read row by row, trade-offs, limitations, **timed 5-minute script**, **25 Q&A**, numbers card |
| [offline.svg](offline.svg) | build once per corpus: data → chunks + context → GPU encoding → indexes → eval → default chosen on dev |
| [online.svg](online.svg) | one question: route → score (dense, SPLADE) → RRF → cited spans |

- **Slide deck** (7 slides): https://claude.ai/artifact/9NLWy1TwW7cesVb8G5kD7c
- **Development notes:** the full conversation log of how this was built is in [../dev-notes.md](../dev-notes.md).
- **The other project (T2, video query intent router):** its guide is in the `user-intent-classifier` repo, at `docs/interview-prep/`.

## How to study (≈ 2–3 hours for this project)
1. Read Part 1 with [online.svg](online.svg) open. After each section, explain it aloud in one sentence.
2. Derive by hand, with no notes:
   - BM25;
   - an RRF example;
   - the router's max-pooled idf;
   - character-level P/R and the 330/500 ceiling;
   - the paired bootstrap.
3. Memorise the numbers card (Part 2 §9).
4. Say the 5-minute script aloud with a timer, 3 times, in your own words.
5. Answer the Q&A aloud; re-read any section you can't answer without notes.

**Theme linking both projects:** "decide where to look before you look". T2 routes a question to an index; T7 routes it to a contract.
