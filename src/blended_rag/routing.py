"""Document routing: decide which contract(s) a question is about before ranking clauses.

A lawyer asks about a specific agreement ("Kenway's NDA", "the merger between Slack and salesforce.com"). Flat
chunk retrieval often finds the right clause type in the wrong contract (every NDA has a confidentiality clause),
so we score whole documents first and rank chunks only inside the routed ones.

  doc score = name match + context match
  name      max over query tokens found in the document's name of idf(token). Max, not sum: one distinctive
            name ("kenway") must beat several generic words ("non", "disclosure", "agreement") that match a
            template NDA's title. The name is matched both camel-split and as written ("AG Projects", "AGProjects").
  context   BM25(query, 3 x name + first 2,000 chars), normalised to [0, 1]; contracts name parties up front.
  route     documents within `margin` of the best score, at most `max_docs`.
Weights are fixed (1 : 1), not tuned. Routing top-1 on LegalBench-RAG-mini: 0.78 with summed BM25, 0.995 with this.
"""

import re
from urllib.parse import unquote

import bm25s
import numpy as np
import Stemmer

from .chunking import doc_title

PREAMBLE = 2000


def _raw_name(file_path: str) -> str:
    name = unquote(file_path.split("/", 1)[-1].rsplit(".txt", 1)[0].split("||")[0])
    return re.sub(r"[_\-]+", " ", name)


class DocRouter:
    def __init__(self, corpus: dict[str, str], max_docs: int = 3, margin: float = 0.5):
        self.docs = sorted(corpus)
        names = [f"{doc_title(d)} {_raw_name(d)}" for d in self.docs]
        self.tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=Stemmer.Stemmer("english"))
        context = [f"{(n + ' ') * 3}\n{corpus[d][:PREAMBLE]}" for d, n in zip(self.docs, names)]
        context_tokens = self._tokens(context)
        self.bm25 = bm25s.BM25()
        self.bm25.index(context_tokens, show_progress=False)
        self.name_tokens = [set(t) for t in self._tokens(names)]
        df: dict[str, int] = {}
        for toks in context_tokens:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(self.docs)
        self.idf = {t: float(np.log(1 + (n - f + 0.5) / (f + 0.5))) for t, f in df.items()}
        self.max_docs, self.margin = max_docs, margin

    def _tokens(self, texts: list[str]) -> list[list[str]]:
        return self.tokenizer.tokenize(texts, return_as="string", update_vocab=True, show_progress=False)

    def scores(self, query: str) -> np.ndarray:
        q = self._tokens([query])[0]
        qset = set(q)
        name = np.array([max((self.idf.get(t, 0.0) for t in qset & nt), default=0.0) for nt in self.name_tokens])
        toks = [t for t in q if t in self.bm25.vocab_dict]
        ctx = self.bm25.get_scores(toks) if toks else np.zeros(len(self.docs))
        return name + (ctx / ctx.max() if ctx.max() > 0 else ctx)

    def route(self, query: str) -> list[str]:
        s = self.scores(query)
        order = np.argsort(-s)[:self.max_docs]
        return [self.docs[i] for i in order if s[i] >= s[order[0]] - self.margin]
