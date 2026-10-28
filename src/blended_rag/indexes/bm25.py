"""Multi-field BM25, the in-process stand-in for IBM's Elasticsearch BM25 index.

Fields (one BM25 index each, over the same stemmed, stopword-free tokens):
  text     chunk text                                   ES `content`
  title    document name, same for every chunk of a doc ES `title`
  section  heading(s) of the chunk (our addition)
  all      title + section + text concatenated          for cross_fields (term stats across fields)
  phrase   word bigrams of title + text                 for phrase queries (ES phrase_prefix)
Built in seconds on CPU; never cached to disk.
"""

import bm25s
import numpy as np
import Stemmer

from ..chunking import Chunk

FIELDS = ("text", "title", "section", "all", "phrase")


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


class BM25Fields:
    def __init__(self, chunks: list[Chunk], corpus: dict[str, str]):
        self.tokenizer = bm25s.tokenization.Tokenizer(stopwords="en", stemmer=Stemmer.Stemmer("english"))
        self.n = len(chunks)
        texts = [corpus[c.doc][c.start:c.end] for c in chunks]
        raw = {
            "text": texts,
            "title": [c.title for c in chunks],
            "section": [c.section for c in chunks],
            "all": [f"{c.title}\n{c.section}\n{t}" for c, t in zip(chunks, texts)],
        }
        tokens = {name: self._tokens(docs) for name, docs in raw.items()}
        tokens["phrase"] = [_bigrams(t) for t in tokens["all"]]
        self.index = {name: self._fit(toks) for name, toks in tokens.items()}

    def _tokens(self, texts: list[str]) -> list[list[str]]:
        # String tokens, not ids: bm25s sizes its idf table per index, so ids from a shared vocab break it.
        # update_vocab=True: the default ("if_empty") silently drops words unseen by the first field indexed.
        return self.tokenizer.tokenize(texts, return_as="string", update_vocab=True, show_progress=False)

    @staticmethod
    def _fit(tokens) -> bm25s.BM25:
        bm = bm25s.BM25()
        bm.index(tokens, show_progress=False)
        return bm

    def scores(self, query: str) -> dict[str, np.ndarray]:
        """BM25 score of every chunk, per field."""
        words = self._tokens([query])[0]
        out = {}
        for field, bm in self.index.items():
            toks = [t for t in (_bigrams(words) if field == "phrase" else words) if t in bm.vocab_dict]
            out[field] = bm.get_scores(toks) if toks else np.zeros(self.n, dtype=np.float32)
        return out
