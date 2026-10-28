"""IBM Blended-RAG query templates (input/search_query/BM25/*.txt), ported from Elasticsearch to score vectors.

  match        {"match": {"content": q}}                                      -> text
  best_fields  multi_match best_fields [content, title], tie_breaker 0.3      -> max field + 0.3 * the others
  most_fields  multi_match most_fields [content, title]                       -> sum of fields
  cross_fields multi_match cross_fields: fields treated as one big field      -> BM25 on the concatenated field
  phrase       multi_match phrase_prefix                                      -> BM25 on word bigrams
bool_prefix is not ported: on full-sentence queries it equals `match` except for prefix-matching the last word.

`fields` defaults to IBM's (text, title); add "section" for our heading field.
"""

import numpy as np

TEMPLATES = ("match", "best_fields", "most_fields", "cross_fields", "phrase")
TIE_BREAKER = 0.3


def apply(template: str, field_scores: dict[str, np.ndarray], fields=("text", "title")) -> np.ndarray:
    if template == "match":
        return field_scores["text"]
    if template == "best_fields":
        stack = np.stack([field_scores[f] for f in fields])
        best = stack.max(0)
        return best + TIE_BREAKER * (stack.sum(0) - best)
    if template == "most_fields":
        return np.sum([field_scores[f] for f in fields], axis=0)
    if template == "cross_fields":
        return field_scores["all"]
    if template == "phrase":
        return field_scores["phrase"]
    raise ValueError(f"unknown template {template!r}")
