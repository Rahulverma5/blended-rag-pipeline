"""End to end on a 3-contract fixture with BM25 + routing only: no models, no downloads."""

from blended_rag import data, evaluate, metrics
from blended_rag.pipeline import BlendedRetriever, Config

NDA = ("MUTUAL NON-DISCLOSURE AGREEMENT\nThis agreement is between Kenway Corp and Acme Inc.\n"
       "Section 1 Confidential Information. The Receiving Party shall keep Confidential Information secret.\n"
       "Section 2 Term. This Agreement expires three years after the Effective Date.\n")
SUPPLY = ("SUPPLY AGREEMENT\nBetween Birch Foods and Mount Holdings.\n"
          "Section 1 Termination. Either party may terminate for convenience on ninety days notice.\n"
          "Section 2 Term. This Agreement expires on December 31, 2030.\n")
POLICY = "Fiverr privacy policy. We collect your IP address and browsing data. We share data with sellers.\n"
CORPUS = {"contractnli/Kenway-NDA.txt": NDA, "cuad/Birch Supply Agreement.txt": SUPPLY, "privacy_qa/Fiverr.txt": POLICY}
CFG = Config("bm25+route", bm25="most_fields", dense=False, splade=False)


def _span(doc, phrase):
    s = CORPUS[doc].index(phrase)
    return data.Snippet(doc, s, s + len(phrase))


def test_routing_picks_the_named_contract_for_a_shared_clause_type():
    r = BlendedRetriever(CORPUS, CFG)
    hits = r.search("Consider the Kenway NDA; when does this Agreement expire?", k=3)
    assert hits and all(h.chunk.doc == "contractnli/Kenway-NDA.txt" for h in hits)
    assert "three years" in hits[0].text


def test_hits_are_exact_citable_spans():
    r = BlendedRetriever(CORPUS, CFG)
    for h in r.search("Consider the Birch Supply Agreement; can a party terminate for convenience?", k=2):
        assert CORPUS[h.chunk.doc][h.chunk.start:h.chunk.end] == h.text


def test_evaluate_run_scores_fixture_queries():
    qs = [data.Query("q1", "Consider the Kenway NDA; when does this Agreement expire?", "contractnli",
                     [_span("contractnli/Kenway-NDA.txt", "This Agreement expires three years after the Effective Date.")]),
          data.Query("q2", "Consider Fiverr's privacy policy; do they collect my IP address?", "privacy_qa",
                     [_span("privacy_qa/Fiverr.txt", "We collect your IP address and browsing data.")])]
    scores = evaluate.run([CFG], CORPUS, qs)
    table = metrics.macro(scores[CFG.name], qs)
    assert table["ALL"]["doc@1"] == 1.0 and table["ALL"]["R@8"] == 1.0
