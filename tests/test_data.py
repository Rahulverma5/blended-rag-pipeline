import zipfile

from blended_rag import data
from blended_rag.data import Query, Snippet


def _q(qid, dataset, doc):
    return Query(qid, "?", dataset, [Snippet(doc, 0, 1)])


def test_committed_mini_matches_paper_counts():
    corpus, queries = data.load_mini()
    assert len(corpus) == 72 and len(queries) == 776
    assert sum(map(len, corpus.values())) == 8_682_104
    assert all(s.file_path in corpus and 0 <= s.start < s.end <= len(corpus[s.file_path])
               for q in queries for s in q.snippets)


def test_dev_test_split_never_shares_a_document():
    qs = [_q(f"{d}{i}", d, f"{d}/{i % 5}.txt") for d in data.DATASETS for i in range(20)]
    dev, test = data.dev_test_split(qs)
    assert dev and test and not {q.doc for q in dev} & {q.doc for q in test}
    assert len(dev) + len(test) == len(qs)


def test_select_mini_is_deterministic_and_document_grouped():
    qs = [_q(str(i), "cuad", f"cuad/{i % 50}.txt") for i in range(500)]
    a, b = data.select_mini(qs), data.select_mini(list(reversed(qs)))
    assert len(a) == data.MINI_PER_DATASET and [q.qid for q in a] == [q.qid for q in data.select_mini(qs)]
    # Ties within the cut-off document follow input order (stable sort, as upstream), but the documents don't.
    assert {q.doc for q in a} == {q.doc for q in b}
    assert len({q.doc for q in a}) <= 20  # 10 queries per doc -> the 194 sit on ~20 docs, not 50


def test_safe_extract_keeps_dotted_names_and_drops_escapes(tmp_path):
    zp = tmp_path / "x.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("corpus/Acme Inc..txt", "ok")
        z.writestr("../evil.txt", "no")
    data.safe_extract(zp, tmp_path / "out")
    assert (tmp_path / "out/corpus/Acme Inc..txt").read_text() == "ok"
    assert not (tmp_path / "evil.txt").exists()
