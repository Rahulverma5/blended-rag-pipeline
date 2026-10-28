import pytest

from blended_rag import chunking

CONTRACT = (
    "AGENCY AGREEMENT\n"
    "ARTICLE I  GENERAL UNDERTAKING\n"
    "Section 1.01 Appointment. Lessor hereby appoints the Company as its agent. " + "The agent shall act. " * 40 + "\n"
    "Section 1.02 Powers. The Company may sign purchase orders.\n"
    "ARTICLE II  DUTIES OF AGENT\n"
    "Section 2.01 Equipment Orders. Upon written acknowledgment the Company shall order equipment.\n"
)


@pytest.mark.parametrize("strategy", ["structure", "naive"])
def test_chunks_respect_size_and_offsets(strategy):
    chunks = chunking.chunk_document("cuad/X_AGENCY AGREEMENT.txt", CONTRACT, strategy, max_chars=200)
    assert chunks
    for c in chunks:
        assert 0 <= c.start < c.end <= len(CONTRACT)
        assert c.end - c.start <= (200 if strategy == "structure" else 500)
        assert CONTRACT[c.start:c.end] == CONTRACT[c.start:c.end].strip()
    # no overlap, in order
    assert all(a.end <= b.start for a, b in zip(chunks, chunks[1:]))


def test_structure_chunks_cover_all_non_space_text():
    chunks = chunking.chunk_document("a.txt", CONTRACT, max_chars=200)
    covered = set().union(*(range(c.start, c.end) for c in chunks))
    assert all(i in covered for i, ch in enumerate(CONTRACT) if not ch.isspace())


def test_chunks_start_at_headings_and_carry_them():
    chunks = chunking.chunk_document("a.txt", CONTRACT, max_chars=200)
    powers = next(c for c in chunks if "may sign purchase orders" in CONTRACT[c.start:c.end])
    assert CONTRACT[powers.start:powers.end].startswith("Section 1.02")
    assert "Section 1.02 Powers." in powers.section
    orders = next(c for c in chunks if "order equipment" in CONTRACT[c.start:c.end])
    assert "Section 2.01 Equipment Orders." in orders.section


def test_continuation_chunk_inherits_previous_heading():
    chunks = chunking.chunk_document("a.txt", CONTRACT, max_chars=200)
    body = [c for c in chunks if CONTRACT[c.start:c.end].startswith("The agent shall act")]
    assert body and all(c.section == "Section 1.01 Appointment." for c in body)


def test_small_pieces_are_merged_into_neighbours():
    chunks = chunking.chunk_document("a.txt", CONTRACT, max_chars=200)
    assert not any(CONTRACT[c.start:c.end] == "ARTICLE II  DUTIES OF AGENT" for c in chunks)


def test_headings_found_inline_and_on_own_line():
    text = "ARTICLE 3   MERGER TRANSACTION   Section 3.1    Merger of Purchaser. Upon the terms"
    found = [h for _, h in chunking.headings(text)]
    assert found[0].startswith("ARTICLE 3") and any(h.startswith("Section 3.1") for h in found)


def test_doc_title_from_file_name():
    assert chunking.doc_title("maud/Slack Technologies, Inc._salesforce.com, inc..txt") == \
        "Slack Technologies, Inc. salesforce.com, inc."
    assert chunking.doc_title("maud/Grubhub Inc._Just Eat.pdf||Grubhub Amendment No. 1.txt") == "Grubhub Inc. Just Eat"
    assert chunking.doc_title("contractnli/VELCO%20NDA%20rev0.txt") == "VELCO NDA rev0"
    assert chunking.doc_title("contractnli/ExcelerateStandardNDAFormat.txt") == "Excelerate Standard NDA Format"
    assert chunking.doc_title("contractnli/Kenway-NDA-Form-Blank.txt") == "Kenway NDA Form Blank"
