from blended_rag.routing import DocRouter

TEMPLATE = "MUTUAL NON-DISCLOSURE AGREEMENT. The Receiving Party shall not disclose Confidential Information. "
CORPUS = {
    "contractnli/Mutual-Non-Disclosure-Agreement-Template.txt": TEMPLATE * 5,
    "contractnli/Kenway-NDA-Form-Blank.txt": "This agreement is made by Kenway Corp. " + TEMPLATE,
    "contractnli/AGProjects-NDA.txt": "AG Projects and the Recipient agree. " + TEMPLATE,
    "maud/Slack Technologies, Inc._salesforce.com, inc..txt": "AGREEMENT AND PLAN OF MERGER among salesforce.com",
}


def test_distinctive_name_beats_generic_title_words():
    r = DocRouter(CORPUS)
    q = "Consider Kenway's Non-Disclosure Agreement; Does the Receiving Party have to return Confidential Information?"
    assert r.route(q)[0] == "contractnli/Kenway-NDA-Form-Blank.txt"


def test_unsplit_camel_case_name_matches():
    r = DocRouter(CORPUS)
    assert r.route("Consider the AGProjects' Non-Disclosure Agreement; is it mutual?")[0] == "contractnli/AGProjects-NDA.txt"


def test_route_respects_max_docs_and_margin():
    r = DocRouter(CORPUS, max_docs=2, margin=100.0)
    assert len(r.route("Does the agreement cover confidential information?")) == 2
    r = DocRouter(CORPUS, margin=0.0)
    assert r.route('Consider the merger between Target "Slack Technologies, Inc." and salesforce.com')[0].startswith("maud/")
