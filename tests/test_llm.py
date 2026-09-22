from paperchat.llm import _format_context, clean_out_of_range_citations


def test_clean_out_of_range_citations_drops_invalid_numbers():
    text = "a [1] b [5] c [1, 4] d [1, 7]"
    assert clean_out_of_range_citations(text, max_index=4) == "a [1] b c [1, 4] d [1]"


def test_clean_out_of_range_citations_keeps_all_valid_numbers():
    text = "nothing to clean [1] [2]"
    assert clean_out_of_range_citations(text, max_index=2) == text


def test_clean_out_of_range_citations_with_zero_chunks_strips_everything():
    assert clean_out_of_range_citations("claim [1]", max_index=0) == "claim"


def test_format_context_strips_source_papers_own_citation_markers():
    chunks = [
        {
            "doc_path": "/tmp/paper.pdf",
            "page": 0,
            "text": "long short-term memory [13] and gated recurrent [7] units",
        }
    ]
    context = _format_context(chunks)
    assert "[13]" not in context
    assert "[7]" not in context
    # our own excerpt label [1] must survive
    assert context.startswith("[1] (paper.pdf, page 1):")
