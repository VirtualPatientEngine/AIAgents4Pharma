"""tests for paper summary builder functions."""

import pytest

from aiagents4pharma.talk2scholars.tools.paper_download.utils.summary_builder import (
    _get_snippet,
    build_summary,
)


@pytest.mark.parametrize(
    "abstract, expected",
    [
        ("", ""),
        ("N/A", ""),
        ("Single sentence", "Single sentence."),
        ("One. Second.", "One. Second."),
        (
            "First sentence. Second sentence. Third sentence.",
            "First sentence. Second sentence.",
        ),
    ],
)
def test_get_snippet_various(abstract, expected):
    """_get_snippet should handle empty, 'N/A', single, and multi-sentence abstracts."""
    assert _get_snippet(abstract) == expected


def test_build_summary_empty():
    """build_summary on empty data should mention zero papers and have no entries."""
    summary = build_summary({})
    assert "Number of papers found: 0" in summary
    # After 'Top 3 papers:' there should be a newline but no numbered entries
    assert summary.endswith("Top 3 papers:\n")


def test_build_summary_single_no_url_no_snippet():
    """One paper, no URL and no snippet → only title line is present."""
    data = {
        "p1": {
            "Title": "Test Paper",
            "Publication Date": "2025-07-07",
            "URL": "",
            "Abstract": "",
        }
    }
    summary = build_summary(data)
    assert "Number of papers found: 1" in summary
    # Check the first line
    assert "1. Test Paper (2025-07-07)" in summary
    # No 'View PDF:' and no 'Abstract snippet:' lines
    assert "View PDF:" not in summary
    assert "Abstract snippet:" not in summary


def test_build_summary_multiple_with_url_and_snippet():
    """Two papers, each with URL and enough abstract for a snippet."""
    data = {
        "p1": {
            "Title": "Alpha",
            "Publication Date": "2025-01-01",
            "URL": "http://a.pdf",
            "Abstract": "Alpha abstract first. Alpha second sentence. Extra.",
        },
        "p2": {
            "Title": "Beta",
            "Publication Date": "2025-02-02",
            "URL": "http://b.pdf",
            "Abstract": "Beta abstract only one sentence",
        },
    }
    summary = build_summary(data)

    # Should list 2 papers
    assert "Number of papers found: 2" in summary
    # Check p1 entry
    assert "1. Alpha (2025-01-01)" in summary
    assert "View PDF: http://a.pdf" in summary
    # Snippet should be first two sentences
    assert "Abstract snippet: Alpha abstract first. Alpha second sentence." in summary
    # Check p2 entry
    assert "2. Beta (2025-02-02)" in summary
    assert "View PDF: http://b.pdf" in summary
    # Snippet for p2 is single sentence + "."
    assert "Abstract snippet: Beta abstract only one sentence." in summary


def test_build_summary_limits_to_top_three():
    """If more than three papers are given, only top 3 are included."""
    data = {}
    for i in range(5):
        data[f"id{i}"] = {
            "Title": f"T{i}",
            "Publication Date": f"2025-0{i+1}-0{i+1}",
            "URL": f"http://{i}.pdf",
            "Abstract": f"Abs{i}.",
        }

    summary = build_summary(data)
    # Should report 5 found, but only list 3
    assert "Number of papers found: 5" in summary
    # Should contain entries 1., 2., 3. but not 4.
    assert "1. T0" in summary
    assert "2. T1" in summary
    assert "3. T2" in summary
    assert "4. T3" not in summary
