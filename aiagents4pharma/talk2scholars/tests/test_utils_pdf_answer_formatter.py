"""answer_formatter tests."""

from unittest.mock import patch

import pytest

from aiagents4pharma.talk2scholars.tools.pdf.utils.answer_formatter import format_answer


@pytest.fixture(name="base_args")
def _base_args():
    """base_args fixture to provide common arguments for tests."""
    return {
        "question": "What is the conclusion?",
        "chunks": [{"content": "chunk1"}, {"content": "chunk2"}],
        "llm": "mock_llm",
        "articles": {
            "paper1": {"Title": "Paper One"},
            "paper2": {"Title": "Paper Two"},
        },
        "config": {"key": "value"},
        "call_id": "test_call_123",
        "has_gpu": True,
    }


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.answer_formatter.generate_answer")
def test_format_answer_with_sources(mock_generate_answer, base_args):
    """test format_answer with sources."""
    mock_generate_answer.return_value = {
        "output_text": "This is the generated answer.",
        "papers_used": ["paper1", "paper2"],
    }

    result = format_answer(**base_args)

    assert "This is the generated answer." in result
    assert "Sources:" in result
    assert "- Paper One" in result
    assert "- Paper Two" in result
    mock_generate_answer.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.answer_formatter.generate_answer")
def test_format_answer_no_sources(mock_generate_answer, base_args):
    """test format_answer with no sources."""
    mock_generate_answer.return_value = {
        "output_text": "No sources were used.",
        "papers_used": [],  # No papers used
    }

    result = format_answer(**base_args)

    assert result == "No sources were used."  # No sources section expected
    mock_generate_answer.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.answer_formatter.generate_answer")
def test_format_answer_missing_output_text(mock_generate_answer, base_args):
    """test format_answer with missing output text."""
    mock_generate_answer.return_value = {"papers_used": ["paper1"]}

    result = format_answer(**base_args)

    assert result.startswith("No answer generated.")
    assert "Sources:" in result
    mock_generate_answer.assert_called_once()
