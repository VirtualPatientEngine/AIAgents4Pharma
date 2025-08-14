"""generate_answer tests for the PDF tool"""

from unittest.mock import MagicMock

import pytest

from aiagents4pharma.talk2scholars.tools.pdf.utils.generate_answer import (
    _build_context_and_sources,
    generate_answer,
)


@pytest.fixture(name="chunks_fixture")
def _chunks_fixture():
    """Fixture providing sample document chunks."""
    doc1 = MagicMock()
    doc1.page_content = "This is chunk one."
    doc1.metadata = {"paper_id": "P1", "title": "Title 1", "page": 1}

    doc2 = MagicMock()
    doc2.page_content = "This is chunk two."
    doc2.metadata = {"paper_id": "P1", "title": "Title 1", "page": 2}

    doc3 = MagicMock()
    doc3.page_content = "This is chunk three."
    doc3.metadata = {"paper_id": "P2", "title": "Title 2", "page": 1}

    return [doc1, doc2, doc3]


def test_build_context_and_sources_formatting(chunks_fixture):
    """_build_context_and_sources should format context and sources correctly."""
    context, sources = _build_context_and_sources(chunks_fixture)

    assert "[Document 1] From: 'Title 1' (ID: P1)" in context
    assert "Page 1: This is chunk one." in context
    assert "Page 2: This is chunk two." in context
    assert "[Document 2] From: 'Title 2' (ID: P2)" in context
    assert "Page 1: This is chunk three." in context
    assert sources == {"P1", "P2"}


def test_generate_answer_success(chunks_fixture):
    """generate_answer should return formatted answer and sources."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "The answer is XYZ."

    config = {
        "prompt_template": "Answer the question based on the context."
        "\n\n{context}\n\nQ: {question}\nA:"
    }

    result = generate_answer("What is the result?", chunks_fixture, mock_llm, config)

    assert result["output_text"] == "The answer is XYZ."
    assert len(result["sources"]) == 3
    assert result["num_sources"] == 3
    assert set(result["papers_used"]) == {"P1", "P2"}


def test_generate_answer_raises_for_none_config(chunks_fixture):
    """generate_answer should raise ValueError for None config."""
    mock_llm = MagicMock()
    with pytest.raises(ValueError, match="Configuration for generate_answer is required."):
        generate_answer("Why?", chunks_fixture, mock_llm, config=None)


def test_generate_answer_raises_for_missing_template(chunks_fixture):
    """generate_answer should raise ValueError for missing prompt_template in config."""
    mock_llm = MagicMock()
    with pytest.raises(ValueError, match="The prompt_template is missing from the configuration."):
        generate_answer("Why?", chunks_fixture, mock_llm, config={})
