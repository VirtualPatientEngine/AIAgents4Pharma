"""retrieve_chunks for PDF tool tests"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks import (
    retrieve_relevant_chunks,
    retrieve_relevant_chunks_with_scores,
)


@pytest.fixture
def mock_vector_store():
    """Fixture to simulate a vector store."""
    return MagicMock()


@pytest.fixture
def mock_chunks():
    """Fixture to simulate PDF chunks."""
    return [
        Document(page_content=f"chunk {i}", metadata={"paper_id": f"P{i % 2}"}) for i in range(5)
    ]


@pytest.fixture
def mock_scored_chunks():
    """Fixture to simulate scored PDF chunks."""
    return [
        (Document(page_content=f"chunk {i}", metadata={}), score)
        for i, score in enumerate([0.9, 0.8, 0.4, 0.95])
    ]


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_cpu_success(mock_logger, request):
    """Test retrieve_relevant_chunks with CPU path."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    vector_store.max_marginal_relevance_search.return_value = chunks

    results = retrieve_relevant_chunks(vector_store, query="AI", top_k=5)

    assert results == chunks
    vector_store.max_marginal_relevance_search.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_gpu_success(mock_logger, request):
    """Test retrieve_relevant_chunks with GPU path."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    vector_store.has_gpu = True
    mock_logger.debug = MagicMock()
    vector_store.max_marginal_relevance_search.return_value = chunks

    results = retrieve_relevant_chunks(vector_store, query="AI", top_k=5)

    assert results == chunks
    vector_store.max_marginal_relevance_search.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_filter(mock_logger, request):
    """Test retrieve_relevant_chunks with paper_id filter."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    vector_store.has_gpu = False
    mock_logger.debug = MagicMock()
    vector_store.max_marginal_relevance_search.return_value = chunks

    results = retrieve_relevant_chunks(vector_store, query="filter test", paper_ids=["P1"], top_k=3)
    assert results == chunks
    args, kwargs = vector_store.max_marginal_relevance_search.call_args
    assert len(args) == 0
    assert kwargs["filter"] == {"paper_id": ["P1"]}


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_no_vector_store(mock_logger):
    """Test when vector store is None."""
    result = retrieve_relevant_chunks(vector_store=None, query="irrelevant")
    assert result == []
    mock_logger.error.assert_called_with("Vector store is not initialized")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_no_vector_store(mock_logger):
    """Test retrieve_relevant_chunks_with_scores when vector store is None."""
    result = retrieve_relevant_chunks_with_scores(vector_store=None, query="none")
    assert result == []
    mock_logger.error.assert_called_with("Vector store is not initialized")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_default_search_params(mock_logger, request):
    """Test default search params used when not defined."""
    vector_store = request.getfixturevalue("mock_vector_store")
    chunks = request.getfixturevalue("mock_chunks")
    vector_store.has_gpu = False
    delattr(vector_store, "search_params")
    vector_store.max_marginal_relevance_search.return_value = chunks

    results = retrieve_relevant_chunks(
        vector_store,
        query="default search param test",
        top_k=5,
    )

    assert results == chunks
    mock_logger.debug.assert_any_call("Using default search parameters (no hardware optimization)")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_paper_filter(mock_logger, request):
    """Test retrieve_relevant_chunks_with_scores applies paper_id filter."""
    vector_store = request.getfixturevalue("mock_vector_store")
    scored_chunks = request.getfixturevalue("mock_scored_chunks")
    vector_store.similarity_search_with_score.return_value = scored_chunks
    mock_logger.debug = MagicMock()

    results = retrieve_relevant_chunks_with_scores(
        vector_store=vector_store,
        query="filtered score",
        paper_ids=["P123"],
        top_k=5,
        score_threshold=0.0,
    )

    assert isinstance(results, list)
    assert vector_store.similarity_search_with_score.call_args[1]["filter"] == {
        "paper_id": ["P123"]
    }


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_gpu_debug(mock_logger, request):
    """Test GPU debug log and correct return in retrieve_relevant_chunks_with_scores."""
    vector_store = request.getfixturevalue("mock_vector_store")
    scored_chunks = request.getfixturevalue("mock_scored_chunks")
    vector_store.has_gpu = True
    vector_store.similarity_search_with_score.return_value = scored_chunks
    mock_logger.debug = MagicMock()

    results = retrieve_relevant_chunks_with_scores(
        vector_store=vector_store, query="gpu test", top_k=4, score_threshold=0.0
    )

    # Should return all scored_chunks since threshold=0.0
    assert results == scored_chunks
    mock_logger.debug.assert_called_with("GPU-accelerated similarity search enabled")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_cpu_debug(mock_logger, request):
    """Test CPU debug log and correct return in retrieve_relevant_chunks_with_scores."""
    vector_store = request.getfixturevalue("mock_vector_store")
    scored_chunks = request.getfixturevalue("mock_scored_chunks")
    vector_store.has_gpu = False
    vector_store.similarity_search_with_score.return_value = scored_chunks
    mock_logger.debug = MagicMock()

    results = retrieve_relevant_chunks_with_scores(
        vector_store=vector_store, query="cpu test", top_k=2, score_threshold=0.0
    )

    assert results == scored_chunks
    mock_logger.debug.assert_called_with("Standard CPU similarity search")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.retrieve_chunks.logger")
def test_retrieve_chunks_with_scores_not_implemented(mock_logger, request):
    """Test NotImplementedError path when similarity_search_with_score is missing."""
    vector_store = request.getfixturevalue("mock_vector_store")
    vector_store.has_gpu = True
    # Remove the method to trigger NotImplementedError
    if hasattr(vector_store, "similarity_search_with_score"):
        delattr(vector_store, "similarity_search_with_score")
    mock_logger.debug = MagicMock()

    with pytest.raises(NotImplementedError) as excinfo:
        retrieve_relevant_chunks_with_scores(
            vector_store=vector_store, query="fail test", top_k=1, score_threshold=0.0
        )
    assert "Vector store does not support similarity_search_with_score" in str(excinfo.value)
    mock_logger.debug.assert_called_with("GPU-accelerated similarity search enabled")
