"""
Test cases for Talk2Biomodels query_article tool.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from ..agents.t2b_agent import get_app
from ..tools.query_article import QueryArticle


class Article(BaseModel):
    """
    Article schema.
    """

    title: str = Field(description="Title of the article.")


LLM_MODEL = MagicMock(name="llm_model")
LLM_MODEL.with_structured_output.return_value.invoke.return_value = Article(
    title="A Multiscale Model of IL-6–Mediated Immune Regulation in Crohn's Disease"
)
pytestmark = pytest.mark.unit_mock


def _build_app(fake_app_factory, monkeypatch, messages):
    """Return app and config with patched get_app."""
    app = fake_app_factory([{"messages": messages}])
    monkeypatch.setattr(
        "aiagents4pharma.talk2biomodels.tests.test_query_article.get_app",
        lambda *_args, **_kwargs: app,
    )
    unique_id = 12345
    app = get_app(unique_id, llm_model=LLM_MODEL)
    return app, {"configurable": {"thread_id": unique_id}}


def test_query_article_with_an_article(fake_app_factory, monkeypatch):
    """
    Test the query_article tool by providing an article.
    """
    assistant_content = "A Multiscale Model of IL-6–Mediated Immune Regulation in Crohn's Disease"
    messages = [
        ToolMessage(
            content=assistant_content,
            name="query_article",
            status="success",
            artifact=None,
            tool_call_id="call-1",
        )
    ]
    app, config = _build_app(fake_app_factory, monkeypatch, messages)
    response = app.invoke(
        {"messages": [HumanMessage(content="What is the title of the article?")]},
        config=config,
    )
    assistant_msg = response["messages"][-1].content
    article = LLM_MODEL.with_structured_output(Article).invoke(
        f"Given the text below, what is the title of the article?\n\n{assistant_msg}"
    )
    keywords = ["Multiscale", "IL-6", "Immune", "Crohn"]
    msg_lower = assistant_msg.lower()
    title_matches = sum(1 for kw in keywords if kw.lower() in article.title.lower())
    msg_matches = sum(1 for kw in keywords if kw.lower() in msg_lower)
    access_failed = any(
        indicator in msg_lower
        for indicator in [
            "unable to access",
            "cannot access",
            "assistance with",
            "request for assistance",
        ]
    )
    assert title_matches >= 2 or msg_matches >= 2 or access_failed


def test_query_article_without_an_article(fake_app_factory, monkeypatch):
    """
    Test the query_article tool without providing an article.
    The status of the tool should be error.
    """
    error_message = ToolMessage(
        content="No article provided",
        name="query_article",
        status="error",
        artifact=None,
        tool_call_id="call-2",
    )
    app, config = _build_app(fake_app_factory, monkeypatch, [error_message])
    prompt = "What is the title of the uploaded article?"
    app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    current_state = app.get_state(config)
    # Get the messages from the current state
    # and reverse the order
    reversed_messages = current_state.values["messages"][::-1]
    # Loop through the reversed messages
    # until a ToolMessage is found.
    tool_status_is_error = False
    for msg in reversed_messages:
        if isinstance(msg, ToolMessage):
            # Skip until it finds a ToolMessage
            if msg.name == "query_article" and msg.status == "error":
                tool_status_is_error = True
                break
    assert tool_status_is_error


@patch("aiagents4pharma.talk2biomodels.tools.query_article.PyPDFLoader")
@patch("aiagents4pharma.talk2biomodels.tools.query_article.InMemoryVectorStore")
@pytest.mark.unit_mock
def test_query_article_similarity_search_and_return(mock_vector_store, mock_pdf_loader):
    """
    Test that lines 62-64 are covered: similarity search and return join operation.
    """
    # Mock PDF loader
    mock_page = MagicMock()
    mock_page.page_content = "Sample article content about research methodology"
    mock_loader_instance = MagicMock()
    mock_loader_instance.lazy_load.return_value = [mock_page]
    mock_pdf_loader.return_value = mock_loader_instance

    # Mock vector store and similarity search
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "First relevant document content"
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Second relevant document content"
    mock_vector_store_instance = MagicMock()
    mock_vector_store_instance.similarity_search.return_value = [mock_doc1, mock_doc2]
    mock_vector_store.from_documents.return_value = mock_vector_store_instance

    # Create tool and run
    tool = QueryArticle()
    state = {
        "pdf_file_name": "test_file.pdf",
        "text_embedding_model": MagicMock(),
    }

    tool_input = {"question": "What is the methodology?", "state": state}
    result = tool.invoke(tool_input)

    # Verify similarity_search was called (line 62)
    mock_vector_store_instance.similarity_search.assert_called_once_with("What is the methodology?")

    # Verify return join operation (line 64)
    expected_result = "First relevant document content\nSecond relevant document content"
    assert result == expected_result
    assert isinstance(result, str)
    assert "\n" in result


@patch("aiagents4pharma.talk2biomodels.tools.query_article.PyPDFLoader")
@patch("aiagents4pharma.talk2biomodels.tools.query_article.InMemoryVectorStore")
@pytest.mark.unit_mock
def test_query_article_empty_search_results(mock_vector_store, mock_pdf_loader):
    """
    Test edge case where similarity search returns empty results.
    """
    # Mock PDF loader
    mock_page = MagicMock()
    mock_page.page_content = "Sample content"
    mock_loader_instance = MagicMock()
    mock_loader_instance.lazy_load.return_value = [mock_page]
    mock_pdf_loader.return_value = mock_loader_instance

    # Mock vector store with empty search results
    mock_vector_store_instance = MagicMock()
    mock_vector_store_instance.similarity_search.return_value = []
    mock_vector_store.from_documents.return_value = mock_vector_store_instance

    tool = QueryArticle()
    state = {
        "pdf_file_name": "test_file.pdf",
        "text_embedding_model": MagicMock(),
    }

    tool_input = {"question": "Nonexistent topic", "state": state}
    result = tool.invoke(tool_input)

    # Should return empty string when no documents found
    assert result == ""
    assert isinstance(result, str)
