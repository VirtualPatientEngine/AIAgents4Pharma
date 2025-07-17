"""tests for the paper download tool."""
# pylint: disable=redefined-outer-name

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from aiagents4pharma.talk2scholars.tools.paper_download import download_tool


@pytest.fixture(autouse=True)
def mock_build_summary():
    """Always return a fixed summary."""
    with patch.object(download_tool, "build_summary", lambda data: "SUMMARY"):
        yield


@pytest.fixture
def fake_llm_model_tuple():
    """A fake LLM with structured output."""
    llm = MagicMock()
    structured = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm, structured


def make_state(llm_model):
    """Making state"""
    return {"llm_model": llm_model}


def test_arxiv_only_path(fake_llm_model_tuple, monkeypatch):
    """Test only arxiv ids"""
    llm = fake_llm_model_tuple[0]
    structured = fake_llm_model_tuple[1]
    # only arxiv_ids
    structured.invoke.return_value = SimpleNamespace(
        arxiv_ids=["arxiv:1"], dois=[], pubmed_ids=[]
    )
    # stub out retrievers
    monkeypatch.setattr(
        download_tool.DownloadArxivPaperInput,
        "paper_retriever",
        lambda self, paper_ids: {"article_data": {"1": {"from": "arxiv"}}},
    )

    # Call the underlying function, not the BaseTool wrapper
    cmd: Command = download_tool.download_paper.func(
        paper_id=["arxiv:1"],
        tool_call_id="TCID",
        state=make_state(llm),
    )
    upd = cmd.update
    assert isinstance(cmd, Command)
    assert upd["article_data"] == {"1": {"from": "arxiv"}}
    msgs = upd["messages"]
    assert len(msgs) == 1
    tm = msgs[0]
    assert isinstance(tm, ToolMessage)
    assert tm.content == "SUMMARY"
    assert tm.tool_call_id == "TCID"
    assert tm.artifact == {"1": {"from": "arxiv"}}


def test_pubmed_only_path(fake_llm_model_tuple, monkeypatch):
    """Testing only pubmed ids"""
    llm = fake_llm_model_tuple[0]
    structured = fake_llm_model_tuple[1]
    structured.invoke.return_value = SimpleNamespace(
        arxiv_ids=[], dois=[], pubmed_ids=["pmc:123"]
    )
    monkeypatch.setattr(
        download_tool.DownloadPubmedPaperInput,
        "paper_retriever",
        lambda self, paper_ids: {"article_data": {"123": {"from": "pubmed"}}},
    )

    cmd = download_tool.download_paper.func(
        paper_id=["pmc:123"],
        tool_call_id="T2",
        state=make_state(llm),
    )
    assert cmd.update["article_data"] == {"123": {"from": "pubmed"}}


def test_biorxiv_success_path(fake_llm_model_tuple, monkeypatch):
    """Testing only biorxiv papers"""
    llm = fake_llm_model_tuple[0]
    structured = fake_llm_model_tuple[1]
    structured.invoke.return_value = SimpleNamespace(
        arxiv_ids=[], dois=["10.1101/ABC"], pubmed_ids=[]
    )
    monkeypatch.setattr(
        download_tool.DownloadBiorxivPaperInput,
        "paper_retriever",
        lambda self, paper_ids: {"article_data": {"ABC": {"src": "biorxiv"}}},
    )

    cmd = download_tool.download_paper.func(
        paper_id=["10.1101/ABC"],
        tool_call_id="T3",
        state=make_state(llm),
    )
    assert cmd.update["article_data"] == {"ABC": {"src": "biorxiv"}}


def test_biorxiv_fails_then_medrxiv(fake_llm_model_tuple, monkeypatch):
    """Medrxiv runs if biorxiv fails"""
    llm = fake_llm_model_tuple[0]
    structured = fake_llm_model_tuple[1]
    structured.invoke.return_value = SimpleNamespace(
        arxiv_ids=[], dois=["10.1101/XYZ"], pubmed_ids=[]
    )
    # Biorxiv raises → Medrxiv used
    monkeypatch.setattr(
        download_tool.DownloadBiorxivPaperInput,
        "paper_retriever",
        lambda self, paper_ids: (_ for _ in ()).throw(ValueError("no meta")),
    )
    monkeypatch.setattr(
        download_tool.DownloadMedrxivPaperInput,
        "paper_retriever",
        lambda self, paper_ids: {"article_data": {"XYZ": {"src": "medrxiv"}}},
    )

    cmd = download_tool.download_paper.func(
        paper_id=["10.1101/XYZ"],
        tool_call_id="T4",
        state=make_state(llm),
    )
    assert cmd.update["article_data"] == {"XYZ": {"src": "medrxiv"}}


def test_multiple_sources_combined(fake_llm_model_tuple, monkeypatch):
    """Multiple sources combined test run"""
    llm = fake_llm_model_tuple[0]
    structured = fake_llm_model_tuple[1]
    structured.invoke.return_value = SimpleNamespace(
        arxiv_ids=["arxiv:9"], dois=["10.1101/DO"], pubmed_ids=["pmc:7"]
    )
    monkeypatch.setattr(
        download_tool.DownloadArxivPaperInput,
        "paper_retriever",
        lambda self, paper_ids: {"article_data": {"9": {"A": 1}}},
    )
    monkeypatch.setattr(
        download_tool.DownloadPubmedPaperInput,
        "paper_retriever",
        lambda self, paper_ids: {"article_data": {"7": {"P": 2}}},
    )
    monkeypatch.setattr(
        download_tool.DownloadBiorxivPaperInput,
        "paper_retriever",
        lambda self, paper_ids: {"article_data": {"DO": {"D": 3}}},
    )

    cmd = download_tool.download_paper.func(
        paper_id=["dummy"],
        tool_call_id="T5",
        state=make_state(llm),
    )
    art = cmd.update["article_data"]
    assert art == {"9": {"A": 1}, "7": {"P": 2}, "DO": {"D": 3}}


def test_no_ids_results_in_empty(fake_llm_model_tuple):
    """If llm returns no ids"""
    llm = fake_llm_model_tuple[0]
    structured = fake_llm_model_tuple[1]
    structured.invoke.return_value = SimpleNamespace(
        arxiv_ids=[], dois=[], pubmed_ids=[]
    )

    cmd = download_tool.download_paper.func(
        paper_id=["nothing"],
        tool_call_id="T6",
        state=make_state(llm),
    )
    assert cmd.update["article_data"] == {}
