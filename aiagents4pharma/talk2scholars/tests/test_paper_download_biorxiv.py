"""tests for the DownloadBiorxivPaperInput tool."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import hydra
import pytest
import requests

from aiagents4pharma.talk2scholars.tools.paper_download.download_biorxiv_input import (
    DownloadBiorxivPaperInput,
)


@pytest.fixture(autouse=True)
def mock_hydra(monkeypatch):
    """Make hydra.initialize a no‐op context manager and hydra.compose return a dummy config."""

    @contextmanager
    def dummy_initialize(*args,**kwargs):
        args = list(args)
        kwargs = list(kwargs)
        yield

    monkeypatch.setattr(hydra, "initialize", dummy_initialize)
    dummy_cfg = SimpleNamespace(
        tools=SimpleNamespace(
            download_biorxiv_paper=SimpleNamespace(
                api_url="http://api.test/", request_timeout=7
            )
        )
    )
    monkeypatch.setattr(hydra, "compose", lambda *args, **kwargs: dummy_cfg)
    yield


def test_load_hydra_configs_returns_expected():
    """Testing hydra configuratiojs"""
    tool = DownloadBiorxivPaperInput()
    cfg = tool.load_hydra_configs()
    assert cfg.api_url == "http://api.test/"
    assert cfg.request_timeout == 7


def test_fetch_metadata_success_and_version_stripping(monkeypatch):
    """Fetch metadata success version stripping test"""
    calls = {}

    def fake_get(url, timeout):
        _ = timeout
        calls["url"] = url
        # simulate a successful HTTP response with JSON payload
        resp = mock.Mock()
        resp.raise_for_status = mock.Mock()
        resp.json = mock.Mock(return_value={"collection": [{"foo": "bar"}]})
        return resp

    monkeypatch.setattr(requests, "get", fake_get)

    tool = DownloadBiorxivPaperInput()
    # include a version suffix in the paper_id
    result = tool.fetch_metadata("http://api.test/", "10.1101/XYZv2")
    assert result == {"foo": "bar"}
    # ensure the version suffix was stripped from the URL
    assert calls["url"] == "http://api.test/10.1101/XYZ"


def test_fetch_metadata_no_collection_raises(monkeypatch):
    """No collection return test run"""
    # simulate HTTP okay but empty collection
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json = mock.Mock(return_value={"collection": []})
    monkeypatch.setattr(requests, "get", lambda url, timeout: resp)

    tool = DownloadBiorxivPaperInput()
    with pytest.raises(ValueError) as exc:
        tool.fetch_metadata("u", "10.1101/ABCv1")
    assert "No metadata found for DOI: 10.1101/ABC" in str(exc.value)


def test_extract_metadata_success(monkeypatch):
    """Extract metadata success test run"""
    # prepare a fake entry dict
    data = {
        "title": "My BioRxiv Title",
        "authors": ["Alice", "Bob"],
        "abstract": "An abstract.",
        "date": "2025-07-07",
        "doi": "10.1101/12345",
    }
    # simulate PDF being available
    fake_pdf_resp = mock.Mock(status_code=200)
    monkeypatch.setattr(requests, "get", lambda url, timeout: fake_pdf_resp)

    tool = DownloadBiorxivPaperInput()
    out = tool.extract_metadata(data, "10.1101/12345")
    assert out["Title"] == "My BioRxiv Title"
    assert out["Authors"] == ["Alice", "Bob"]
    assert out["Abstract"] == "An abstract."
    assert out["Publication Date"] == "2025-07-07"
    expected_url = "https://www.biorxiv.org/content/10.1101/12345.full.pdf"
    assert out["URL"] == expected_url
    assert out["pdf_url"] == expected_url
    assert out["filename"] == "12345.pdf"
    assert out["source"] == "biorxiv"
    assert out["biorxiv_id"] == "10.1101/12345"


def test_extract_metadata_pdf_not_accessible(monkeypatch, capsys):
    """PDF not accessible test run"""
    data = {"doi": "10.1101/ZZZ"}
    # simulate PDF not accessible
    fake_pdf_resp = mock.Mock(status_code=404)
    monkeypatch.setattr(requests, "get", lambda url, timeout: fake_pdf_resp)

    tool = DownloadBiorxivPaperInput()
    with pytest.raises(ValueError, match="Pdf not accessible"):
        tool.extract_metadata(data, "10.1101/ZZZ")
    captured = capsys.readouterr()
    assert (
        "No PDF found or access denied at https://www.biorxiv.org/content/10.1101/ZZZ.full.pdf"
        in captured.out
    )


def test_paper_retriever_happy_and_skip_paths(monkeypatch):
    """Paper retriever happy run with skips"""
    tool = DownloadBiorxivPaperInput()
    # stub config
    fake_cfg = SimpleNamespace(api_url="u/", request_timeout=2)
    monkeypatch.setattr(tool, "load_hydra_configs", lambda: fake_cfg)
    # stub fetch_metadata (its return value is ignored by our extract stub)
    monkeypatch.setattr(tool, "fetch_metadata", lambda url, doi: {"dummy": True})

    # make extract_metadata return a real dict for "good" and an empty dict (falsey) for "bad"
    def fake_extract(data, doi):
        data = list(data)
        return {} if doi == "bad" else {"val": doi}

    monkeypatch.setattr(tool, "extract_metadata", fake_extract)

    result = tool.paper_retriever(["biorxiv:good", "biorxiv:bad"])
    # only "good" should remain
    assert "article_data" in result
    assert set(result["article_data"].keys()) == {"good"}
    assert result["article_data"]["good"]["val"] == "good"
