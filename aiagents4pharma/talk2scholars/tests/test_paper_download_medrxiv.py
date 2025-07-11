"""tests for the DownloadBiorxivPaperInput tool."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import hydra
import pytest
import requests

from aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input import (
    DownloadMedrxivPaperInput,
)


@pytest.fixture(autouse=True)
def mock_hydra(monkeypatch):
    """Make hydra.initialize a no-op context manager and hydra.compose return a dummy config."""

    @contextmanager
    def dummy_initialize(*args, **kwargs):
        args = list(args)
        kwargs = list(kwargs)
        yield

    monkeypatch.setattr(hydra, "initialize", dummy_initialize)
    dummy_cfg = SimpleNamespace(
        tools=SimpleNamespace(
            download_medrxiv_paper=SimpleNamespace(
                api_url="http://api.test/", request_timeout=9
            )
        )
    )
    monkeypatch.setattr(hydra, "compose", lambda *args, **kwargs: dummy_cfg)
    yield


def test_load_hydra_configs_returns_expected():
    """Hydra configurations success"""
    tool = DownloadMedrxivPaperInput()
    cfg = tool.load_hydra_configs()
    assert cfg.api_url == "http://api.test/"
    assert cfg.request_timeout == 9


def test_fetch_metadata_success_and_version_stripping(monkeypatch):
    """Fetch metadata success with version stripping"""
    seen = {}

    def fake_get(url, timeout):
        _=timeout
        seen["url"] = url
        resp = mock.Mock()
        resp.raise_for_status = mock.Mock()
        resp.json = mock.Mock(return_value={"collection": [{"id": "X"}]})
        return resp

    monkeypatch.setattr(requests, "get", fake_get)
    tool = DownloadMedrxivPaperInput()
    # include a version suffix
    result = tool.fetch_metadata("http://api.test/", "10.1101/ABCv3")
    assert result == {"id": "X"}
    # version 'v3' stripped
    assert seen["url"] == "http://api.test/10.1101/ABC"


def test_fetch_metadata_raises_http_error(monkeypatch):
    """fetch metadata raises http error"""
    resp = mock.Mock()
    resp.raise_for_status.side_effect = requests.exceptions.HTTPError("fail")
    monkeypatch.setattr(requests, "get", lambda url, timeout: resp)
    tool = DownloadMedrxivPaperInput()
    with pytest.raises(requests.exceptions.HTTPError):
        tool.fetch_metadata("u", "10.1101/DEFv1")


def test_fetch_metadata_empty_collection_raises(monkeypatch):
    """fetch metadata finds no collection"""
    resp = mock.Mock()
    resp.raise_for_status = mock.Mock()
    resp.json = mock.Mock(return_value={"collection": []})
    monkeypatch.setattr(requests, "get", lambda url, timeout: resp)
    tool = DownloadMedrxivPaperInput()
    with pytest.raises(ValueError) as exc:
        tool.fetch_metadata("u", "10.1101/GHIv2")
    assert "No metadata found for DOI: 10.1101/GHI" in str(exc.value)


def test_extract_metadata_success(monkeypatch):
    """ectract metadata works successfully"""
    data = {
        "title": "MedRxiv Title",
        "authors": ["Dr A", "Dr B"],
        "abstract": "Summary text",
        "date": "2025-07-07",
        "doi": "10.1101/XYZ123",
    }
    # simulate PDF accessible
    monkeypatch.setattr(
        requests, "get", lambda url, timeout: mock.Mock(status_code=200)
    )

    tool = DownloadMedrxivPaperInput()
    out = tool.extract_metadata(data, "10.1101/XYZ123")
    expected_url = "https://www.medrxiv.org/content/10.1101/XYZ123.full.pdf"

    assert out["Title"] == "MedRxiv Title"
    assert out["Authors"] == ["Dr A", "Dr B"]
    assert out["Abstract"] == "Summary text"
    assert out["Publication Date"] == "2025-07-07"
    assert out["URL"] == expected_url
    assert out["pdf_url"] == expected_url
    assert out["filename"] == "XYZ123.pdf"
    assert out["source"] == "medrxiv"
    assert out["medrxiv_id"] == "10.1101/XYZ123"


def test_extract_metadata_pdf_not_accessible(monkeypatch, capsys):
    """Pdf not accessible in extract metadata"""
    data = {"doi": "10.1101/NOP456"}
    # simulate PDF not accessible
    monkeypatch.setattr(
        requests, "get", lambda url, timeout: mock.Mock(status_code=404)
    )

    tool = DownloadMedrxivPaperInput()
    result = tool.extract_metadata(data, "10.1101/NOP456")
    # Should print warning and return empty dict
    captured = capsys.readouterr()
    assert (
        "No PDF found or access denied at https://www.medrxiv.org/content/10.1101/NOP456.full.pdf"
        in captured.out
    )
    assert not result


def test_paper_retriever_happy_and_skip(monkeypatch):
    """Paper retirver happy path test"""
    tool = DownloadMedrxivPaperInput()
    fake_cfg = SimpleNamespace(api_url="http://api.test/", request_timeout=4)
    monkeypatch.setattr(tool, "load_hydra_configs", lambda: fake_cfg)

    # stub fetch_metadata; content irrelevant for extract
    monkeypatch.setattr(tool, "fetch_metadata", lambda url, pid: {"dummy": True})

    # extract_metadata returns non-empty for 'good', empty for 'bad'
    def fake_extract(data, pid):
        data= list(data)
        return {"id": pid} if pid == "good" else {}

    monkeypatch.setattr(tool, "extract_metadata", fake_extract)

    result = tool.paper_retriever(["medrxiv:good", "medrxiv:bad"])
    assert "article_data" in result
    # only "good" should appear
    assert set(result["article_data"].keys()) == {"good"}
    assert result["article_data"]["good"]["id"] == "good"
