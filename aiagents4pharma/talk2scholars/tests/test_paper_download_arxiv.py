"""Test cases for the DownloadArxivPaperInput tool in aiagents4pharma.talk2scholars."""

import xml.etree.ElementTree as ET
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import hydra
import pytest
import requests

from aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input import (
    DownloadArxivPaperInput,
)
from aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input import (
    logger as arxiv_logger,
)


@pytest.fixture(autouse=True)
def mock_hydra(monkeypatch):
    """Prevent real Hydra from running; make initialize a no-op context manager and compose return a dummy config."""

    @contextmanager
    def dummy_initialize(*args, **kwargs):
        yield

    monkeypatch.setattr(hydra, "initialize", dummy_initialize)
    dummy_cfg = SimpleNamespace(
        tools=SimpleNamespace(
            download_arxiv_paper=SimpleNamespace(
                api_url="http://api.test", request_timeout=5
            )
        )
    )
    monkeypatch.setattr(hydra, "compose", lambda *args, **kwargs: dummy_cfg)
    yield


def make_simple_feed(entry_xml: str) -> ET.Element:
    """Helper: wrap an <entry> fragment in a <feed> root with Atom namespace."""
    feed = ET.Element("{http://www.w3.org/2005/Atom}feed")
    entry = ET.fromstring(entry_xml)
    feed.append(entry)
    return feed


def test_load_hydra_configs_returns_expected():
    tool = DownloadArxivPaperInput()
    cfg = tool.load_hydra_configs()
    assert cfg.api_url == "http://api.test"
    assert cfg.request_timeout == 5


def test_fetch_metadata_success(monkeypatch):
    xml_body = "<feed></feed>"
    fake_resp = mock.Mock()
    fake_resp.text = xml_body
    fake_resp.raise_for_status = mock.Mock()
    monkeypatch.setattr(requests, "get", lambda url, timeout: fake_resp)

    tool = DownloadArxivPaperInput()
    out = tool.fetch_metadata("http://api.test", "1234.5678")
    assert "data" in out
    assert isinstance(out["data"], ET.Element)


def test_fetch_metadata_raises_on_http_error(monkeypatch):
    fake_resp = mock.Mock()
    fake_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("boom")
    monkeypatch.setattr(requests, "get", lambda url, timeout: fake_resp)

    tool = DownloadArxivPaperInput()
    with pytest.raises(requests.exceptions.HTTPError):
        tool.fetch_metadata("http://api.test", "1234.5678")


def test_extract_metadata_all_fields_present():
    entry_xml = """
    <entry xmlns="http://www.w3.org/2005/Atom">
      <title>  My Title  </title>
      <author><name>Alice</name></author>
      <author><name>Bob</name></author>
      <summary>  An abstract.  </summary>
      <published>2025-07-07T12:00:00Z</published>
      <link href="http://example.com/other.pdf" title="other"/>
      <link href="http://example.com/paper.pdf" title="pdf"/>
    </entry>
    """
    xml_root = make_simple_feed(entry_xml)
    data = {"data": xml_root}
    tool = DownloadArxivPaperInput()
    out = tool.extract_metadata(data, "1234.5678")

    assert out["Title"] == "My Title"
    assert out["Authors"] == ["Alice", "Bob"]
    assert out["Abstract"] == "An abstract."
    assert out["Publication Date"] == "2025-07-07T12:00:00Z"
    assert out["URL"] == "http://example.com/paper.pdf"
    assert out["pdf_url"] == "http://example.com/paper.pdf"
    assert out["filename"] == "1234.5678.pdf"
    assert out["source"] == "arxiv"
    assert out["arxiv_id"] == "1234.5678"


def test_extract_metadata_missing_pdf_raises():
    entry_xml = """
    <entry xmlns="http://www.w3.org/2005/Atom">
      <title>Title</title>
      <summary>Abstract</summary>
    </entry>
    """
    xml_root = make_simple_feed(entry_xml)
    data = {"data": xml_root}
    tool = DownloadArxivPaperInput()
    with pytest.raises(RuntimeError) as exc:
        tool.extract_metadata(data, "9999.9999")
    assert "Could not find PDF URL for arXiv ID 9999.9999" in str(exc.value)


def test_extract_metadata_missing_optional_fields():
    entry_xml = """
    <entry xmlns="http://www.w3.org/2005/Atom">
      <link href="http://example.com/foo.pdf" title="pdf"/>
    </entry>
    """
    xml_root = make_simple_feed(entry_xml)
    data = {"data": xml_root}
    tool = DownloadArxivPaperInput()
    out = tool.extract_metadata(data, "0000.0000")
    assert out["Title"] == "N/A"
    assert out["Authors"] == []
    assert out["Abstract"] == "N/A"
    assert out["Publication Date"] == "N/A"
    assert out["pdf_url"] == "http://example.com/foo.pdf"


def test_paper_retriever_happy_path(monkeypatch):
    tool = DownloadArxivPaperInput()
    fake_cfg = SimpleNamespace(api_url="http://api.test", request_timeout=3)
    monkeypatch.setattr(tool, "load_hydra_configs", lambda: fake_cfg)

    entry_xml = """
    <entry xmlns="http://www.w3.org/2005/Atom">
      <link href="http://example.com/1.pdf" title="pdf"/>
    </entry>
    """
    xml_root = {"data": make_simple_feed(entry_xml)}
    monkeypatch.setattr(tool, "fetch_metadata", lambda url, pid: xml_root)
    monkeypatch.setattr(
        tool,
        "extract_metadata",
        lambda data, pid: {"foo": "bar", "arxiv_id": pid},
    )

    result = tool.paper_retriever(["arxiv:1", "arxiv:2"])
    assert set(result["article_data"].keys()) == {"1", "2"}
    assert result["article_data"]["1"]["foo"] == "bar"


def test_paper_retriever_invokes_extract_when_entry_present(monkeypatch):
    # Arrange
    tool = DownloadArxivPaperInput()
    fake_cfg = SimpleNamespace(api_url="u", request_timeout=1)
    monkeypatch.setattr(tool, "load_hydra_configs", lambda: fake_cfg)

    # Build a feed with one <entry> so extract_metadata will be called
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = ET.Element("{http://www.w3.org/2005/Atom}entry")
    feed = ET.Element("{http://www.w3.org/2005/Atom}feed")
    feed.append(entry)

    # fetch_metadata returns our fake feed
    monkeypatch.setattr(tool, "fetch_metadata", lambda url, pid: {"data": feed})

    # Spy on extract_metadata
    called = False

    def fake_extract(data, pid):
        nonlocal called
        called = True
        return {"foo": "bar", "arxiv_id": pid}

    monkeypatch.setattr(tool, "extract_metadata", fake_extract)

    # Act
    result = tool.paper_retriever(["arxiv:XYZ"])

    # Assert
    assert called, "extract_metadata should have been called when <entry> is present"
    assert result == {"article_data": {"XYZ": {"foo": "bar", "arxiv_id": "XYZ"}}}


def test_paper_retriever_calls_warning_without_entry(monkeypatch):

    # Arrange
    tool = DownloadArxivPaperInput()
    fake_cfg = SimpleNamespace(api_url="u", request_timeout=1)
    monkeypatch.setattr(tool, "load_hydra_configs", lambda: fake_cfg)

    # A feed with no <entry>
    empty_feed = ET.Element("{http://www.w3.org/2005/Atom}feed")
    monkeypatch.setattr(tool, "fetch_metadata", lambda url, pid: {"data": empty_feed})

    # Spy on the warning call
    warning_called = False

    def fake_warning(msg, aid):
        nonlocal warning_called
        warning_called = True
        # optional: sanity‐check the message / aid
        assert "No xml_root found for arXiv ID" in msg
        assert aid == "XYZ"

    monkeypatch.setattr(arxiv_logger, "warning", fake_warning)

    # Act
    result = tool.paper_retriever(["arxiv:XYZ"])

    # Assert
    assert result == {"article_data": {}}
    assert warning_called, "Expected logger.warning to be called when no entry is found"
