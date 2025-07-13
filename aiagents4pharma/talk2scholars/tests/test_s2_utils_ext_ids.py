"""
Unit tests for external ID handling in S2 helper modules.
"""

from types import SimpleNamespace

import hydra
import pytest
import requests

from aiagents4pharma.talk2scholars.tools.s2.utils.multi_helper import MultiPaperRecData
from aiagents4pharma.talk2scholars.tools.s2.utils.search_helper import SearchData
from aiagents4pharma.talk2scholars.tools.s2.utils.single_helper import (
    SinglePaperRecData,
)


@pytest.fixture(autouse=True)
def patch_hydra(monkeypatch):
    """Patch Hydra's initialize and compose to provide dummy configs for tests."""

    class DummyHydraContext:
        """Dummy Hydra context manager to bypass config loading."""

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_val, traceback):
            return False

    # Dummy config with necessary fields for multi, search, and single helpers
    dummy_cfg = SimpleNamespace(
        tools=SimpleNamespace(
            multi_paper_recommendation=SimpleNamespace(
                api_endpoint="",
                headers={},
                api_fields=["paperId", "title", "authors", "externalIds"],
                request_timeout=1,
            ),
            search=SimpleNamespace(
                api_endpoint="",
                api_fields=["paperId", "title", "authors", "externalIds"],
            ),
            single_paper_recommendation=SimpleNamespace(
                api_endpoint="",
                api_fields=["paperId", "title", "authors", "externalIds"],
                request_timeout=1,
                recommendation_params=SimpleNamespace(from_pool="test_pool"),
            ),
        )
    )
    monkeypatch.setattr(
        hydra, "initialize", lambda version_base, config_path: DummyHydraContext()
    )
    monkeypatch.setattr(hydra, "compose", lambda config_name, overrides: dummy_cfg)


def test_multi_helper_pmc_and_doi_ids(monkeypatch):
    """Test PubMedCentral and DOI ID handling in MultiPaperRecData."""
    rec = MultiPaperRecData(paper_ids=["p"], limit=1, year=None, tool_call_id="tid")
    # Setup dummy API response
    data = {
        "recommendedPapers": [
            {
                "paperId": "p1",
                "title": "Test",
                "authors": [{"name": "A", "authorId": "A1"}],
                "externalIds": {"PubMedCentral": "pmc1", "DOI": "doi1"},
            }
        ]
    }
    response = SimpleNamespace(
        status_code=200, json=lambda: data, raise_for_status=lambda: None
    )
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    results = rec.process_recommendations()
    ids_list = results["papers"]["p1"]["paper_ids"]
    assert ids_list == ["pmc:pmc1", "doi:doi1"]


def test_search_helper_pmc_and_doi_ids(monkeypatch):
    """Test PubMedCentral and DOI ID handling in SearchData."""
    sd = SearchData(query="q", limit=1, year=None, tool_call_id="tid")
    data = {
        "data": [
            {
                "paperId": "s1",
                "title": "Test",
                "authors": [{"name": "B", "authorId": "B1"}],
                "externalIds": {"PubMedCentral": "pmc2", "DOI": "doi2"},
            }
        ]
    }
    response = SimpleNamespace(
        status_code=200, json=lambda: data, raise_for_status=lambda: None
    )
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    results = sd.process_search()
    ids_list = results["papers"]["s1"]["paper_ids"]
    assert ids_list == ["pmc:pmc2", "doi:doi2"]


def test_single_helper_pmc_and_doi_ids(monkeypatch):
    """Test PubMedCentral and DOI ID handling in SinglePaperRecData."""
    sp = SinglePaperRecData(paper_id="x", limit=1, year=None, tool_call_id="tid")
    data = {
        "recommendedPapers": [
            {
                "paperId": "x1",
                "title": "Test",
                "authors": [{"name": "C", "authorId": "C1"}],
                "externalIds": {"PubMedCentral": "pmc3", "DOI": "doi3"},
            }
        ]
    }
    response = SimpleNamespace(
        status_code=200, json=lambda: data, raise_for_status=lambda: None
    )
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    results = sp.process_recommendations()
    ids_list = results["papers"]["x1"]["paper_ids"]
    assert ids_list == ["pmc:pmc3", "doi:doi3"]


def test_helpers_empty_when_no_external_ids(monkeypatch):
    """Test that MultiPaperRecData, SearchData, and SinglePaperRecData
    return empty lists when externalIds are missing or empty."""
    # Test that no IDs are returned when externalIds is empty or missing
    rec = MultiPaperRecData(paper_ids=["p"], limit=1, year=None, tool_call_id="tid")

    # Simulate empty externalIds in API response
    class DummyResp1:
        """dummy response for multi-paper recommendation with empty externalIds"""

        def __init__(self, data):
            """initialize with data"""
            self._data = data
            self.status_code = 200

        def json(self):
            """json method to return data"""
            return self._data

        def raise_for_status(self):
            """raise_for_status method to simulate successful response"""
            return None

    def dummy_post1(*_, **__):
        """dummy response for multi-paper recommendation with empty externalIds"""
        return DummyResp1(
            {
                "recommendedPapers": [
                    {
                        "paperId": "p2",
                        "title": "Test2",
                        "authors": [{"name": "D", "authorId": "D1"}],
                        "externalIds": {},
                    },
                ]
            }
        )

    monkeypatch.setattr(requests, "post", dummy_post1)
    assert rec.process_recommendations()["papers"].get("p2", {}).get("paper_ids") == []
    sd = SearchData(query="q2", limit=1, year=None, tool_call_id="tid2")

    # Simulate empty externalIds in search API response
    class DummyResp2:
        """dummy response for search with empty externalIds"""

        def __init__(self, data):
            """initialize with data"""
            self._data = data
            self.status_code = 200

        def json(self):
            """json method to return data"""
            return self._data

        def raise_for_status(self):
            """raise_for_status method to simulate successful response"""
            return None

    def dummy_get2(*_, **__):
        """dummy response for search with empty externalIds"""
        return DummyResp2(
            {
                "data": [
                    {
                        "paperId": "s2",
                        "title": "Test2",
                        "authors": [{"name": "E", "authorId": "E1"}],
                        "externalIds": {},
                    },
                ]
            }
        )

    monkeypatch.setattr(requests, "get", dummy_get2)
    assert sd.process_search()["papers"].get("s2", {}).get("paper_ids") == []
    sp = SinglePaperRecData(paper_id="y", limit=1, year=None, tool_call_id="tid3")

    # Simulate empty externalIds in single-paper API response
    class DummyResp3:
        """dummy response for single paper recommendation with empty externalIds"""

        def __init__(self, data):
            """initialize with data"""
            self._data = data
            self.status_code = 200

        def json(self):
            """json method to return data"""
            return self._data

        def raise_for_status(self):
            """raise_for_status method to simulate successful response"""
            return None

    def dummy_get3(*_, **__):
        """dummy response for single paper recommendation with empty externalIds"""
        return DummyResp3(
            {
                "recommendedPapers": [
                    {
                        "paperId": "y1",
                        "title": "Test3",
                        "authors": [{"name": "F", "authorId": "F1"}],
                        "externalIds": {},
                    },
                ]
            }
        )

    monkeypatch.setattr(requests, "get", dummy_get3)
    assert sp.process_recommendations()["papers"].get("y1", {}).get("paper_ids") == []


def test_multi_helper_arxiv_and_pubmed_ids(monkeypatch):
    """Test ArXiv and PubMed ID handling in MultiPaperRecData."""
    rec = MultiPaperRecData(paper_ids=["p"], limit=1, year=None, tool_call_id="tid")

    class DummyResp5:
        """dummy response for multi-paper recommendation with ArXiv and PubMed IDs"""

        def __init__(self, data):
            """initialize with data"""
            self._data = data
            self.status_code = 200

        def json(self):
            """json method to return data"""
            return self._data

        def raise_for_status(self):
            """raise_for_status method to simulate successful response"""
            return None

    def dummy_post5(*_, **__):
        """dummy response for multi-paper recommendation with ArXiv and PubMed IDs"""
        return DummyResp5(
            {
                "recommendedPapers": [
                    {
                        "paperId": "pX",
                        "title": "TestX",
                        "authors": [{"name": "A", "authorId": "A1"}],
                        "externalIds": {"ArXiv": "ax1", "PubMed": "pm1"},
                    },
                ]
            }
        )

    monkeypatch.setattr(requests, "post", dummy_post5)
    ids_list = rec.process_recommendations()["papers"].get("pX", {}).get("paper_ids")
    assert ids_list == ["arxiv:ax1", "pubmed:pm1"]


def test_search_helper_arxiv_and_pubmed_ids(monkeypatch):
    """Test ArXiv and PubMed ID handling in SearchData."""
    sd = SearchData(query="q", limit=1, year=None, tool_call_id="tid")

    class DummyResp6:
        """dummy response for search with ArXiv and PubMed IDs"""

        def __init__(self, data):
            """initialize with data"""
            self._data = data
            self.status_code = 200

        def json(self):
            """json method to return data"""
            return self._data

        def raise_for_status(self):
            """ "raise_for_status method to simulate successful response"""
            return None

    def dummy_get6(*_, **__):
        """dummy response for search with ArXiv and PubMed IDs"""
        return DummyResp6(
            {
                "data": [
                    {
                        "paperId": "sX",
                        "title": "TestS",
                        "authors": [{"name": "B", "authorId": "B1"}],
                        "externalIds": {"ArXiv": "ax2", "PubMed": "pm2"},
                    },
                ]
            }
        )

    monkeypatch.setattr(requests, "get", dummy_get6)
    ids_list = sd.process_search()["papers"].get("sX", {}).get("paper_ids")
    assert ids_list == ["arxiv:ax2", "pubmed:pm2"]


def test_single_helper_arxiv_and_pubmed_ids(monkeypatch):
    """Test ArXiv and PubMed ID handling in SinglePaperRecData."""
    sp = SinglePaperRecData(paper_id="x", limit=1, year=None, tool_call_id="tid")

    class DummyResp7:
        """dummy response for single paper recommendation with ArXiv and PubMed IDs"""

        def __init__(self, data):
            """initialize with data"""
            self._data = data
            self.status_code = 200

        def json(self):
            """json method to return data"""
            return self._data

        def raise_for_status(self):
            """raise_for_status method to simulate successful response"""
            return None

    def dummy_get7(*_, **__):
        """dummy response for single paper recommendation with ArXiv and PubMed IDs"""
        return DummyResp7(
            {
                "recommendedPapers": [
                    {
                        "paperId": "xY",
                        "title": "TestY",
                        "authors": [{"name": "C", "authorId": "C1"}],
                        "externalIds": {"ArXiv": "ax3", "PubMed": "pm3"},
                    },
                ]
            }
        )

    monkeypatch.setattr(requests, "get", dummy_get7)
    ids_list = sp.process_recommendations()["papers"].get("xY", {}).get("paper_ids")
    assert ids_list == ["arxiv:ax3", "pubmed:pm3"]


def test_search_helper_create_content_snippet(monkeypatch):
    """Test that SearchData._create_content includes snippets appropriately."""
    sd = SearchData(query="QueryX", limit=3, year="2022", tool_call_id="tid")
    sd.filtered_papers = {
        "p1": {"Title": "Title1", "Year": "2021", "Abstract": "First. Second. Third."},
        "p2": {"Title": "Title2", "Year": "2020", "Abstract": ""},
    }
    # Stub out network fetch/filter to rely on preset filtered_papers
    monkeypatch.setattr(SearchData, "_fetch_papers", lambda self: None)
    monkeypatch.setattr(SearchData, "_filter_papers", lambda self: None)
    results = sd.process_search()
    content = results["content"]
    assert "1. Title1 (2021)" in content
    assert "Abstract snippet: First. Second." in content
    assert "2. Title2 (2020)" in content
    # Only one snippet present
    assert content.count("Abstract snippet:") == 1


def test_single_helper_create_content_snippet(monkeypatch):
    """Test that SinglePaperRecData._create_content includes snippets appropriately."""
    sp = SinglePaperRecData(paper_id="pid", limit=2, year=None, tool_call_id="tid")
    sp.filtered_papers = {
        "x1": {"Title": "STitle1", "Year": "2019", "Abstract": "SOne. STwo. SThree."},
        "x2": {"Title": "STitle2", "Year": "2018", "Abstract": ""},
    }
    # Stub out network fetch/filter to rely on preset filtered_papers
    monkeypatch.setattr(SinglePaperRecData, "_fetch_recommendations", lambda self: None)
    monkeypatch.setattr(SinglePaperRecData, "_filter_papers", lambda self: None)
    results = sp.process_recommendations()
    content = results["content"]
    assert "1. STitle1 (2019)" in content
    assert "Abstract snippet: SOne. STwo." in content
    assert "2. STitle2 (2018)" in content
    assert content.count("Abstract snippet:") == 1


def test_multi_helper_create_content_snippet(monkeypatch):
    """Test that MultiPaperRecData._create_content includes snippets appropriately."""
    mr = MultiPaperRecData(
        paper_ids=["a", "b"], limit=2, year="2021", tool_call_id="tid"
    )
    mr.filtered_papers = {
        "m1": {"Title": "MTitle1", "Year": "2017", "Abstract": "MOne. MTwo. MThree."},
        "m2": {"Title": "MTitle2", "Year": "2016", "Abstract": ""},
    }
    # Stub out network fetch/filter to rely on preset filtered_papers
    monkeypatch.setattr(MultiPaperRecData, "_fetch_recommendations", lambda self: None)
    monkeypatch.setattr(MultiPaperRecData, "_filter_papers", lambda self: None)
    results = mr.process_recommendations()
    content = results["content"]
    assert "1. MTitle1 (2017)" in content
    assert "Abstract snippet: MOne. MTwo." in content
    assert "2. MTitle2 (2016)" in content
    assert content.count("Abstract snippet:") == 1
