"""
Unit tests for external ID handling in S2 helper modules.
"""

from types import SimpleNamespace

import hydra
import pytest

from aiagents4pharma.talk2scholars.tools.s2.utils.multi_helper import MultiPaperRecData
from aiagents4pharma.talk2scholars.tools.s2.utils.search_helper import SearchData
from aiagents4pharma.talk2scholars.tools.s2.utils.single_helper import (
    SinglePaperRecData,
)


@pytest.fixture(autouse=True)
def patch_hydra(monkeypatch):
    """Patch Hydra's initialize and compose to provide dummy configs for tests."""

    class DummyHydraContext:
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


def test_multi_helper_pmc_and_doi_ids():
    """test that MultiPaperRecData correctly filters and formats PubMedCentral and DOI IDs."""
    rec = MultiPaperRecData(paper_ids=["p"], limit=1, year=None, tool_call_id="tid")
    # Recommendation with PubMedCentral and DOI external IDs
    rec.recommendations = [
        {
            "paperId": "p1",
            "title": "Test",
            "authors": [{"name": "A", "authorId": "A1"}],
            "externalIds": {"PubMedCentral": "pmc1", "DOI": "doi1"},
        }
    ]
    rec._filter_papers()
    ids_list = rec.filtered_papers.get("p1", {}).get("paper_ids")
    assert ids_list == ["pmc:pmc1", "doi:doi1"]


def test_search_helper_pmc_and_doi_ids():
    """test that SearchData correctly filters and formats PubMedCentral and DOI IDs."""
    sd = SearchData(query="q", limit=1, year=None, tool_call_id="tid")
    # Paper with PubMedCentral and DOI external IDs
    sd.papers = [
        {
            "paperId": "s1",
            "title": "Test",
            "authors": [{"name": "B", "authorId": "B1"}],
            "externalIds": {"PubMedCentral": "pmc2", "DOI": "doi2"},
        }
    ]
    sd._filter_papers()
    ids_list = sd.filtered_papers.get("s1", {}).get("paper_ids")
    assert ids_list == ["pmc:pmc2", "doi:doi2"]


def test_single_helper_pmc_and_doi_ids():
    """test that SinglePaperRecData correctly filters and formats PubMedCentral and DOI IDs."""
    sp = SinglePaperRecData(paper_id="x", limit=1, year=None, tool_call_id="tid")
    # Recommendation with PubMedCentral and DOI external IDs
    sp.recommendations = [
        {
            "paperId": "x1",
            "title": "Test",
            "authors": [{"name": "C", "authorId": "C1"}],
            "externalIds": {"PubMedCentral": "pmc3", "DOI": "doi3"},
        }
    ]
    sp._filter_papers()
    ids_list = sp.filtered_papers.get("x1", {}).get("paper_ids")
    assert ids_list == ["pmc:pmc3", "doi:doi3"]


def test_helpers_empty_when_no_external_ids():
    """test that MultiPaperRecData, SearchData, and SinglePaperRecData
    return empty lists when externalIds are missing or empty."""
    # Test that no IDs are returned when externalIds is empty or missing
    rec = MultiPaperRecData(paper_ids=["p"], limit=1, year=None, tool_call_id="tid")
    rec.recommendations = [
        {
            "paperId": "p2",
            "title": "Test2",
            "authors": [{"name": "D", "authorId": "D1"}],
            # externalIds missing keys
            "externalIds": {},
        }
    ]
    rec._filter_papers()
    assert rec.filtered_papers.get("p2", {}).get("paper_ids") == []
    sd = SearchData(query="q2", limit=1, year=None, tool_call_id="tid2")
    sd.papers = [
        {
            "paperId": "s2",
            "title": "Test2",
            "authors": [{"name": "E", "authorId": "E1"}],
            "externalIds": {},
        }
    ]
    sd._filter_papers()
    assert sd.filtered_papers.get("s2", {}).get("paper_ids") == []
    sp = SinglePaperRecData(paper_id="y", limit=1, year=None, tool_call_id="tid3")
    sp.recommendations = [
        {
            "paperId": "y1",
            "title": "Test3",
            "authors": [{"name": "F", "authorId": "F1"}],
            "externalIds": {},
        }
    ]
    sp._filter_papers()
    assert sp.filtered_papers.get("y1", {}).get("paper_ids") == []


def test_multi_helper_arxiv_and_pubmed_ids():
    """test that MultiPaperRecData correctly filters and formats ArXiv and PubMed IDs."""
    rec = MultiPaperRecData(paper_ids=["p"], limit=1, year=None, tool_call_id="tid")
    rec.recommendations = [
        {
            "paperId": "pX",
            "title": "TestX",
            "authors": [{"name": "A", "authorId": "A1"}],
            "externalIds": {"ArXiv": "ax1", "PubMed": "pm1"},
        }
    ]
    rec._filter_papers()
    ids_list = rec.filtered_papers.get("pX", {}).get("paper_ids")
    assert ids_list == ["arxiv:ax1", "pubmed:pm1"]


def test_search_helper_arxiv_and_pubmed_ids():
    """test that SearchData correctly filters and formats ArXiv and PubMed IDs."""
    sd = SearchData(query="q", limit=1, year=None, tool_call_id="tid")
    sd.papers = [
        {
            "paperId": "sX",
            "title": "TestS",
            "authors": [{"name": "B", "authorId": "B1"}],
            "externalIds": {"ArXiv": "ax2", "PubMed": "pm2"},
        }
    ]
    sd._filter_papers()
    ids_list = sd.filtered_papers.get("sX", {}).get("paper_ids")
    assert ids_list == ["arxiv:ax2", "pubmed:pm2"]


def test_single_helper_arxiv_and_pubmed_ids():
    """test that SinglePaperRecData correctly filters and formats ArXiv and PubMed IDs."""
    sp = SinglePaperRecData(paper_id="x", limit=1, year=None, tool_call_id="tid")
    sp.recommendations = [
        {
            "paperId": "xY",
            "title": "TestY",
            "authors": [{"name": "C", "authorId": "C1"}],
            "externalIds": {"ArXiv": "ax3", "PubMed": "pm3"},
        }
    ]
    sp._filter_papers()
    ids_list = sp.filtered_papers.get("xY", {}).get("paper_ids")
    assert ids_list == ["arxiv:ax3", "pubmed:pm3"]
