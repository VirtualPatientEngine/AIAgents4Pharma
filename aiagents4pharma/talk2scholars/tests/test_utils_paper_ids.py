# """
# Unit tests for paper_ids logic in S2 helper modules.
# """
#
# from types import SimpleNamespace
#
# import hydra
# import pytest
# import requests
#
# from aiagents4pharma.talk2scholars.tools.s2.utils.multi_helper import MultiPaperRecData
# from aiagents4pharma.talk2scholars.tools.s2.utils.search_helper import SearchData
# from aiagents4pharma.talk2scholars.tools.s2.utils.single_helper import (
#     SinglePaperRecData,
# )
#
#
# @pytest.fixture
# def dummy_config(monkeypatch):
#     """dummy hydra config for testing SearchData"""
#     # Patch Hydra config for SearchData
#
#     class DummyHydraContext:
#         """dummy hydra context manager to avoid actual config loading"""
#
#         def __enter__(self):
#             """dummy enter method"""
#             return None
#
#         def __exit__(self, exc_type, exc, tb):
#             """empty exit method"""
#             return False
#
#     dummy_cfg = SimpleNamespace(
#         tools=SimpleNamespace(
#             search=SimpleNamespace(
#                 api_endpoint="http://dummy",
#                 api_fields=["paperId"],
#             )
#         )
#     )
#     monkeypatch.setattr(
#         hydra, "initialize", lambda version_base, config_path: DummyHydraContext()
#     )
#     monkeypatch.setattr(hydra, "compose", lambda config_name, overrides: dummy_cfg)
#     return dummy_cfg
#
#
# def test_multi_helper_paper_ids_and_filter():
#     """ "test MultiPaperRecData paper_ids and filtering logic"""
#     rec = MultiPaperRecData(paper_ids=["id1"], limit=1, year=None, tool_call_id="t1")
#     rec.recommendations = [
#         {
#             "paperId": "p1",
#             "title": "T1",
#             "authors": [{"name": "A"}],
#             "year": 2020,
#             "citationCount": 0,
#             "url": "u1",
#             "externalIds": {"ArXiv": "x1", "PubMed": "m1"},
#         },
#         {
#             "paperId": "p2",
#             "title": "T2",
#             "authors": None,
#             "year": 2021,
#             "citationCount": 0,
#             "url": "u2",
#             "externalIds": {},
#         },
#     ]
#     rec._filter_papers()
#     # Only paper1 should be present
#     assert list(rec.filtered_papers.keys()) == ["p1"]
#     meta = rec.filtered_papers["p1"]
#     # paper_ids should include arxiv and pubmed
#     assert meta["paper_ids"] == ["arxiv:x1", "pubmed:m1"]
#
#
# def test_single_helper_paper_ids_and_filter():
#     """test SinglePaperRecData paper_ids and filtering logic"""
#     rec = SinglePaperRecData(paper_id="p3", limit=1, year=None, tool_call_id="t2")
#     rec.recommendations = [
#         {
#             "paperId": "p3",
#             "title": "T3",
#             "authors": [{"name": "B"}],
#             "year": 2019,
#             "citationCount": 0,
#             "url": "u3",
#             "externalIds": {"PubMedCentral": "c1"},
#         },
#     ]
#     rec._filter_papers()
#     assert list(rec.filtered_papers.keys()) == ["p3"]
#     meta = rec.filtered_papers["p3"]
#     assert meta["paper_ids"] == ["pmc:c1"]
#     # Test missing externalIds produces empty list
#     rec2 = SinglePaperRecData(paper_id="p4", limit=1, year=None, tool_call_id="t3")
#     rec2.recommendations = [
#         {
#             "paperId": "p4",
#             "title": "T4",
#             "authors": [{"name": "D"}],
#             "year": 2022,
#             "citationCount": 0,
#             "url": "u4",
#             "externalIds": {},
#         },
#     ]
#     rec2._filter_papers()
#     meta2 = rec2.filtered_papers["p4"]
#     assert meta2["paper_ids"] == []
#
#
# def test_search_helper_paper_ids_and_content(dummy_config):
#     """test SearchData paper_ids and content generation"""
#     sd = SearchData(query="q", limit=1, year=None, tool_call_id="t3")
#     # Patch requests.get to return expected JSON
#
#     class DummyResponse:
#         """dummy response class to mock requests.get"""
#
#         def __init__(self, data):
#             """initialize with data"""
#             self.data = data
#
#         def raise_for_status(self):
#             """raise for status"""
#             pass
#
#         def json(self):
#             """simulate json response"""
#             return self.data
#
#     data = {
#         "data": [
#             {
#                 "paperId": "p4",
#                 "title": "T4",
#                 "authors": [{"name": "C"}],
#                 "year": 2018,
#                 "citationCount": 0,
#                 "url": "u4",
#                 "externalIds": {"DOI": "d1"},
#             }
#         ]
#     }
#     requests.get = lambda endpoint, params, timeout: DummyResponse(data)
#     # Run search
#     res = sd.process_search()
#     papers = res["papers"]
#     # Check paper_ids present
#     assert papers["p4"]["paper_ids"] == ["doi:d1"]
#     # Check content includes Top 3 section
#     assert "Top 3 papers" in res["content"]
#     # Test with year filter includes Year line
#     sd2 = SearchData(query="q2", limit=1, year="2021", tool_call_id="t4")
#     sd2.data = data
#     sd2.papers = data["data"]
#     sd2.filtered_papers = {"p4": {
#         **sd2.filtered_papers.get("p4", {}),
#         **sd2.filtered_papers.get("p4", {})
#     }}  # ensure not empty
#     sd2._create_content()
#     assert "Year: 2021" in sd2.content
#
# def test_multi_helper_content():
#     rec = MultiPaperRecData(paper_ids=["x"], limit=1, year=None, tool_call_id="t5")
#     rec.recommendations = [
#         {"paperId": "pa", "title": "Tpa", "authors": [{"name": "A"}],
#          "year": 2022, "citationCount": 0, "url": "upa",
#          "externalIds": {"ArXiv": "arxivPa"}},
#         {"paperId": "pb", "title": "Tpb", "authors": [{"name": "B"}],
#          "year": 2023, "citationCount": 0, "url": "upb",
#          "externalIds": {}},
#     ]
#     rec._filter_papers()
#     rec._create_content()
#     assert "Recommendations based on multiple papers" in rec.content
#     assert "Number of recommended papers found: 2" in rec.content
#
# def test_single_helper_content():
#     rec = SinglePaperRecData(paper_id="pc", limit=1, year=None, tool_call_id="t6")
#     rec.recommendations = [
#         {"paperId": "pc", "title": "Tpc", "authors": [{"name": "C"}],
#          "year": 2024, "citationCount": 0, "url": "upc",
#          "externalIds": {"PubMed": "pmc"}},
#     ]
#     rec._filter_papers()
#     rec._create_content()
#     assert "Recommendations based on the single paper" in rec.content
#     assert "Query Paper ID: pc" in rec.content
