"""
Test cases for utils/pubchem_utils.py

These tests mock PubChem HTTP calls to avoid network dependency and verify
parsing logic without modifying business logic.
"""

from types import SimpleNamespace

import pytest

from ..utils import pubchem_utils


def _resp(json_data):
    """Create a minimal response-like object exposing .json()."""
    return SimpleNamespace(json=lambda: json_data)


@pytest.fixture(autouse=True)
def mock_requests_get(monkeypatch):
    """Mock requests.get used by pubchem_utils to return deterministic JSON."""

    def _mock_get(url, _timeout=60, **_kwargs):
        # CAS RN lookup: ethyl carbonate 105-58-8 -> CID 7766
        if "substance/xref/RN/105-58-8/record/JSON" in url:
            return _resp(
                {"PC_Substances": [{"compound": [{"id": {"type": 1, "id": {"cid": 7766}}}]}]}
            )

        # DrugBank: DB00240 (Alclometasone) -> CID 5311000
        if (
            "substance/sourceid//drugbank/DB00240/JSON" in url
            or "substance/sourceid/drugbank/DB00240/JSON" in url
        ):
            return _resp(
                {"PC_Substances": [{"compound": [{"id": {"type": 1, "id": {"cid": 5311000}}}]}]}
            )

        # CTD: D002083 (Butylated Hydroxyanisole) -> CID 24667
        if (
            "substance/sourceid//comparative toxicogenomics database/D002083/JSON" in url
            or "substance/sourceid/comparative toxicogenomics database/D002083/JSON" in url
        ):
            return _resp(
                {"PC_Substances": [{"compound": [{"id": {"type": 1, "id": {"cid": 24667}}}]}]}
            )

        # CID description for 5311000
        if (
            "compound/cid//5311000/description/JSON" in url
            or "compound/cid/5311000/description/JSON" in url
        ):
            return _resp(
                {
                    "InformationList": {
                        "Information": [
                            {
                                "Description": (
                                    "Alclometasone is a prednisolone compound having an "
                                    "alpha-chloro substituent at the 9alpha-position."
                                ),
                            }
                        ]
                    }
                }
            )

        # Default empty response
        return _resp({})

    monkeypatch.setattr(pubchem_utils.requests, "get", _mock_get)
    return _mock_get


def test_cas_rn2pubchem_cid():
    """
    Test the casRN2pubchem_cid function.

    The CAS RN for ethyl carbonate is 105-58-8.
    The PubChem CID for ethyl carbonate is 7766.
    """
    casrn = "105-58-8"
    pubchem_cid = pubchem_utils.cas_rn2pubchem_cid(casrn)
    assert pubchem_cid == 7766


def test_external_id2pubchem_cid():
    """
    Test the external_id2pubchem_cid function.

    The DrugBank ID for Alclometasone is DB00240 -> CID 5311000.
    The CTD ID for Butylated Hydroxyanisole is D002083 -> CID 24667.
    """
    drugbank_id = "DB00240"
    pubchem_cid = pubchem_utils.external_id2pubchem_cid("drugbank", drugbank_id)
    assert pubchem_cid == 5311000

    ctd_id = "D002083"
    pubchem_cid = pubchem_utils.external_id2pubchem_cid(
        "comparative toxicogenomics database", ctd_id
    )
    assert pubchem_cid == 24667


def test_pubchem_cid_description():
    """
    Test the pubchem_cid_description function.

    The PubChem CID for Alclometasone is 5311000.
    The description starts with the expected prefix.
    """
    pubchem_cid = 5311000
    description = pubchem_utils.pubchem_cid_description(pubchem_cid)
    assert description.startswith(
        "Alclometasone is a prednisolone compound having an alpha-chloro substituent"
    )


def test_mock_fallback_path_coverage():
    """Exercise the default branch of the mocked requests.get for coverage."""
    resp = pubchem_utils.requests.get("unknown://unmatched/url", timeout=1)
    assert resp.json() == {}
