"""
Test cases for Talk2Biomodels API helpers with mocked network calls.
"""

import pytest
import requests

from ..api.ols import fetch_from_ols, fetch_ols_labels, search_ols_labels
from ..api.uniprot import search_uniprot_labels

pytestmark = pytest.mark.unit_mock


class _Resp:
    """Minimal response stub for requests.get."""

    def __init__(self, json_data, ok=True, error=None):
        self._json = json_data
        self._error = error
        self.status_code = 200 if ok else 500

    def json(self):
        """Return JSON payload or raise error."""
        if self._error:
            raise self._error
        return self._json

    def raise_for_status(self):
        """Raise stored error if present."""
        if self._error:
            raise self._error


def test_search_uniprot_labels(monkeypatch):
    """
    Test search_uniprot_labels success and error paths.
    """

    def fake_get(url, **_kwargs):
        if "P61764" in url:
            return _Resp(
                {
                    "proteinDescription": {
                        "recommendedName": {"fullName": {"value": "Syntaxin-binding protein 1"}}
                    }
                }
            )
        return _Resp({}, ok=False, error=requests.exceptions.RequestException("boom"))

    monkeypatch.setattr("requests.get", fake_get)

    identifiers = ["P61764", "P0000Q"]
    results = search_uniprot_labels(identifiers)
    assert results["P61764"] == "Syntaxin-binding protein 1"
    assert results["P0000Q"].startswith("Error: boom")


def test_fetch_from_ols(monkeypatch):
    """
    Test fetch_from_ols success and failure paths.
    """

    def fake_get(_url, params, **_kwargs):
        if params["obo_id"] == "GO:0005886":
            payload = {"_embedded": {"terms": [{"label": "plasma membrane"}]}}
            return _Resp(payload)
        raise requests.exceptions.RequestException("network fail")

    monkeypatch.setattr("requests.get", fake_get)

    assert fetch_from_ols("GO:0005886") == "plasma membrane"
    assert fetch_from_ols("GO:999999").startswith("Error: network fail")


def test_ols_helpers(monkeypatch):
    """
    Test fetch_ols_labels and search_ols_labels utilities.
    """

    def fake_fetch(term):
        return {"GO:1": "label-1", "GO:2": "label-2"}.get(term, "Error: not found")

    monkeypatch.setattr("aiagents4pharma.talk2biomodels.api.ols.fetch_from_ols", fake_fetch)
    assert fetch_ols_labels(["GO:1", "GO:2"]) == {"GO:1": "label-1", "GO:2": "label-2"}

    data = [{"Id": "GO:1", "Database": "GO"}, {"Id": "CHEBI:2", "Database": "CHEBI"}]
    results = search_ols_labels(data)
    assert results == {"go": {"GO:1": "label-1"}, "chebi": {"CHEBI:2": "Error: not found"}}


def test_fetch_ols_labels_empty():
    """
    Ensure empty input returns empty mapping.
    """
    assert not fetch_ols_labels([])


def test_search_ols_labels_empty():
    """
    Ensure empty list input returns empty mapping.
    """
    assert not search_ols_labels([])


def test_resp_raise_for_status_error():
    """
    Cover _Resp.raise_for_status error path.
    """
    resp = _Resp({}, ok=False, error=requests.exceptions.RequestException("err"))
    with pytest.raises(requests.exceptions.RequestException):
        resp.raise_for_status()


def test_resp_json_error():
    """
    Cover _Resp.json error path.
    """
    resp = _Resp({}, ok=False, error=requests.exceptions.RequestException("err"))
    with pytest.raises(requests.exceptions.RequestException):
        resp.json()
