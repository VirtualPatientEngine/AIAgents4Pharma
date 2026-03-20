"""
Test cases for datasets/primekg.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ..datasets import primekg as primekg_module
from ..datasets.primekg import PrimeKG


def _write_nodes(path: Path) -> None:
    df = pd.DataFrame(
        {
            "node_index": [0, 1],
            "node_name": ["A", "B"],
            "node_source": ["src", "src"],
            "node_id": ["ID0", "ID1"],
            "node_type": ["gene", "disease"],
        }
    )
    df.to_csv(path, sep="\t", index=False)


def _write_edges(path: Path) -> None:
    df = pd.DataFrame(
        {
            "x_index": [0],
            "y_index": [1],
            "display_relation": ["rel"],
            "relation": ["rel"],
        }
    )
    df.to_csv(path, index=False)


@pytest.fixture(name="primekg")
def primekg_fixture(tmp_path, monkeypatch):
    """Fixture for creating a PrimeKG instance with mocked download."""
    local_dir = tmp_path / "primekg"
    primekg = PrimeKG(local_dir=str(local_dir))

    def fake_download(_remote_url: str, local_path: str):
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "nodes.tab":
            _write_nodes(path)
        else:
            _write_edges(path)

    monkeypatch.setattr(primekg, "_download_file", fake_download, raising=True)
    return primekg


def _assert_primekg_outputs(primekg: PrimeKG) -> None:
    nodes = primekg.get_nodes()
    edges = primekg.get_edges()
    assert nodes is not None
    assert edges is not None
    assert nodes.shape[0] == 2
    assert edges.shape[0] == 1


def test_download_primekg(primekg):
    """Test loading PrimeKG by creating files via mocked download."""
    primekg.load_data()
    _assert_primekg_outputs(primekg)

    files = [
        "nodes.tab",
        f"{primekg.name}_nodes.tsv.gz",
        "edges.csv",
        f"{primekg.name}_edges.tsv.gz",
    ]
    for file in files:
        assert (Path(primekg.local_dir) / file).exists()


def test_load_existing_primekg(primekg):
    """Test loading PrimeKG when cached files exist."""
    primekg.load_data()
    _assert_primekg_outputs(primekg)
    # second call should hit existing file path branches
    primekg.load_data()
    _assert_primekg_outputs(primekg)


def test_primekg_download_file(tmp_path, monkeypatch):
    """Exercise the PrimeKG download helper with a fake response."""
    primekg = PrimeKGPublic(local_dir=str(tmp_path))

    class FakeResponse:
        """Minimal response stub for download."""

        headers = {"content-length": "4"}

        def raise_for_status(self):
            """No-op raise_for_status stub."""
            return None

        def iter_content(self, _chunk_size):
            """Yield a single data chunk."""
            return iter([b"data"])

    class FakeTqdm:
        """Progress bar stub."""

        def __init__(self, total=None, unit=None, unit_scale=None):
            self.total = total
            self.unit = unit
            self.unit_scale = unit_scale
            self.seen = 0

        def update(self, size):
            """Track the progress update."""
            self.seen += size

        def close(self):
            """No-op close."""
            return None

    monkeypatch.setattr(primekg_module.requests, "get", lambda *_a, **_k: FakeResponse())
    monkeypatch.setattr(primekg_module, "tqdm", FakeTqdm, raising=True)

    target = tmp_path / "nodes.tab"
    primekg.download_file("https://example.com/nodes", str(target))

    assert target.read_bytes() == b"data"


class PrimeKGPublic(PrimeKG):
    """Public wrapper for protected helpers."""

    def download_file(self, remote_url: str, local_path: str) -> None:
        """Proxy to protected download helper."""
        return self._download_file(remote_url, local_path)
