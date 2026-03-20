"""
Test cases for datasets/biobridge_primekg.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ..datasets import biobridge_primekg as biobridge_module
from ..datasets.biobridge_primekg import BioBridgePrimeKG


class BioBridgePrimeKGPublic(BioBridgePrimeKG):
    """Public wrapper for protected helpers."""

    def download_file(self, remote_url: str, local_dir: str, local_filename: str) -> None:
        """Proxy to protected download helper."""
        return self._download_file(remote_url, local_dir, local_filename)

    def build_node_embeddings(self) -> dict:
        """Proxy to protected embedding builder."""
        return self._build_node_embeddings()


class FakePrimeKG:
    """PrimeKG stub with minimal nodes/edges."""

    def __init__(self, edges: pd.DataFrame, nodes: pd.DataFrame):
        self._edges = edges
        self._nodes = nodes

    def load_data(self):
        """No-op load for stub."""
        return None

    def get_edges(self):
        """Return fake edges."""
        return self._edges

    def get_nodes(self):
        """Return fake nodes."""
        return self._nodes


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(obj))


def _write_node_csv(path: Path, node_indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"node_index": node_indices}).to_csv(path, index=False)


def _write_embeddings(path: Path, node_indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "node_index": node_indices,
            "embedding": [[0.1, 0.2] for _ in node_indices],
        },
        path,
    )


@pytest.fixture(name="biobridge_primekg")
def biobridge_primekg_fixture(tmp_path, monkeypatch):
    """Create a BioBridgePrimeKG instance with mocked dependencies."""
    primekg_dir = tmp_path / "primekg"
    local_dir = tmp_path / "biobridge"

    edges = pd.DataFrame(
        {
            "head_index": [0],
            "tail_index": [1],
            "head_type": ["gene/protein"],
            "tail_type": ["disease"],
            "display_relation": ["rel"],
            "relation": ["rel"],
        }
    )
    nodes = pd.DataFrame(
        {
            "node_index": [0, 1],
            "node_name": ["A", "B"],
            "node_source": ["src", "src"],
            "node_id": ["ID0", "ID1"],
            "node_type": ["gene/protein", "disease"],
        }
    )

    def fake_primekg_ctor(*_args, **_kwargs):
        return FakePrimeKG(edges=edges, nodes=nodes)

    monkeypatch.setattr(biobridge_module, "PrimeKG", fake_primekg_ctor, raising=True)

    def fake_download(remote_url: str, local_dir: str, local_filename: str):
        del remote_url
        path = Path(local_dir) / local_filename
        if local_filename == "data_config.json":
            _write_json(
                path,
                {
                    "node_type": {
                        "gene/protein": "gene/protein",
                        "disease": "disease",
                    },
                    "relation_type": {"rel": "rel"},
                    "emb_dim": {
                        "protein": 2,
                        "mf": 2,
                        "cc": 2,
                        "bp": 2,
                        "drug": 2,
                        "disease": 2,
                    },
                },
            )
        elif local_filename.endswith(".pkl"):
            _write_embeddings(path, node_indices=[0, 1])
        elif local_filename.endswith(".csv"):
            _write_node_csv(path, node_indices=[0, 1])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    instance = BioBridgePrimeKGPublic(primekg_dir=str(primekg_dir), local_dir=str(local_dir))
    monkeypatch.setattr(instance, "_download_file", fake_download, raising=True)

    return instance


def _assert_outputs(biobridge: BioBridgePrimeKG) -> None:
    assert biobridge.get_primekg() is not None
    assert biobridge.get_data_config() is not None
    assert biobridge.get_node_embeddings() is not None
    assert biobridge.get_primekg_triplets() is not None
    splits = biobridge.get_train_test_split()
    assert set(splits.keys()) == {"train", "node_train", "test", "node_test"}
    assert biobridge.get_node_info_dict() is not None


def test_load_biobridge_primekg(biobridge_primekg):
    """Exercise BioBridgePrimeKG load_data and cached branches."""
    biobridge_primekg.load_data()
    _assert_outputs(biobridge_primekg)

    # second run should hit cached-branch paths
    biobridge_primekg.load_data()
    _assert_outputs(biobridge_primekg)


def test_fake_primekg_get_nodes():
    """Cover FakePrimeKG.get_nodes helper."""
    edges = pd.DataFrame({"head_index": [], "tail_index": []})
    nodes = pd.DataFrame({"node_index": [1], "node_name": ["A"]})
    fake = FakePrimeKG(edges=edges, nodes=nodes)

    assert fake.get_nodes().shape[0] == 1


def test_biobridge_download_file(tmp_path, monkeypatch):
    """Exercise BioBridgePrimeKG download helper."""
    instance = BioBridgePrimeKGPublic(
        primekg_dir=str(tmp_path / "primekg"),
        local_dir=str(tmp_path),
    )

    class FakeResponse:
        """Minimal response stub."""

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

    monkeypatch.setattr(biobridge_module.requests, "get", lambda *_a, **_k: FakeResponse())
    monkeypatch.setattr(biobridge_module, "tqdm", FakeTqdm, raising=True)

    instance.download_file("https://example.com/file", str(tmp_path / "downloads"), "file.txt")
    assert (tmp_path / "downloads" / "file.txt").read_bytes() == b"data"


def test_biobridge_download_file_cached(tmp_path):
    """Cover cached download path in BioBridgePrimeKG._download_file."""
    instance = BioBridgePrimeKGPublic(
        primekg_dir=str(tmp_path / "primekg"),
        local_dir=str(tmp_path),
    )
    target_dir = tmp_path / "downloads"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "file.txt").write_text("cached")

    instance.download_file("https://example.com/file", str(target_dir), "file.txt")
    assert (target_dir / "file.txt").read_text() == "cached"


def test_biobridge_load_embeddings_converts_numpy(tmp_path, monkeypatch):
    """Ensure numpy embeddings are converted to lists."""
    instance = BioBridgePrimeKGPublic(
        primekg_dir=str(tmp_path / "primekg"),
        local_dir=str(tmp_path),
    )
    instance.preselected_node_types = ["gene"]

    def fake_download(remote_url: str, local_dir: str, local_filename: str):
        del remote_url
        path = Path(local_dir) / local_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub")

    monkeypatch.setattr(instance, "_download_file", fake_download, raising=True)

    def fake_joblib_load(_path: str):
        return {"node_index": [0], "embedding": np.array([[1.0, 2.0]])}

    monkeypatch.setattr(biobridge_module.joblib, "load", fake_joblib_load, raising=True)

    embeddings = instance.build_node_embeddings()
    assert embeddings[0] == [1.0, 2.0]


def test_fake_download_fallback(biobridge_primekg):
    """Exercise the fallback branch in fake_download helper."""
    biobridge_primekg.download_file(
        "https://example.com/file",
        biobridge_primekg.local_dir,
        "misc.txt",
    )
