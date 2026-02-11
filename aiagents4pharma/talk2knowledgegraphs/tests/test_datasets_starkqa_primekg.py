"""
Test cases for datasets/starkqa_primekg.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
import torch

from ..datasets import starkqa_primekg as starkqa_module
from ..datasets.starkqa_primekg import StarkQAPrimeKG


def _write_starkqa_files(base: Path) -> None:
    (base / "qa/prime/stark_qa").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": [1, 2], "question": ["q1", "q2"]})
    df.to_csv(base / "qa/prime/stark_qa/stark_qa.csv", index=False)
    (base / "qa/prime/split").mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test", "test-0.1"]:
        (base / f"qa/prime/split/{split}.index").write_text("1\n")
    (base / "skb/prime/processed").mkdir(parents=True, exist_ok=True)
    joblib.dump({"node": "info"}, base / "skb/prime/processed/node_info.pkl")
    (base / "skb/prime/processed.zip").write_text("zip")


def _write_embeddings(base: Path) -> None:
    emb_dir = base / "text-embedding-ada-002"
    query_emb_dir = emb_dir / "query"
    node_emb_dir = emb_dir / "doc"
    query_emb_dir.mkdir(parents=True, exist_ok=True)
    node_emb_dir.mkdir(parents=True, exist_ok=True)
    torch.save({0: torch.zeros((1, 2))}, query_emb_dir / "query_emb_dict.pt")
    torch.save({0: torch.zeros((1, 2))}, node_emb_dir / "candidate_emb_dict.pt")


@pytest.fixture(name="starkqa_primekg")
def starkqa_primekg_fixture(tmp_path, monkeypatch):
    """Create StarkQAPrimeKG with mocked download helpers."""
    local_dir = tmp_path / "starkqa"
    instance = StarkQAPrimeKG(local_dir=str(local_dir))

    def fake_list_repo_files(_repo_id, repo_type="dataset"):
        del repo_type
        return [
            "qa/prime/stark_qa/stark_qa.csv",
            "qa/prime/stark_qa/stark_qa_human_generated_eval.csv",
            "qa/prime/split/train.index",
            "qa/prime/split/val.index",
            "qa/prime/split/test.index",
            "qa/prime/split/test-0.1.index",
            "skb/prime/processed.zip",
        ]

    def fake_hf_hub_download(_repo_id, filename, repo_type="dataset", local_dir=None):
        del repo_type
        base = Path(local_dir)
        if filename.endswith("stark_qa.csv") or filename.endswith(
            "stark_qa_human_generated_eval.csv"
        ):
            (base / "qa/prime/stark_qa").mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id": [1, 2], "question": ["q1", "q2"]}).to_csv(
                base / filename, index=False
            )
        elif filename.endswith(".index"):
            (base / "qa/prime/split").mkdir(parents=True, exist_ok=True)
            (base / filename).write_text("1\n")
        else:
            (base / "skb/prime").mkdir(parents=True, exist_ok=True)
            (base / filename).write_text("zip")
        return str(base / filename)

    def fake_unpack_archive(src, dest):
        del src
        base = Path(dest)
        (base / "processed").mkdir(parents=True, exist_ok=True)
        joblib.dump({"node": "info"}, base / "processed/node_info.pkl")

    def fake_gdown_download(_url, output, quiet=False):
        del _url, quiet
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({0: torch.zeros((1, 2))}, output)
        return output

    monkeypatch.setattr(starkqa_module, "list_repo_files", fake_list_repo_files, raising=True)
    monkeypatch.setattr(starkqa_module, "hf_hub_download", fake_hf_hub_download, raising=True)
    monkeypatch.setattr(starkqa_module.shutil, "unpack_archive", fake_unpack_archive, raising=True)
    monkeypatch.setattr(starkqa_module.gdown, "download", fake_gdown_download, raising=True)

    return instance


def _assert_outputs(starkqa: StarkQAPrimeKG) -> None:
    df = starkqa.get_starkqa()
    node_info = starkqa.get_starkqa_node_info()
    split_idx = starkqa.get_starkqa_split_indicies()
    query_emb = starkqa.get_query_embeddings()
    node_emb = starkqa.get_node_embeddings()

    assert df is not None and df.shape[0] == 2
    assert node_info == {"node": "info"}
    assert set(split_idx.keys()) == {"train", "val", "test", "test-0.1"}
    assert isinstance(query_emb, dict)
    assert isinstance(node_emb, dict)


def test_download_starkqa_primekg(starkqa_primekg):
    """Test download path and loading of StarkQA PrimeKG data."""
    starkqa_primekg.load_data()
    _assert_outputs(starkqa_primekg)


def test_load_existing_starkqa_primekg(starkqa_primekg):
    """Test loading when files already exist in local directory."""
    _write_starkqa_files(Path(starkqa_primekg.local_dir))
    _write_embeddings(Path(starkqa_primekg.local_dir))

    starkqa_primekg.load_data()
    _assert_outputs(starkqa_primekg)
