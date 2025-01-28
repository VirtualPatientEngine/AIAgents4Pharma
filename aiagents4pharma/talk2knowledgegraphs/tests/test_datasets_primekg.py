"""
Test cases for datasets/primekg_loader.py
"""

import os
import shutil
import pytest
from ..datasets.primekg import PrimeKG

# Remove the data folder for testing if it exists
LOCAL_DIR = "../data/primekg_test/"
shutil.rmtree(LOCAL_DIR, ignore_errors=True)

@pytest.fixture(name="primekg")
def primekg_fixture():
    """
    Fixture for creating an instance of PrimeKG.
    """
    return PrimeKG(local_dir=LOCAL_DIR)

def test_download_primekg(primekg):
    """
    Test the loading method of the PrimeKG class by downloading PrimeKG from server.
    """
    # Load PrimeKG data
    primekg.load_data()
    primekg_nodes = primekg.get_nodes()
    primekg_edges = primekg.get_edges()

    # Check if the local directory exists
    assert os.path.exists(primekg.local_dir)
    # Check if downloaded and processed files exist
    files = ["nodes.tab", f"{primekg.name}_nodes.tsv.gz",
             "edges.csv", f"{primekg.name}_edges.tsv.gz"]
    for file in files:
        path = f"{primekg.local_dir}/{file}"
        assert os.path.exists(path)
    # Check processed PrimeKG dataframes
    # Nodes
    assert primekg_nodes is not None
    assert len(primekg_nodes) > 0
    assert primekg_nodes.shape[0] == 129375
    # Edges
    assert primekg_edges is not None
    assert len(primekg_edges) > 0
    assert primekg_edges.shape[0] == 8100498

def test_load_existing_primekg(primekg):
    """
    Test the loading method of the PrimeKG class by loading existing PrimeKG in local.
    """
    # Load PrimeKG data
    primekg.load_data()
    primekg_nodes = primekg.get_nodes()
    primekg_edges = primekg.get_edges()

    # Check if the local directory exists
    assert os.path.exists(primekg.local_dir)
    # Check if downloaded and processed files exist
    files = ["nodes.tab", f"{primekg.name}_nodes.tsv.gz",
             "edges.csv", f"{primekg.name}_edges.tsv.gz"]
    for file in files:
        path = f"{primekg.local_dir}/{file}"
        assert os.path.exists(path)
    # Check processed PrimeKG dataframes
    # Nodes
    assert primekg_nodes is not None
    assert len(primekg_nodes) > 0
    assert primekg_nodes.shape[0] == 129375
    # Edges
    assert primekg_edges is not None
    assert len(primekg_edges) > 0
    assert primekg_edges.shape[0] == 8100498
