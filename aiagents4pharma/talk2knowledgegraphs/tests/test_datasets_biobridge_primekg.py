"""
Test cases for datasets/primekg_loader.py
"""

import os
import shutil
import pytest
from ..datasets.biobridge_primekg import BioBridgePrimeKG

# Remove the data folder for testing if it exists
PRIMEKG_LOCAL_DIR = "../data/primekg_test/"
LOCAL_DIR = "../data/biobridge_primekg_test/"
shutil.rmtree(LOCAL_DIR, ignore_errors=True)

@pytest.fixture(name="biobridge_primekg")
def biobridge_primekg_fixture():
    """
    Fixture for creating an instance of PrimeKG.
    """
    return BioBridgePrimeKG(primekg_dir=PRIMEKG_LOCAL_DIR,
                            local_dir=LOCAL_DIR)

def test_download_primekg(biobridge_primekg):
    """
    Test the loading method of the BioBridge-PrimeKG class by downloading data from repository.
    """
    # Load BioBridge-PrimeKG data
    biobridge_primekg.load_data()
    primekg_nodes = biobridge_primekg.get_primekg().get_nodes()
    primekg_edges = biobridge_primekg.get_primekg().get_edges()
    biobridge_data_config = biobridge_primekg.get_data_config()
    biobridge_emb_dict = biobridge_primekg.get_node_embeddings()
    biobridge_triplets = biobridge_primekg.get_primekg_triplets()
    biobridge_splits = biobridge_primekg.get_train_test_split()
    biobridge_node_info = biobridge_primekg.get_node_info_dict()

    # Check if the local directories exists
    assert os.path.exists(biobridge_primekg.primekg_dir)
    assert os.path.exists(biobridge_primekg.local_dir)
    # Check if downloaded and processed files exist
    # PrimeKG files
    files = ["nodes.tab", "primekg_nodes.tsv.gz",
             "edges.csv", "primekg_edges.tsv.gz"]
    for file in files:
        path = f"{biobridge_primekg.primekg_dir}/{file}"
        assert os.path.exists(path)
    # BioBridge data config
    assert os.path.exists(f"{biobridge_primekg.local_dir}/data_config.json")
    # BioBridge embeddings
    files = [
        "protein.pkl",
        "mf.pkl",
        "cc.pkl",
        "bp.pkl",
        "drug.pkl",
        "disease.pkl",
        "embedding_dict.pkl"
    ]
    for file in files:
        path = f"{biobridge_primekg.local_dir}/embeddings/{file}"
        assert os.path.exists(path)
    # BioBridge processed files
    files = [
        "protein.csv",
        "mf.csv",
        "cc.csv",
        "bp.csv",
        "drug.csv",
        "disease.csv",
        "triplet_full.tsv.gz",
        "triplet_full_altered.tsv.gz",
        "node_train.tsv.gz",
        "triplet_train.tsv.gz",
        "node_test.tsv.gz",
        "triplet_test.tsv.gz",
    ]
    for file in files:
        path = f"{biobridge_primekg.local_dir}/processed/{file}"
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
    # Check processed BioBridge data config
    assert biobridge_data_config is not None
    assert len(biobridge_data_config) > 0
    assert len(biobridge_data_config['node_type']) == 10
    assert len(biobridge_data_config['relation_type']) == 18
    assert len(biobridge_data_config['emb_dim']) == 6
    # Check processed BioBridge embeddings
    assert biobridge_emb_dict is not None
    assert len(biobridge_emb_dict) > 0
    assert len(biobridge_emb_dict) == 85466
    # Check processed BioBridge triplets
    assert biobridge_triplets is not None
    assert len(biobridge_triplets) > 0
    assert biobridge_triplets.shape[0] == 3904610
    assert list(biobridge_splits.keys()) == ['train', 'node_train', 'test', 'node_test']
    assert len(biobridge_splits['train']) == 3510930
    assert len(biobridge_splits['node_train']) == 76486
    assert len(biobridge_splits['test']) == 393680
    assert len(biobridge_splits['node_test']) == 8495
    # Check node info dictionary
    assert list(biobridge_node_info.keys()) == ['gene/protein',
                                                'molecular_function',
                                                'cellular_component',
                                                'biological_process',
                                                'drug',
                                                'disease']
    assert len(biobridge_node_info['gene/protein']) == 19162
    assert len(biobridge_node_info['molecular_function']) == 10966
    assert len(biobridge_node_info['cellular_component']) == 4013
    assert len(biobridge_node_info['biological_process']) == 27478
    assert len(biobridge_node_info['drug']) == 6948
    assert len(biobridge_node_info['disease']) == 44133


def test_load_existing_primekg(biobridge_primekg):
    """
    Test the loading method of the BioBridge-PrimeKG class by loading existing data in local.
    """
    # Load BioBridge-PrimeKG data
    biobridge_primekg.load_data()
    primekg_nodes = biobridge_primekg.get_primekg().get_nodes()
    primekg_edges = biobridge_primekg.get_primekg().get_edges()
    biobridge_data_config = biobridge_primekg.get_data_config()
    biobridge_emb_dict = biobridge_primekg.get_node_embeddings()
    biobridge_triplets = biobridge_primekg.get_primekg_triplets()
    biobridge_splits = biobridge_primekg.get_train_test_split()
    biobridge_node_info = biobridge_primekg.get_node_info_dict()

    # Check if the local directories exists
    assert os.path.exists(biobridge_primekg.primekg_dir)
    assert os.path.exists(biobridge_primekg.local_dir)
    # Check if downloaded and processed files exist
    # PrimeKG files
    files = ["nodes.tab", "primekg_nodes.tsv.gz",
             "edges.csv", "primekg_edges.tsv.gz"]
    for file in files:
        path = f"{biobridge_primekg.primekg_dir}/{file}"
        assert os.path.exists(path)
    # BioBridge data config
    assert os.path.exists(f"{biobridge_primekg.local_dir}/data_config.json")
    # BioBridge embeddings
    files = [
        "protein.pkl",
        "mf.pkl",
        "cc.pkl",
        "bp.pkl",
        "drug.pkl",
        "disease.pkl",
        "embedding_dict.pkl"
    ]
    for file in files:
        path = f"{biobridge_primekg.local_dir}/embeddings/{file}"
        assert os.path.exists(path)
    # BioBridge processed files
    files = [
        "protein.csv",
        "mf.csv",
        "cc.csv",
        "bp.csv",
        "drug.csv",
        "disease.csv",
        "triplet_full.tsv.gz",
        "triplet_full_altered.tsv.gz",
        "node_train.tsv.gz",
        "triplet_train.tsv.gz",
        "node_test.tsv.gz",
        "triplet_test.tsv.gz",
    ]
    for file in files:
        path = f"{biobridge_primekg.local_dir}/processed/{file}"
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
    # Check processed BioBridge data config
    assert biobridge_data_config is not None
    assert len(biobridge_data_config) > 0
    assert len(biobridge_data_config['node_type']) == 10
    assert len(biobridge_data_config['relation_type']) == 18
    assert len(biobridge_data_config['emb_dim']) == 6
    # Check processed BioBridge embeddings
    assert biobridge_emb_dict is not None
    assert len(biobridge_emb_dict) > 0
    assert len(biobridge_emb_dict) == 85466
    # Check processed BioBridge triplets
    assert biobridge_triplets is not None
    assert len(biobridge_triplets) > 0
    assert biobridge_triplets.shape[0] == 3904610
    assert list(biobridge_splits.keys()) == ['train', 'node_train', 'test', 'node_test']
    assert len(biobridge_splits['train']) == 3510930
    assert len(biobridge_splits['node_train']) == 76486
    assert len(biobridge_splits['test']) == 393680
    assert len(biobridge_splits['node_test']) == 8495
    # Check node info dictionary
    assert list(biobridge_node_info.keys()) == ['gene/protein',
                                                'molecular_function',
                                                'cellular_component',
                                                'biological_process',
                                                'drug',
                                                'disease']
    assert len(biobridge_node_info['gene/protein']) == 19162
    assert len(biobridge_node_info['molecular_function']) == 10966
    assert len(biobridge_node_info['cellular_component']) == 4013
    assert len(biobridge_node_info['biological_process']) == 27478
    assert len(biobridge_node_info['drug']) == 6948
    assert len(biobridge_node_info['disease']) == 44133

# def test_load_existing_primekg_with_negative_triplets(biobridge_primekg):
#     """
#     Test the loading method of the BioBridge-PrimeKG class by loading existing data in local.
#     In addition, it builds negative triplets for training data.
#     """
#     # Load BioBridge-PrimeKG data
#     # Using 1 negative sample per positive triplet
#     biobridge_primekg.load_data(build_neg_triplest=True, n_neg_samples=1)
#     biobridge_neg_triplets = biobridge_primekg.get_primekg_triplets_negative()

#     # Check if the local directories exists
#     assert os.path.exists(biobridge_primekg.primekg_dir)
#     assert os.path.exists(biobridge_primekg.local_dir)
#     # Check if downloaded and processed files exist
#     path = f"{biobridge_primekg.local_dir}/processed/triplet_train_negative.tsv.gz"
#     assert os.path.exists(path)
#     # Check processed BioBridge triplets
#     assert biobridge_neg_triplets is not None
#     assert len(biobridge_neg_triplets) > 0
#     assert biobridge_neg_triplets.shape[0] == 3510930
#     assert len(biobridge_neg_triplets.negative_tail_index[0]) == 1
