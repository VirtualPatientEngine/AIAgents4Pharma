"""
Test cases for datasets/starkqa_primekg_loader.py
"""

import os
import shutil
import pytest
from ..datasets.starkqa_primekg import StarkQAPrimeKG

# Remove the data folder for testing if it exists
LOCAL_DIR = "../data/starkqa_primekg_test/"
shutil.rmtree(LOCAL_DIR, ignore_errors=True)

@pytest.fixture(name="starkqa_primekg")
def starkqa_primekg_fixture():
    """
    Fixture for creating an instance of StarkQAPrimeKGData.
    """
    return StarkQAPrimeKG(local_dir=LOCAL_DIR)

def test_download_starkqa_primekg(starkqa_primekg):
    """
    Test the loading method of the StarkQAPrimeKGLoaderTool class by downloading files
    from HuggingFace Hub.
    """
    # Load StarkQA PrimeKG data
    starkqa_primekg.load_data()
    starkqa_df = starkqa_primekg.get_starkqa()
    primekg_node_info = starkqa_primekg.get_starkqa_node_info()
    split_idx = starkqa_primekg.get_starkqa_split_indicies()
    query_embeddings = starkqa_primekg.get_query_embeddings()
    node_embeddings = starkqa_primekg.get_node_embeddings()

    # Check if the local directory exists
    assert os.path.exists(starkqa_primekg.local_dir)
    # Check if downloaded files exist in the local directory
    files = ['qa/prime/split/test-0.1.index',
             'qa/prime/split/test.index',
             'qa/prime/split/train.index',
             'qa/prime/split/val.index',
             'qa/prime/stark_qa/stark_qa.csv',
             'qa/prime/stark_qa/stark_qa_human_generated_eval.csv',
             'skb/prime/processed.zip']
    for file in files:
        path = f"{starkqa_primekg.local_dir}/{file}"
        assert os.path.exists(path)
    # Check dataframe
    assert starkqa_df is not None
    assert len(starkqa_df) > 0
    assert starkqa_df.shape[0] == 11204
    # Check node information
    assert primekg_node_info is not None
    assert len(primekg_node_info) == 129375
    # Check split indices
    assert list(split_idx.keys()) == ['train', 'val', 'test', 'test-0.1']
    assert len(split_idx['train']) == 6162
    assert len(split_idx['val']) == 2241
    assert len(split_idx['test']) == 2801
    assert len(split_idx['test-0.1']) == 280
    # Check query embeddings
    assert query_embeddings is not None
    assert len(query_embeddings) == 11204
    assert query_embeddings[0].shape[1] == 1536
    # Check node embeddings
    assert node_embeddings is not None
    assert len(node_embeddings) == 129375
    assert node_embeddings[0].shape[1] == 1536

def test_load_existing_starkqa_primekg(starkqa_primekg):
    """

    Test the loading method of the StarkQAPrimeKGLoaderTool class by loading existing files
    in the local directory.
    """
    # Load StarkQA PrimeKG data
    starkqa_primekg.load_data()
    starkqa_df = starkqa_primekg.get_starkqa()
    primekg_node_info = starkqa_primekg.get_starkqa_node_info()
    split_idx = starkqa_primekg.get_starkqa_split_indicies()
    query_embeddings = starkqa_primekg.get_query_embeddings()
    node_embeddings = starkqa_primekg.get_node_embeddings()

    # Check if the local directory exists
    assert os.path.exists(starkqa_primekg.local_dir)
    # Check if downloaded and processed files exist
    files = ['qa/prime/split/test-0.1.index',
             'qa/prime/split/test.index',
             'qa/prime/split/train.index',
             'qa/prime/split/val.index',
             'qa/prime/stark_qa/stark_qa.csv',
             'qa/prime/stark_qa/stark_qa_human_generated_eval.csv',
             'skb/prime/processed.zip']
    for file in files:
        path = f"{starkqa_primekg.local_dir}/{file}"
        assert os.path.exists(path)
    # Check dataframe
    assert starkqa_df is not None
    assert len(starkqa_df) > 0
    assert starkqa_df.shape[0] == 11204
    # Check node information
    assert primekg_node_info is not None
    assert len(primekg_node_info) == 129375
    # Check split indices
    assert list(split_idx.keys()) == ['train', 'val', 'test', 'test-0.1']
    assert len(split_idx['train']) == 6162
    assert len(split_idx['val']) == 2241
    assert len(split_idx['test']) == 2801
    assert len(split_idx['test-0.1']) == 280
    # Check query embeddings
    assert query_embeddings is not None
    assert len(query_embeddings) == 11204
    assert query_embeddings[0].shape[1] == 1536
    # Check node embeddings
    assert node_embeddings is not None
    assert len(node_embeddings) == 129375
    assert node_embeddings[0].shape[1] == 1536
