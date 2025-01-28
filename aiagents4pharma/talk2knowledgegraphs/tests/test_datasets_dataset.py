"""
Test cases for datasets/dataset.py
"""

from ..datasets.dataset import Dataset

class MockDataset(Dataset):
    """
    Mock dataset class for testing purposes.
    """
    def setup(self):
        pass

    def load_data(self):
        pass

def test_dataset_setup():
    """
    Test the setup method of the Dataset class.
    """
    dataset = MockDataset()
    assert dataset.setup() is None

def test_dataset_load_data():
    """
    Test the load_data method of the Dataset class.
    """
    dataset = MockDataset()
    assert dataset.load_data() is None
