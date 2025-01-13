#!/usr/bin/env python3

"""
Abstract class for dataset.
"""

from abc import ABC, abstractmethod

class Dataset(ABC):
    """
    Abstract class for dataset.
    """
    @abstractmethod
    def setup(self):
        """
        A method to set up the dataset.
        """

    @abstractmethod
    def load_data(self):
        """
        A method to load the dataset and potentially preprocess it.
        """
