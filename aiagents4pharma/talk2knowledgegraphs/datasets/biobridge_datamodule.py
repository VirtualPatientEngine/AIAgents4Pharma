"""
Loads the BioBridge dataset and prepares it for training and evaluation
using LightningDataModule from PyTorch Lightning.
"""

import os
from typing import Any, Dict, Optional
from lightning import LightningDataModule
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from .biobridge_primekg import BioBridgePrimeKG

class BioBridgeDataModule(LightningDataModule):
    """
    `LightningDataModule` for the BioBridge dataset.
    """
    def __init__(self,
                 primekg_dir: str = "../../../../data/primekg/",
                 biobridge_dir: str = "../../../../data/biobridge_primekg/",
                 configs: Dict[str, Any] = None) -> None:
        """
        Initializes the BioBridgeDataModule.

        Args:
            primekg_dir (str): Directory where the PrimeKG dataset is stored.
            biobridge_dir (str): Directory where the BioBridge dataset is stored.
            configs (Dict[str, Any]): Configuration dictionary for the data module.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.primekg_dir = primekg_dir
        self.biobridge_dir = biobridge_dir
        self.biobridge = None
        self.mapper = {}
        self.data = {}
        self.triplets = None
        self.nodes = None
        self.configs = configs

    def prepare_data(self) -> None:
        """
        Prepare the data by downloading and processing it.
        """
        # Define biobridge primekg data by providing a local directory where the data is stored
        self.biobridge = BioBridgePrimeKG(primekg_dir=self.primekg_dir,
                                          local_dir=self.biobridge_dir)

        # Invoke a method to load the data
        self.biobridge.load_data()

        # Map node type id <> node type
        self.mapper['nt2ntid'] = self.biobridge.get_data_config()["node_type"]
        self.mapper['ntid2nt'] = {v: k for k, v in self.mapper['nt2ntid'].items()}

        # Map edge type id <> edge type
        self.mapper['et2etid'] = self.biobridge.get_data_config()["relation_type"]
        self.mapper['etid2et'] = {v: k for k, v in self.mapper['et2etid'].items()}

        # Map node type id <> embedding dimension
        self.mapper["ntid2dim"] = {self.mapper["nt2ntid"][k]: v
                                   for k, v in self.biobridge.get_data_config()["emb_dim"].items()}

        # Prepare BioBridge-PrimeKG triplets
        # Build the node index list
        node_index_list = []
        for node_type in self.biobridge.preselected_node_types:
            df_node = pd.read_csv(os.path.join(self.biobridge.local_dir,
                                               "processed", f"{node_type}.csv"))
            node_index_list.extend(df_node["node_index"].tolist())

        # Filter the PrimeKG dataset to take into account only the selected node types
        triplets = self.biobridge.primekg.get_edges().copy()
        triplets = triplets[
            triplets["head_index"].isin(node_index_list) &\
            triplets["tail_index"].isin(node_index_list)
        ]
        triplets = triplets.reset_index(drop=True)

        # Further filtering out some nodes in the embedding dictionary
        triplets = triplets[
            triplets["head_index"].isin(list(self.biobridge.emb_dict.keys())) &\
            triplets["tail_index"].isin(list(self.biobridge.emb_dict.keys()))
        ].reset_index(drop=True)

        # Removing columns that are not needed for the PyG Data
        triplets.drop(columns=['head_name', 'head_source', 'head_id',
                               'tail_name', 'tail_source', 'tail_id',
                               'relation'], inplace=True)

        # Perform mapping of node types
        triplets["head_type"] = triplets["head_type"].apply(lambda x: self.mapper["nt2ntid"][x])
        triplets["tail_type"] = triplets["tail_type"].apply(lambda x: self.mapper["nt2ntid"][x])

        # Perform mapping of relation types
        triplets["display_relation"] = triplets["display_relation"].apply(
            lambda x: self.mapper["et2etid"][x]
        )

        # Treat the triplet as a tuple of (head_type, relation_type, tail_type) for class
        triplets['triplet_class'] = triplets.apply(
            lambda x: (x['head_type'], x['display_relation'], x['tail_type']
        ), axis=1)

        # Prepare BioBridge-PrimeKG nodes
        nodes = self.biobridge.primekg.get_nodes().copy()
        nodes = nodes[nodes["node_index"].isin(
            np.unique(np.concatenate([triplets.head_index.unique(),
                                      triplets.tail_index.unique()]))
        )].reset_index(drop=True)
        nodes["node_type"] = nodes["node_type"].apply(lambda x: self.mapper["nt2ntid"][x])
        nodes.drop(columns=['node_name', 'node_source', 'node_id'], inplace=True)

        self.triplets = triplets
        self.nodes = nodes

    def _split(self):
        """
        Splits dataframe into train, validation, and test sets.
        Subsequently, it creates a PyG Data object for each set.
        """
        # Perform the first stratified shuffle split for training and (validation + testing)
        strat_split = StratifiedShuffleSplit(n_splits=1,
                                             test_size=(self.configs["val_ratio"] +
                                                        self.configs["test_ratio"]),
                                             random_state=self.configs["random_state"])
        train_idx, valtest_idx = next(strat_split.split(self.triplets,
                                                        self.triplets['triplet_class']))
        df_train = self.triplets.copy().iloc[train_idx].reset_index(drop=True)
        df_valtest = self.triplets.copy().iloc[valtest_idx].reset_index(drop=True)

        # Perform the second stratified shuffle split for validation and testing
        test_ratio_relative = self.configs["test_ratio"] / (self.configs["val_ratio"] +
                                                            self.configs["test_ratio"])
        strat_split = StratifiedShuffleSplit(n_splits=1,
                                             test_size=test_ratio_relative,
                                             random_state=self.configs["random_state"])
        val_idx, test_idx = next(strat_split.split(df_valtest, df_valtest['triplet_class']))
        df_val = df_valtest.copy().iloc[val_idx].reset_index(drop=True)
        df_test = df_valtest.copy().iloc[test_idx].reset_index(drop=True)

        # Remove the triplet class column from the dataframes
        df_train = df_train.drop(columns=['triplet_class'])
        df_val = df_val.drop(columns=['triplet_class'])
        df_test = df_test.drop(columns=['triplet_class'])

        # Construct the PyG Data objects for train, val, and test sets
        # Prepare the PyG Data object for each split
        dataset = []
        for df in [df_train, df_val, df_test]:
            data_ = Data()
            data_.edge_index = torch.tensor(
                self.triplets[['head_index', 'tail_index']].values.T, dtype=torch.long
            )
            data_.edge_type = torch.tensor(
                self.triplets['display_relation'].values, dtype=torch.long
            )
            data_.target_edge_index = torch.tensor(
                df[['head_index', 'tail_index']].values.T, dtype=torch.long
            )
            data_.target_edge_type = torch.tensor(
                df['display_relation'].values, dtype=torch.long
            )
            data_.num_nodes = max(self.triplets["head_index"].max(),
                                  self.triplets["tail_index"].max())+1
            dataset.append(data_)

        return dataset[0], dataset[1], dataset[2]

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the datasets for training, validation, and testing.
        """
        # Split the data into train, validation, and test sets
        self.data["train"], self.data["val"], self.data["test"] = self._split()

        # Initialized the overall dataset as PyG InMemoryDataset
        self.data["set"] = InMemoryDataset()
        self.data["set"].data, self.data["set"].slices = self.data["set"].collate(
            [self.data["train"], self.data["val"], self.data["test"]]
        )
        # self.data["set"].num_relations = len(np.unique(self.triplets['display_relation'].values))
        self.data["set"].num_relations = self.triplets.display_relation.max()+1

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Returns the training data loader.
        """
        return DataLoader(self.data["train"],
                          batch_size=self.configs["batch_size"],
                          num_workers=0,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """
        Returns the validation data loader.
        """
        return DataLoader(self.data["val"],
                          batch_size=self.configs["batch_size"],
                          num_workers=0,
                          shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """
        Returns the test data loader.
        """
        return DataLoader(self.data["test"],
                          batch_size=self.configs["batch_size"],
                          num_workers=0,
                          shuffle=False)

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        Args:
            stage: The stage being torn down.
                    Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
                    Defaults to ``None``.

        Returns:
            None
        """
        # pass

    def state_dict(self) -> Dict[Any, Any]:
        """
        Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Args:
            None

        Returns:
            A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Args:
            state_dict: The datamodule state returned by `self.state_dict()`.

        Returns:
            None
        """
        # pass
