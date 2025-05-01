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
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit
from .biobridge_primekg import BioBridgePrimeKG

class BioBridgeDataModule(LightningDataModule):
    """
    `LightningDataModule` for the BioBridge dataset.
    """
    def __init__(self,
                 primekg_dir: str = "../../../../data/primekg/",
                 biobridge_dir: str = "../../../../data/biobridge_primekg/",
                 batch_size: int = 64) -> None:
        """
        Initializes the BioBridgeDataModule.

        Args:
            primekg_dir (str): Directory where the PrimeKG dataset is stored.
            biobridge_dir (str): Directory where the BioBridge dataset is stored.
            batch_size (int): Batch size for training and evaluation.
        """
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.primekg_dir = primekg_dir
        self.biobridge_dir = biobridge_dir
        self.batch_size = batch_size
        self.biobridge = None
        self.mapper = {}
        self.data = {}

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
        # self.mapper['et2etid'] = self.biobridge.get_data_config()["relation_type"]
        # self.mapper['etid2et'] = {v: k for k, v in self.mapper['et2etid'].items()}

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

        # Prepare BioBridge-PrimeKG nodes
        nodes = self.biobridge.primekg.get_nodes().copy()
        nodes = nodes[nodes["node_index"].isin(
            np.unique(np.concatenate([triplets.head_index.unique(),
                                      triplets.tail_index.unique()])))].reset_index(drop=True)

        # Obtain the node type ids and edge type ids
        node_types = np.unique(nodes['node_type'].tolist())
        # edge_types = np.unique(triplets["display_relation"].tolist())

        # Prepare the HeteroData object
        self.data["init"] = HeteroData()

        # Add node attributes to the HeteroData object
        for nt in node_types:
            # Create a mapping for each node type
            self.mapper[nt] = {}
            self.mapper[nt]['to_nidx'] = nodes[
                nodes['node_type'] == nt
            ]['node_index'].reset_index(drop=True).to_dict()
            self.mapper[nt]['from_nidx'] = {v: k for k, v in self.mapper[nt]['to_nidx'].items()}

            # Add node attributes
            self.data["init"][nt].num_nodes = len(self.mapper[nt]['from_nidx'])
            emb_ = np.array(
                [self.biobridge.emb_dict[i] for i in list(self.mapper[nt]['from_nidx'].keys())]
            )
            self.data["init"][nt].x = torch.tensor(emb_, dtype=torch.float32)

        # Add edge attributes to the HeteroData object
        for ht, rt, tt in triplets[
            ["head_type", "display_relation", "tail_type"]
            ].drop_duplicates().reset_index(drop=True).values:
            t_ = triplets[
                (triplets['head_type'] == ht) &
                (triplets['display_relation'] == rt) &
                (triplets['tail_type'] == tt)
            ]
            src_ids = t_[t_['head_type'] == ht]["head_index"].map(
                self.mapper[ht]['from_nidx']
            ).values
            dst_ids = t_[t_['tail_type'] == tt]["tail_index"].map(
                self.mapper[tt]['from_nidx']
            ).values
            self.data["init"][(ht, rt, tt)].edge_index = torch.tensor([src_ids,
                                                                       dst_ids], dtype=torch.long)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup the datasets for training, validation, and testing.
        """
        # Split the data into train, validation, and test sets
        transform = RandomLinkSplit(num_val=0.1,
                                    num_test=0.2,
                                    is_undirected=False,
                                    add_negative_train_samples=True,
                                    neg_sampling_ratio=1.0,
                                    split_labels=True,
                                    edge_types=self.data["init"].edge_types)

        self.data["train"], self.data["val"], self.data["test"] = transform(self.data["init"])

    def train_dataloader(self) -> DataLoader[Any]:
        """
        Returns the training data loader.
        """
        return DataLoader(self.data["train"],
                          batch_size=self.batch_size,
                          num_workers=0,
                          shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """
        Returns the validation data loader.
        """
        return DataLoader(self.data["val"],
                          batch_size=self.batch_size,
                          num_workers=0,
                          shuffle=False)

    def test_dataloader(self) -> DataLoader[Any]:
        """
        Returns the test data loader.
        """
        return DataLoader(self.data["test"],
                          batch_size=self.batch_size,
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
