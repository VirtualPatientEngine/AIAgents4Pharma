from typing import Any, Optional, List, Dict, Tuple, Union, NamedTuple

import torch
import numpy as np
from torch.utils.data import Dataset

class NodeDataset(Dataset):
    def __init__(self, nodes):
        self.nodes = nodes
    def __len__(self):
        return len(self.nodes)
    def __getitem__(self, idx):
        return self.nodes.iloc[idx]

class EncodeNodeOutput(NamedTuple):
    embeddings: Dict[int, torch.Tensor] # key is the node type index
    node_index: Dict[int, np.ndarray] # key is the node type index

class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]

class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.inputs))
        else:
            return iter((self.predictions))

    def __getitem__(self, idx):
        if idx < 0 or idx > 1:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.inputs