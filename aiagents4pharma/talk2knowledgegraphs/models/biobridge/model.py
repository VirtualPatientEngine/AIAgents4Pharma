"""
BioBridge model implementation.
"""

from typing import Optional
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from lightning import LightningModule
from transformers.utils import ModelOutput
from tqdm import tqdm
from .losses import InfoNCE


@dataclass
class BioBridgeOutput(ModelOutput):
    """
    Class for the output of BioBridge model.
    """
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[torch.FloatTensor] = None
    tail_embeddings: Optional[torch.FloatTensor] = None


class BioBridge(LightningModule):
    """
    BioBridge model implementation.
    """

    def __init__(
        self,
        n_node: int,  # number of node types
        n_relation: int,  # number of relation types
        proj_dim: dict,  # dimension of the projection layer for each node type
        *,
        hidden_dim: int = 768,  # dimension of the hidden layer
        n_layer: int = 6,  # the number of transformer layers
    ) -> None:
        """
        Initialize the BioBridge model adapted from:
        https://github.com/RyanWangZf/BioBridge/blob/main/src/model.py

        Args:
            n_node: The number of node types.
            n_relation: The number of relation types.
            proj_dim: The dimension of the projection layer for each node type.
            hidden_dim: The dimension of the hidden layer.
            n_layer: The number of transformer layers.
        """
        super().__init__()
        self.save_hyperparameters()

        # Define loss function
        self.paired_loss_fn = InfoNCE(negative_mode="paired")
        self.unpaired_loss_fn = InfoNCE(negative_mode="unpaired")

        # Define node type embedding matrix
        self.node_type_embed = nn.Sequential(
            nn.Embedding(n_node, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),
        )

        # Define relation type embedding matrix
        self.relation_type_embed = nn.Sequential(
            nn.Embedding(n_relation, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),
        )

        # Define projection layer for each node type
        self.proj_layer = nn.ModuleDict()
        for node_type, dim in proj_dim.items():
            self.proj_layer[str(node_type)] = nn.Linear(dim, hidden_dim, bias=False)

        # Define transformation layer based on transformers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=12,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
        )

    def forward(
        self,
        head_emb,
        head_type_ids,
        rel_type_ids,
        tail_type_ids,
        *,
        tail_emb=None,
        neg_tail_emb=None,
        return_loss=False,
        **kwargs,
    ):
        """Forward pass of the binding model.

        Args:
            head_emb (List[torch.Tensor]): the embedding of the head node
            head_type_ids (torch.Tensor): the type id of the head node
            rel_type_ids (torch.Tensor): the type id of the relation
            tail_type_ids (torch.Tensor): the type id of the tail node
            tail_emb (List[torch.Tensor]): the embedding of the tail node.
                Only used when compute loss.
            neg_tail_emb (List[torch.Tensor]): the embedding of the negative tail
                node that does not match the head tail.
                Only used when compute loss.
        """
        # encode type ids
        # Dimensionality: (batch_size, 1, hidden_dim)
        type_emb = {}
        type_emb["head"] = self.node_type_embed(head_type_ids).unsqueeze(1)
        type_emb["rel"] = self.relation_type_embed(rel_type_ids).unsqueeze(1)
        type_emb["tail"] = self.node_type_embed(tail_type_ids).unsqueeze(1)

        # project head embeddings
        input_embs = {}
        input_embs["head"] = self._groupby_and_project(head_emb, head_type_ids)
        input_embs["head"] = input_embs["head"].unsqueeze(1)

        # project tail embeddings
        input_embs["tail"] = (
            self._groupby_and_project(tail_emb, tail_type_ids)
            if tail_emb is not None
            else None
        )
        input_embs["neg_tail"] = (
            self._groupby_and_project(neg_tail_emb, tail_type_ids)
            if neg_tail_emb is not None
            else None
        )

        # transformer encoder
        concat_embs = torch.cat(
            [input_embs["head"], type_emb["head"], type_emb["rel"], type_emb["tail"]], dim=1
        )
        output_embs = self._encode(concat_embs)

        # compute loss
        loss = None
        if tail_emb is not None and return_loss:
            if neg_tail_emb is not None:
                # use negative paired InfoNCE loss
                loss = self.paired_loss_fn(
                    output_embs, input_embs["tail"], input_embs["neg_tail"]
                )
            else:
                # use plain InfoNCE loss
                loss = self.unpaired_loss_fn(output_embs, input_embs["tail"])

        return BioBridgeOutput(
            embeddings=output_embs,  # projected and transformed head embeddings
            loss=loss,
            tail_embeddings=input_embs["tail"]
            if tail_emb is not None
            else None,  # projected tail embeddings
        )

    def training_step(self, batch, batch_idx):
        """
        Function to perform a training step.

        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
        
        Returns:
            The loss value.
        """
        head_emb, head_type_ids, rel_type_ids, tail_type_ids, tail_emb, neg_tail_emb = (
            batch
        )
        output = self.forward(
            head_emb=head_emb,
            head_type_ids=head_type_ids,
            rel_type_ids=rel_type_ids,
            tail_type_ids=tail_type_ids,
            tail_emb=tail_emb,
            neg_tail_emb=neg_tail_emb,
            return_loss=True,
        )
        self.log("train_loss", output.loss)
        return output.loss

    def validation_step(self, batch, batch_idx):
        """
        Function to perform a validation step.
        
        Args:
            batch: The input batch.
            batch_idx: The index of the batch.
        
        Returns:
            The loss value.
        """
        head_emb, head_type_ids, rel_type_ids, tail_type_ids, tail_emb, neg_tail_emb = (
            batch
        )
        output = self.forward(
            head_emb=head_emb,
            head_type_ids=head_type_ids,
            rel_type_ids=rel_type_ids,
            tail_type_ids=tail_type_ids,
            tail_emb=tail_emb,
            neg_tail_emb=neg_tail_emb,
            return_loss=True,
        )
        self.log("val_loss", output.loss)
        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def projection(self, node_emb, node_type_id) -> torch.Tensor:
        """
        Project the raw embeddings to the standard embedding space.
        """
        return self.proj_layer[str(node_type_id)](node_emb)

    @torch.no_grad()
    def encode(
        self,
        head_emb,
        head_type_id,
        rel_type_id,
        tail_type_id,
        *,
        batch_size=None,
        return_projected_head=False,
    ):
        """
        Encode the input embeddings and return the output embeddings. 
        Only process a single type of triplets, e.g., protein - interacts with - biological process.

        Args:
            head_emb (torch.Tensor): the embedding of the head node
            head_type_id (int): the type id of the head node
            rel_type_id (int): the type id of the relation
            tail_type_id (int): the type id of the tail node
            batch_size (int): the batch size of the input embeddings. 
                If None, use the size of the input embeddings.
            return_projected_head (bool): whether to return the projected head embeddings 
                (before passing to the encoder) or not.
        """
        # encode type ids
        type_emb = {}
        # Dimensionality: (batch_size, 1, hidden_dim)
        type_emb["head"] = self.node_type_embed(
            torch.tensor([head_type_id]).to(head_emb.device)
        ).unsqueeze(1)
        # Dimensionality: (batch_size, 1, hidden_dim)
        type_emb["rel"] = self.relation_type_embed(
            torch.tensor([rel_type_id]).to(head_emb.device)
        ).unsqueeze(1)
        # Dimensionality: (batch_size, 1, hidden_dim)
        type_emb["tail"] = self.node_type_embed(
            torch.tensor([tail_type_id]).to(head_emb.device)
        ).unsqueeze(1)

        num_samples = head_emb.size(0)
        if batch_size is None:
            batch_size = num_samples

        outputs = []
        projected_inputs = []
        for i in tqdm(range(0, num_samples, batch_size), "encoding..."):
            # project head embeddings
            input_emb = {}
            input_emb["head"] = self.projection(
                head_emb[i : i + batch_size], head_type_id.item()
            )
            projected_inputs.append(input_emb["head"].cpu().detach().numpy())
            input_emb["head"] = input_emb["head"].unsqueeze(1)
            input_emb["head_type"] = type_emb["head"].repeat(len(input_emb["head"]), 1, 1)
            input_emb["rel_type"] = type_emb["rel"].repeat(len(input_emb["head"]), 1, 1)
            input_emb["tail_type"] = type_emb["tail"].repeat(len(input_emb["head"]), 1, 1)
            concat_embs = torch.cat(
                [
                    input_emb["head"],
                    input_emb["head_type"],
                    input_emb["rel_type"],
                    input_emb["tail_type"],
                ],
                dim=1,
            )
            output_embs = self._encode(concat_embs)
            outputs.append(output_embs.cpu().detach().numpy())

        outputs = np.concatenate(outputs, axis=0)
        projected_inputs = np.concatenate(projected_inputs, axis=0)
        if return_projected_head:
            return {"tail_emb": outputs, "head_emb": projected_inputs}

        return outputs

    def _encode(self, input_embs):
        output_embs = self.encoder(input_embs)
        output_embs = output_embs[:, 0, :]
        # try TransE, converge faster
        output_embs = output_embs + input_embs[:, 0, :]
        return output_embs

    def _groupby_and_project(self, head_emb, head_type_ids):
        """Groupby the batch sample index by head_type_ids and project the embeddings."""
        # groupby batch sample index by head_type_ids
        head_type_id_uniq = torch.unique(head_type_ids)
        sample_index_groupby_head_type = defaultdict(list)
        batch_indexes = torch.arange(head_type_ids.size(0)).to(head_type_ids.device)
        for i, head_type_id in enumerate(head_type_id_uniq):
            sample_index_groupby_head_type[head_type_id.item()] = batch_indexes[
                head_type_ids == head_type_id
            ].tolist()

        # forward for each head type
        head_input_embs, sample_indexes = [], []
        for head_type_id in head_type_id_uniq:
            subsample_index = sample_index_groupby_head_type[head_type_id.item()]
            head_emb_subsample = torch.cat([head_emb[i][None] for i in subsample_index])
            # projection
            head_emb_subsample = self.projection(
                head_emb_subsample, head_type_id.item()
            )
            head_input_embs.append(head_emb_subsample)
            sample_indexes.extend(subsample_index)

        # sort head_input_embs by sample index from 0 to batch_size
        head_input_embs = torch.cat(head_input_embs, dim=0)
        head_input_embs = head_input_embs[torch.argsort(torch.tensor(sample_indexes))]
        return head_input_embs
