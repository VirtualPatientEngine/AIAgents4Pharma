"""Implementations of the transformation models.
"""
import pdb
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from transformers.utils import ModelOutput
from tqdm import tqdm

from .losses import InfoNCE

@dataclass
class BindingModelOutput(ModelOutput):
    """Class for the output of the binding model.
    """
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[torch.FloatTensor] = None
    tail_embeddings: Optional[torch.FloatTensor] = None


class BindingModel(nn.Module):
    """Build a binding transformation model.
    """
    def __init__(self,
        n_node: int, # number of node types
        n_relation: int, # number of relation types
        proj_dim: dict, # dimension of the projection layer for each node type
        hidden_dim: int = 768, # dimension of the hidden layer
        n_layer: int = 6, # the number of transformer layers
        ) -> None:
        super().__init__()

        # build loss function
        self.paired_loss_fn = InfoNCE(negative_mode="paired")
        self.unpaired_loss_fn = InfoNCE(negative_mode="unpaired")

        # build node type embedding matrix
        self.node_type_embed = nn.Sequential(
            nn.Embedding(n_node, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),
        )

        # build relation type embedding matrix
        self.relation_type_embed =  nn.Sequential(
            nn.Embedding(n_relation, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),
        )
        
        # build projection layer for each node type
        self.proj_layer = nn.ModuleDict()
        for node_type, dim in proj_dim.items():
            self.proj_layer[str(node_type)] = nn.Linear(dim, hidden_dim, bias=False)
        
        # build transformation layer based on transformers
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

    def forward(self, 
        head_emb,
        head_type_ids,
        rel_type_ids,
        tail_type_ids,
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
            neg_tail_emb (List[torch.Tensor]): the embedding of the negative tail node that does not match the head tail. 
                Only used when compute loss.
        """
        # encode type ids
        head_type_emb = self.node_type_embed(head_type_ids).unsqueeze(1) # (batch_size, 1, hidden_dim)
        rel_type_emb = self.relation_type_embed(rel_type_ids).unsqueeze(1) # (batch_size, 1, hidden_dim)
        tail_type_emb = self.node_type_embed(tail_type_ids).unsqueeze(1) # (batch_size, 1, hidden_dim)

        # project head embeddings
        head_input_embs = self._groupby_and_project(head_emb, head_type_ids)
        head_input_embs = head_input_embs.unsqueeze(1)

        # project tail embeddings
        tail_input_embs = self._groupby_and_project(tail_emb, tail_type_ids) if tail_emb is not None else None
        neg_tail_input_embs = self._groupby_and_project(neg_tail_emb, tail_type_ids) if neg_tail_emb is not None else None

        # ###################
        # Deprecated forloop version
        # project head and tail embeddings
        # head_input_embs = []
        # tail_input_embs = [] if tail_emb is not None else None
        # neg_tail_input_embs = [] if neg_tail_emb is not None else None
        # for i, emb in enumerate(head_emb):
        #     # project the embedding to the hidden dimension
        #     head_input_emb = self.projection(emb, head_type_ids[i].item())
        #     head_input_embs.append(head_input_emb)
        #     if tail_emb is not None:
        #         tail_input_embs.append(self.projection(tail_emb[i], tail_type_ids[i].item()))
        #     if neg_tail_emb is not None:
        #         neg_tail_input_embs.append(self.projection(neg_tail_emb[i], tail_type_ids[i].item()))
        
        # head_input_embs = torch.stack(head_input_embs, dim=0).unsqueeze(1)
        # if tail_emb is not None:
        #     tail_input_embs = torch.stack(tail_input_embs, dim=0)
        # if neg_tail_emb is not None:
        #     neg_tail_input_embs = torch.stack(neg_tail_input_embs, dim=0)
        # ###################

        # transformer encoder
        input_embs = torch.cat([head_input_embs, head_type_emb, rel_type_emb, tail_type_emb], dim=1)
        output_embs = self._encode(input_embs)

        # compute loss
        loss = None
        if tail_emb is not None and return_loss:
            if neg_tail_emb is not None:
                # use negative paired InfoNCE loss
                # input
                # query: output_embs: (batch_size, hidden_dim)
                # positive keys: tail_input_embs: (batch_size, hidden_dim)
                # negative keys: neg_tail_input_embs: (batch_size, negative_sample_size, hidden_dim)
                loss = self.paired_loss_fn(output_embs, tail_input_embs, neg_tail_input_embs)

            else:
                # use plain InfoNCE loss
                # input
                # query: output_embs: (batch_size, hidden_dim)
                # positive keys: tail_input_embs: (batch_size, hidden_dim)
                loss = self.unpaired_loss_fn(output_embs, tail_input_embs)

        return BindingModelOutput(
            embeddings=output_embs, # projected and transformed head embeddings
            loss=loss,
            tail_embeddings=tail_input_embs if tail_emb is not None else None, # projected tail embeddings
            )

    def projection(self, node_emb, node_type_id) -> torch.Tensor:
        """Project the raw embeddings to the standard embedding space.
        """
        return self.proj_layer[str(node_type_id)](node_emb)
    
    @torch.no_grad()
    def encode(self, 
        head_emb,
        head_type_id,
        rel_type_id,
        tail_type_id,
        batch_size=None,
        return_projected_head=False,
        ):
        """
        Encode the input embeddings and return the output embeddings. Only process a single type of triplets, e.g.,
        protein - interacts with - biological process.

        Args:
            head_emb (torch.Tensor): the embedding of the head node
            head_type_id (int): the type id of the head node
            rel_type_id (int): the type id of the relation
            tail_type_id (int): the type id of the tail node
            batch_size (int): the batch size of the input embeddings. If None, use the size of the input embeddings.
            return_projected_head (bool): whether to return the projected head embeddings (before passing to the encoder) or not.
        """
        # encode type ids
        tgt_device = head_emb.device
        head_type_id = torch.tensor([head_type_id]).to(tgt_device)
        head_type_emb = self.node_type_embed(head_type_id).unsqueeze(1) # (batch_size, 1, hidden_dim)
        rel_type_id = torch.tensor([rel_type_id]).to(tgt_device)
        rel_type_emb = self.relation_type_embed(rel_type_id).unsqueeze(1) # (batch_size, 1, hidden_dim)
        tail_type_id = torch.tensor([tail_type_id]).to(tgt_device)
        tail_type_emb = self.node_type_embed(tail_type_id).unsqueeze(1) # (batch_size, 1, hidden_dim)

        num_samples = head_emb.size(0)
        if batch_size is None:
            batch_size = num_samples

        outputs = []
        projected_inputs = []
        for i in tqdm(range(0, num_samples, batch_size), "encoding..."):
            # project head embeddings
            head_input_emb = self.projection(head_emb[i:i+batch_size], head_type_id.item())
            projected_inputs.append(head_input_emb.cpu().detach().numpy())
            head_input_emb = head_input_emb.unsqueeze(1)
            head_type_input_emb = head_type_emb.repeat(len(head_input_emb), 1, 1)
            rel_type_input_emb = rel_type_emb.repeat(len(head_input_emb), 1, 1)
            tail_type_input_emb = tail_type_emb.repeat(len(head_input_emb), 1, 1)
            input_embs = torch.cat([head_input_emb, head_type_input_emb, rel_type_input_emb, tail_type_input_emb], dim=1)
            output_embs = self._encode(input_embs)
            outputs.append(output_embs.cpu().detach().numpy())

        outputs = np.concatenate(outputs, axis=0)
        projected_inputs = np.concatenate(projected_inputs, axis=0)
        if return_projected_head:
            return {"tail_emb":outputs, "head_emb":projected_inputs}
        else:
            return outputs
    
    def _encode(self, input_embs):
        output_embs = self.encoder(input_embs)
        output_embs = output_embs[:, 0, :]
        # try TransE, converge faster
        output_embs = output_embs + input_embs[:, 0, :]
        return output_embs

    def _groupby_and_project(self, head_emb, head_type_ids):
        """Groupby the batch sample index by head_type_ids and project the embeddings.
        """
        # groupby batch sample index by head_type_ids
        head_type_id_uniq = torch.unique(head_type_ids)
        sample_index_groupby_head_type = defaultdict(list)
        batch_indexes = torch.arange(head_type_ids.size(0)).to(head_type_ids.device)
        for i, head_type_id in enumerate(head_type_id_uniq):
            sample_index_groupby_head_type[head_type_id.item()] = batch_indexes[head_type_ids == head_type_id].tolist()

        # forward for each head type
        head_input_embs, sample_indexes = [], []
        for head_type_id in head_type_id_uniq:
            subsample_index = sample_index_groupby_head_type[head_type_id.item()]
            head_emb_subsample = torch.cat([head_emb[i][None] for i in subsample_index])
            # projection
            head_emb_subsample = self.projection(head_emb_subsample, head_type_id.item())
            head_input_embs.append(head_emb_subsample)
            sample_indexes.extend(subsample_index)

        # sort head_input_embs by sample index from 0 to batch_size
        head_input_embs = torch.cat(head_input_embs, dim=0)
        head_input_embs = head_input_embs[torch.argsort(torch.tensor(sample_indexes))]
        return head_input_embs

def build_model_config(data_config):
    # build model config
    model_config = {
        "n_node": len(data_config["node_type"]),
        "n_relation": len(data_config["relation_type"]),
        }
    proj_dim = {}
    for node_type, dim in data_config["emb_dim"].items():
        proj_dim[data_config["node_type"][node_type]] = dim
    model_config["proj_dim"] = proj_dim
    return model_config