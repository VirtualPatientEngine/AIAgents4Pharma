import pdb
from collections import defaultdict

import torch
import numpy as np

class TrainCollator:
    def __init__(self, embedding_dict, n_negative=5) -> None:
        self.embedding_dict = embedding_dict
        self.n_negative = n_negative
        
    def __call__(self, batch):
        embedding_dict = self.embedding_dict
        # randomly sample a negative tail_index
        outputs = defaultdict(list)
        for row in batch:
            if row["head_index"] not in embedding_dict:
                print("[WARN] node index {} not in embedding dict, skip it.".format(row["head_index"]))
                continue

            if row["tail_index"] not in embedding_dict:
                print("[WARN] node index {} not in embedding dict, skip it.".format(row["tail_index"]))
                continue

            head_emb = torch.tensor(embedding_dict[row["head_index"]])
            tail_emb = torch.tensor(embedding_dict[row["tail_index"]])
            neg_tail_emb = None

            if "negative_tail_index" in row:
                negative_tail_index = row["negative_tail_index"]
                if isinstance(negative_tail_index, str):
                    negative_tail_index = eval(negative_tail_index)

                assert self.n_negative <= len(negative_tail_index), \
                    f"the number of negative samples {self.n_negative} is larger than the number of negative tail_index {len(negative_tail_index)}, please use a smaller `n_negative`!"

                # randomly sample a set of negative tail_index
                negative_tail_index = [idx for idx in negative_tail_index if idx in embedding_dict]
                sub_neg_y_index = np.random.choice(negative_tail_index, self.n_negative, replace=False)
                neg_tail_emb = torch.stack([torch.tensor(embedding_dict[idx]) for idx in sub_neg_y_index])

            # append to the outputs
            outputs["head_emb"].append(head_emb)
            outputs["head_type_ids"].append(row["head_type"])
            outputs["rel_type_ids"].append(row["display_relation"])
            outputs["tail_type_ids"].append(row["tail_type"])
            outputs["tail_emb"].append(tail_emb)
            if neg_tail_emb is not None:
                outputs["neg_tail_emb"].append(neg_tail_emb)

        # stack the tensors
        outputs["head_type_ids"] = torch.tensor(outputs["head_type_ids"])
        outputs["rel_type_ids"] = torch.tensor(outputs["rel_type_ids"])
        outputs["tail_type_ids"] = torch.tensor(outputs["tail_type_ids"])
        return outputs


class ValCollator:
    """Evaluate the model on the prediction tasks with retrieval.
    """
    def __init__(self, embedding_dict) -> None:
        self.embedding_dict = embedding_dict
        self.is_triplet = True

    def __call__(self, batch):
        """aggregate the data to a batch for model inference.
        """
        if self.is_triplet:
            return self._collate_triplet(batch)
        else:
            return self._collate_node(batch)
    
    def set_mode(self, is_triplet):
        """Set the mode of the collator.
        """
        self.is_triplet = is_triplet

    def _collate_triplet(self, batch):
        """Collate the triplet data.
        """
        outputs = defaultdict(list)
        embedding_dict = self.embedding_dict
        for row in batch:
            tail_emb = [torch.tensor(embedding_dict[i]) for i in row["tail_index"]]
            num_tail = len(tail_emb)
            head_emb = [torch.tensor(embedding_dict[row["head_index"]]) for _ in range(num_tail)]
            outputs["head_index"].append(row["head_index"])
            outputs["tail_index"].append(row["tail_index"])
            outputs["head_emb"].append(head_emb)
            outputs["head_type_ids"].append(torch.tensor([row["head_type"]]*num_tail))
            outputs["rel_type_ids"].append(torch.tensor(row["display_relation"])) # list of tensor
            outputs["tail_type_ids"].append(torch.tensor(row["tail_type"])) # list of tensor
            outputs["tail_emb"].append(tail_emb) # list of list of tensor
        return outputs
    
    def _collate_node(self, batch):
        """Collate for the input nodes
        """
        outputs = defaultdict(list)
        embedding_dict = self.embedding_dict
        for row in batch:
            node_index = row["node_index"]
            node_emb = torch.tensor(embedding_dict[node_index])
            outputs["node_emb"].append(node_emb)
            outputs["node_type_id"].append(row["node_type"])
            outputs["node_index"].append(node_index)

        # #################
        # debug if there is any empty list in the outputs
        # for key, value in outputs.items():
        #     if has_length(value):
        #         if len(value) == 0:
        #             pdb.set_trace()
        #         for val in value:
        #             if has_length(val):
        #                 if len(val) == 0:
        #                     pdb.set_trace()
        # ################## 

        return outputs
    
def has_length(inputs):
    return getattr(inputs, "__len__", None) is not None