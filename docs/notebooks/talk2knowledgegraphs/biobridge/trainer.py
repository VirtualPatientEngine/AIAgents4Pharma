"""Adapted from transformers.Trainer with `transformers==4.29.2`
"""
import time
import pdb
import math
from collections import defaultdict
from typing import Any, Optional, List, Dict, Tuple, Union, NamedTuple

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
from transformers.integrations.deepspeed import (
    deepspeed_init, 
)
from transformers.utils import logging
from transformers.trainer_utils import (
    PredictionOutput,
    speed_metrics,
    has_length,
    denumpify_detensorize,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_numpify,
    nested_truncate,
    nested_concat,
)

from tqdm import tqdm

from .schema import (
    NodeDataset,
    EncodeNodeOutput,
    EvalLoopOutput,
    EvalPrediction
)

logger = logging.get_logger(__name__)

def cos_sim_fn(a, b):
    return F.normalize(a) @ F.normalize(b).t()

class BindingTrainer(Trainer):
    def __init__(self, test_data_collator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_data_collate_fn = test_data_collator
        self.encoded_nodes = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss of the model.
        """
        outputs = model(return_loss=True, **inputs)
        loss = outputs.get("loss")
        outputs = outputs.get("embeddings")
        return (loss, outputs) if return_outputs else loss
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        # set the test collate function
        data_collator = self.test_data_collate_fn if self.test_data_collate_fn is not None else self.data_collator
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
        
        return DataLoader(eval_dataset, **dataloader_params)

    def get_eval_node_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        nodes = eval_dataset.get_all_node()
        collate_fn = self.test_data_collate_fn
        
        # collate function for node
        collate_fn.set_mode(is_triplet=False)
        eval_dataset = NodeDataset(nodes)
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": collate_fn,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        dataloader = DataLoader(eval_dataset, **dataloader_params)
        return dataloader
    
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:
        """Evaluate for the downstream prediction tasks.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        # Step 1: need to project all node to their embedding space with the same dim first
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        node_loader = self.get_eval_node_dataloader(eval_dataset)
        self.encoded_nodes = self.encode_node_loop(node_loader)

        # Step 2: go through all the groundtruth triplet, get the transformed head embeddings, one node transformed to multiple embeddings related to the relation types
        # Step 3: retrieve the tail embeddings with the transformed head embeddings and relation types
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)
        
        # target tail node types for link predictions
        tail_node_types = eval_dataset.tail_node_types
        num_test_nodes = len(eval_dataset.node)

        if args.past_index >= 0:
            self._past = None

        # predictions on GPU
        preds_host = None

        # losses/preds/labels on CPU (final containers)
        all_preds = None

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in tqdm(enumerate(dataloader), desc="Prediction", total=num_test_nodes//batch_size):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            predictions = self.prediction_step(model, inputs)

            # if predictions's keys are not in preds_host, add padding index to avoid error in nested_concat
            for k in tail_node_types:
                for pred in predictions:
                    if k not in pred["prediction"]:
                        pred["prediction"][k] = torch.tensor([-100])[None]
                    if k not in pred['label']:
                        pred["label"][k] = torch.tensor([-100])[None]

            # concatenate the triplets
            preds_host = predictions if preds_host is None else preds_host + predictions

        # get all preds from preds_host
        all_preds = defaultdict(list)
        all_preds["prediction"] = defaultdict(list)
        all_preds["label"] = defaultdict(list)
        num_samples = len(preds_host)
        for pred in preds_host:
            all_preds['node_index'].append(pred['node_index'])
            all_preds['node_type'].append(pred['node_type'])
            for k in tail_node_types:
                all_preds['prediction'][k].append(pred['prediction'][k])
                all_preds['label'][k].append(pred['label'][k])
        
        # stack the tensors
        all_preds['node_index'] = torch.cat(all_preds['node_index'], dim=0)
        all_preds['node_type'] = torch.cat(all_preds['node_type'], dim=0)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds))
        else:
            metrics = {}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=predictions, metrics=metrics, num_samples=num_samples)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
    ) -> List[Dict]:
        """Perform an evaluation step on `model` using `inputs`.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        num_batch = len(inputs.get("head_emb", None))

        # get the pre-encoded and projected node embeddings
        encoded_nodes = self.encoded_nodes
        output_list = []
        
        with torch.no_grad():
            for i in range(num_batch):
                single_inputs = {k: v[i] for k, v in inputs.items()}
                single_inputs = self._prepare_inputs(single_inputs)
                head_node_index = single_inputs.pop("head_index")
                tail_node_index = torch.tensor(single_inputs.pop("tail_index"))
                try:
                    outputs = model(**single_inputs)
                except Exception as err:
                    raise ValueError("Error occurs in model forward Error: {}".format(err))
                predicted_tail_emb = outputs.get("embeddings", None)
                tail_node_index = tail_node_index.to(predicted_tail_emb.device)

                # predicted_tail_emb = predicted_tail_emb.cpu()

                # match the predicted tail embeddings with the target type embeddings
                # rank the tail node based on similarity and return the similarities
                """
                # outputs should be the following format
                {
                    "node_index": node_index,
                    "prediction": 
                        {
                        0: [simialrity_0, simialrity_1, ...],
                        1: [[simialrity_0, simialrity_1, ...],
                        ...}
                    "label": 
                    {0: [label_0, label_1, ...],
                    1: [label_0, label_1, ...],
                    ...
                    }
                }
                """
                tail_type_ids = single_inputs["tail_type_ids"]
                head_type_ids = single_inputs["head_type_ids"]
                outputs = {"node_index": torch.tensor([head_node_index]), 
                           "node_type": torch.tensor([head_type_ids[0].item()]),
                           "prediction": {},
                           "label": {},
                           }
                for tail_type_id in tail_type_ids.unique():
                    # get the query node embedding
                    tail_type_id = tail_type_id.item()
                    tail_type_index = (tail_type_ids == tail_type_id).nonzero(as_tuple=True)[0]
                    tail_node_index_label = tail_node_index[tail_type_index]
                    query_emb = predicted_tail_emb[tail_type_index][0][None] # head id, rel id, same tail type id, so should be the same across the x-axis

                    # retrieve the target node embeddings
                    candidate_node_embs = encoded_nodes.embeddings[tail_type_id]
                    candidate_node_embs = candidate_node_embs.to(query_emb.device)

                    cos_similarity = cos_sim_fn(query_emb, candidate_node_embs)
                    candidate_node_indexes = encoded_nodes.node_index[tail_type_id]
                    prediction = candidate_node_indexes[torch.argsort(cos_similarity[0].cpu(), descending=True).numpy()]
                    
                    # rank the target node embeddings based on similarity
                    outputs["prediction"][tail_type_id] = torch.tensor(prediction[None]).cpu()
                    outputs["label"][tail_type_id] = tail_node_index_label[None].cpu()

                output_list.append(outputs)
        
        return output_list


    def encode_node_loop(
        self,
        dataloader: DataLoader,
    ) -> EncodeNodeOutput:
        """Encode all the given nodes, make projection, and return the embeddings.
        """
        model = self.model
        model.eval()

        logger.info(f"***** Running Node Encoding for Evaluation *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")

        encoded_nodes = defaultdict(list)
        for step, inputs in enumerate(dataloader):
            num_batch = len(inputs.get("node_emb", None))
            with torch.no_grad():
                for i in range(num_batch):
                    single_inputs = {k: v[i] for k, v in inputs.items()}
                    single_inputs = self._prepare_inputs(single_inputs)
                    node_index = single_inputs.pop("node_index")
                    outputs = model.projection(**single_inputs)
                    outputs = outputs.cpu()
                    if len(outputs.shape) == 1:
                        outputs = outputs[None]
                    encoded_nodes["embeddings"].append(outputs)
                    encoded_nodes["node_index"].append(node_index)
                    encoded_nodes["node_type_id"].append(single_inputs["node_type_id"])

        encoded_nodes["embeddings"] = torch.cat(encoded_nodes["embeddings"], dim=0)
        encoded_nodes["node_index"] = np.array(encoded_nodes["node_index"])
        encoded_nodes["node_type_id"] = np.array(encoded_nodes["node_type_id"])

        # groupby node type
        outputs = {
            "embeddings": {},
            "node_index": {},
        }
        node_type_ids = encoded_nodes["node_type_id"]
        unique_node_type_ids = np.unique(node_type_ids)
        for node_type_id in unique_node_type_ids:
            node_type_index = np.where(node_type_ids == node_type_id)[0]
            outputs["embeddings"][node_type_id] = encoded_nodes["embeddings"][node_type_index]
            outputs["node_index"][node_type_id] = encoded_nodes["node_index"][node_type_index]
            
        # reset the collate function mode back to normal
        # otherwise it raises error when the dataloader is used again for training
        self.test_data_collate_fn.set_mode(is_triplet=True)
        return EncodeNodeOutput(**outputs)
