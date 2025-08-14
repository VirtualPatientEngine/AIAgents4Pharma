#!/cloud-home/U1013680/.magellan/conda/envs/scvi_python_3_10/bin/python
# coding: utf-8
# 20250318 with Multi-GPU

"""
LangCell: A language model for single-cell RNA-seq data analysis
This module implements a zero-shot cell type classification approach
using pretrained language and cell embeddings.
"""

import os
import json
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_auc_score, 
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import scanpy as sc
from datasets import load_from_disk
from transformers import BertTokenizer, BertModel

# Import custom modules (assuming they exist in a utils directory)
from utils import BertModel as MedBertModel
from utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification


# Setup logging
def setup_logging(output_dir=None, level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log file
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger('langcell')
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            output_dir / f"langcell_{timestamp}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class Config:
    """Configuration class to store all hyperparameters and paths."""
    
    def __init__(self, config_path: Optional[str] = None, logger=None):
        """
        Initialize configuration with default values or from a config file.
        
        Args:
            config_path: Optional path to a JSON configuration file
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger('langcell')
        
        # Default configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 256
        self.embedding_dim = 256
        self.similarity_temperature = 0.05
        self.alpha = 0.5  # Weight for ensemble prediction
        self.max_length = 512
        
        # Model paths
        self.base_path = Path("/cloud-data/digitalrnd-projects-ireland/SC_LLM/Magellan/model/langcell")
        self.cell_bert_path = self.base_path / "ckpt/cell_bert"
        self.cell_proj_path = self.base_path / "ckpt/cell_proj.bin"
        self.text_bert_path = self.base_path / "ckpt/text_bert"
        self.text_proj_path = self.base_path / "ckpt/text_proj.bin"
        self.ctm_head_path = self.base_path / "ckpt/ctm_head.bin"
        self.text_pretrained_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        
        # Data paths
        self.dataset_path = self.base_path / "data_zeroshot/pbmc10k.dataset"
        self.type2text_path = self.base_path / "data_zeroshot/type2text.json"
        
        # Load configuration from file if provided
        if config_path is not None:
            self._load_config(config_path)
            
        self.logger.info(f"Configuration initialized with device: {self.device}")
        self.logger.info(f"Using alpha={self.alpha} for ensemble prediction")
        self.logger.info(f"Dataset path: {self.dataset_path}")
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from a JSON file."""
        self.logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Update attributes from config file
            for key, value in config_dict.items():
                if key.endswith('_path'):
                    setattr(self, key, Path(value))
                else:
                    setattr(self, key, value)
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        attrs = vars(self)
        return '\n'.join(f"{key}: {attrs[key]}" for key in attrs if key != 'logger')


class Pooler(nn.Module):
    """
    Pooler module for extracting embeddings from the first token of the sequence
    and projecting them to the required dimension.
    """
    
    def __init__(self, config, pretrained_proj_path: Union[str, Path], proj_dim: int, logger=None):
        """
        Initialize the pooler.
        
        Args:
            config: Model configuration
            pretrained_proj_path: Path to pretrained projection weights
            proj_dim: Projection dimension
            logger: Logger instance
        """
        super().__init__()
        self.logger = logger or logging.getLogger('langcell')
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        
        self.logger.info(f"Loading projection weights from {pretrained_proj_path}")
        try:
            self.proj.load_state_dict(torch.load(pretrained_proj_path, map_location='cpu'))
            self.logger.info(f"Projection weights loaded successfully: {config.hidden_size} -> {proj_dim}")
        except Exception as e:
            self.logger.error(f"Failed to load projection weights: {str(e)}")
            raise
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extract the first token embedding and project it.
        
        Args:
            hidden_states: Hidden states from the model
            
        Returns:
            Normalized projected embedding
        """
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


class LangCellModel:
    """
    LangCell model for zero-shot cell type classification using language model embeddings
    and cell embeddings.
    """
    
    def __init__(self, config: Config, logger=None):
        """
        Initialize the LangCell model components.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.device = config.device
        self.logger = logger or logging.getLogger('langcell')
        
        self.logger.info("Initializing LangCell model components")
        start_time = time.time()
        
        # Initialize tokenizer
        self.logger.info(f"Loading tokenizer from {config.text_pretrained_model}")
        try:
            self.tokenizer = BertTokenizer.from_pretrained(config.text_pretrained_model)
            self.tokenizer.add_special_tokens({'bos_token': '[DEC]'})
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
            self.tokenizer.enc_token_id = self.tokenizer.additional_special_tokens_ids[0]
            self.logger.info("Tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {str(e)}")
            raise
        
        # Initialize cell encoder
        self.logger.info(f"Loading cell encoder from {config.cell_bert_path}")
        try:
            self.cell_encoder = BertModel.from_pretrained(config.cell_bert_path)
            self.cell_encoder.pooler = Pooler(
                self.cell_encoder.config, 
                config.cell_proj_path, 
                config.embedding_dim,
                logger=self.logger
            )
            # Move to device
            self.cell_encoder.to(self.device)
            self.logger.info("Cell encoder loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load cell encoder: {str(e)}")
            raise
        
        # Initialize text encoder
        self.logger.info(f"Loading text encoder from {config.text_bert_path}")
        try:
            self.text_encoder = MedBertModel.from_pretrained(
                config.text_bert_path, 
                add_pooling_layer=True
            )
            self.text_encoder.pooler = Pooler(
                self.text_encoder.config, 
                config.text_proj_path, 
                config.embedding_dim,
                logger=self.logger
            )
            # Move to device
            self.text_encoder.to(self.device)
            self.logger.info("Text encoder loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load text encoder: {str(e)}")
            raise
        
        # Initialize CTM head
        self.logger.info(f"Loading CTM head from {config.ctm_head_path}")
        try:
            self.ctm_head = nn.Linear(self.text_encoder.config.hidden_size, 2)
            self.ctm_head.load_state_dict(torch.load(config.ctm_head_path, map_location='cpu'))
            self.ctm_head.to(self.device)
            self.logger.info("CTM head loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load CTM head: {str(e)}")
            raise
        
        # [Multi-GPU]: Wrap models with DataParallel if multiple GPUs are available
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            self.logger.info(f"Multiple GPUs detected: {gpu_count}. Using DataParallel.")
            self.cell_encoder = nn.DataParallel(self.cell_encoder)
            self.text_encoder = nn.DataParallel(self.text_encoder)
            self.ctm_head = nn.DataParallel(self.ctm_head)
        
        # Set models to evaluation mode
        self.cell_encoder.eval()
        self.text_encoder.eval()
        self.ctm_head.eval()
        
        self.logger.info(f"LangCell model initialized in {time.time() - start_time:.2f} seconds")
    
    def text_encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text into embedding space.
        
        Args:
            text: Text or list of texts to encode
            
        Returns:
            Text embeddings
        """
        self.logger.debug(
            f"Encoding text: {text[:100]}..." if isinstance(text, str) else f"Encoding {len(text)} text items"
        )
        
        text_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            # When using DataParallel, calls remain the same:
            text_embedding = self.text_encoder(**text_input).pooler_output
        
        self.logger.debug(f"Text encoding completed, shape: {text_embedding.shape}")
        return text_embedding
    
    def cell_encode(
        self, cell_input_ids: torch.Tensor, cell_atts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode cell data into embedding space.
        
        Args:
            cell_input_ids: Cell input IDs
            cell_atts: Cell attention mask
            
        Returns:
            Tuple of (hidden states, pooled embedding)
        """
        self.logger.debug(f"Encoding cell batch of shape {cell_input_ids.shape}")
        
        with torch.no_grad():
            cell_output = self.cell_encoder(
                cell_input_ids.to(self.device),
                cell_atts.to(self.device)
            )
            cell_last_hidden = cell_output.last_hidden_state
            cell_pooler = cell_output.pooler_output
        
        self.logger.debug(f"Cell encoding completed, hidden shape: {cell_last_hidden.shape}, pooler shape: {cell_pooler.shape}")
        return cell_last_hidden, cell_pooler
    
    def ctm(
        self, text: List[str], cell_emb: torch.Tensor, cell_atts: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-modal alignment between text and cell.
        
        Args:
            text: List of text descriptions
            cell_emb: Cell hidden states
            cell_atts: Cell attention mask
            
        Returns:
            CTM logits (probability of alignment)
        """
        self.logger.debug(f"Computing CTM for {len(text)} text items against cell batch")
        
        text_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            # For cross-modal alignment, pass cell embeddings as encoder_hidden_states
            output = self.text_encoder(
                **text_input,
                encoder_hidden_states=cell_emb.to(self.device),
                encoder_attention_mask=cell_atts.to(self.device),
                return_dict=True,
                mode='multimodal',
            )
            logits = self.ctm_head(output.last_hidden_state[:, 0, :])
            probs = F.softmax(logits, dim=-1)[..., 1]  # Take positive class probability
        
        self.logger.debug(f"CTM computation completed, output shape: {probs.shape}")
        return probs
    
    def batch_ctm(
        self, 
        texts: List[str], 
        cell_emb: torch.Tensor, 
        cell_atts: torch.Tensor, 
        batch_size: int = 8
    ) -> torch.Tensor:
        """
        Efficient batched CTM calculation across all text types.
        
        Args:
            texts: List of all text descriptions
            cell_emb: Cell hidden states
            cell_atts: Cell attention mask
            batch_size: Number of texts to process at once
            
        Returns:
            CTM logits matrix
        """
        self.logger.debug(f"Batch CTM: processing {len(texts)} texts against {cell_emb.size(0)} cells")
        start_time = time.time()
        
        num_cells = cell_emb.size(0)
        num_texts = len(texts)
        ctm_logits = torch.zeros(num_cells, num_texts, device=self.device)
        
        # Process texts in batches
        for i in range(0, num_texts, batch_size):
            batch_start_time = time.time()
            text_batch = texts[i:min(i+batch_size, num_texts)]
            text_idx = list(range(i, min(i+batch_size, num_texts)))
            
            self.logger.debug(f"Processing text batch {i//batch_size + 1}/{(num_texts-1)//batch_size + 1}, size: {len(text_batch)}")
            
            # For each text in the batch, create copies for all cells
            for j, (text, idx) in enumerate(zip(text_batch, text_idx)):
                text_list = [text] * num_cells
                ctm_logits[:, idx] = self.ctm(text_list, cell_emb, cell_atts)
            
            self.logger.debug(f"Text batch processed in {time.time() - batch_start_time:.2f} seconds")
        
        # Normalize to create a proper probability distribution
        ctm_logits = F.softmax(ctm_logits, dim=-1)
        
        self.logger.debug(f"Batch CTM completed in {time.time() - start_time:.2f} seconds, output shape: {ctm_logits.shape}")
        return ctm_logits


class ZeroShotCellTypeClassifier:
    """
    Zero-shot cell type classifier using LangCell model.
    """
    
    def __init__(self, config: Config, logger=None):
        """
        Initialize the classifier.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger('langcell')
        self.logger.info("Initializing ZeroShotCellTypeClassifier")
        
        try:
            self.model = LangCellModel(config, logger=self.logger)
            self.logger.info("LangCell model initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LangCell model: {str(e)}")
            raise
        
        # Load cell type descriptions
        self.logger.info(f"Loading cell type descriptions from {config.type2text_path}")
        try:
            with open(config.type2text_path, 'r') as f:
                self.type2text = json.load(f)
            self.logger.info(f"Loaded descriptions for {len(self.type2text)} cell types")
        except Exception as e:
            self.logger.error(f"Failed to load cell type descriptions: {str(e)}")
            raise
    
    def load_dataset(self) -> Tuple:
        """
        Load and prepare the dataset.
        
        Returns:
            Tuple of dataset and prepared data loader
        """
        self.logger.info(f"Loading dataset from {self.config.dataset_path}")
        start_time = time.time()
        
        try:
            dataset = load_from_disk(str(self.config.dataset_path), keep_in_memory=True)
            dataset = dataset.shuffle(seed=42)
            self.logger.info(f"Dataset loaded successfully, size: {len(dataset)}")
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            raise
        
        # Find label column name
        label_column = None
        for candidate in ["celltype", "cell_type", "str_labels", "labels"]:
            if candidate in dataset.column_names:
                label_column = candidate
                break
        
        if label_column is None:
            error_msg = "Could not find label column in dataset"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"Found label column: '{label_column}'")
        
        if label_column != "celltype":
            dataset = dataset.rename_column(label_column, "celltype")
            self.logger.info(f"Renamed label column from '{label_column}' to 'celltype'")
        
        # Get unique cell types and their descriptions
        cell_types = list(set(dataset['celltype']))
        self.logger.info(f"Found {len(cell_types)} unique cell types in dataset")
        
        # Check if all cell types have descriptions
        missing_descriptions = []
        for cell_type in cell_types:
            if cell_type not in self.type2text:
                missing_descriptions.append(cell_type)
                
        if missing_descriptions:
            self.logger.warning(f"Missing descriptions for {len(missing_descriptions)} cell types: {missing_descriptions}")
            
        type_descriptions = [self.type2text.get(typename, f"A {typename} cell") for typename in cell_types]
        self.logger.info("Cell type descriptions prepared")
        
        # Create mapping from type to index
        type2idx = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
        
        # Encode cell type descriptions
        self.logger.info("Encoding cell type descriptions")
        encoding_start = time.time()
        try:
            with torch.no_grad():
                text_embeddings = torch.cat(
                    [self.model.text_encode(text) for text in type_descriptions],
                    0
                ).T.to(self.config.device)  # Shape: [embedding_dim, num_types]
            self.logger.info(f"Cell type descriptions encoded, shape: {text_embeddings.shape}")
        except Exception as e:
            self.logger.error(f"Failed to encode cell type descriptions: {str(e)}")
            raise
        
        self.logger.debug(f"Description encoding completed in {time.time() - encoding_start:.2f} seconds")
        
        # Prepare dataset for classification
        self.logger.info("Preparing dataset for classification")
        
        def classes_to_ids(example):
            example["label"] = type2idx[example["celltype"]]
            return example
        
        processed_dataset = dataset.map(classes_to_ids, num_proc=16)
        
        # Remove unnecessary columns
        remove_columns = processed_dataset.column_names.copy()
        remove_columns.remove('input_ids')
        remove_columns.remove('label')
        
        # Keep attention_mask if it exists
        if 'attention_mask' in remove_columns:
            remove_columns.remove('attention_mask')
            self.logger.info("Keeping 'attention_mask' column for processing")
        
        processed_dataset = processed_dataset.remove_columns(remove_columns)
        self.logger.info(f"Processed dataset prepared, retained columns: {processed_dataset.column_names}")
        
        # Create data loader
        collator = DataCollatorForCellClassification()
        data_loader = DataLoader(
            processed_dataset, 
            batch_size=self.config.batch_size, 
            collate_fn=collator, 
            shuffle=False
        )
        self.logger.info(f"Data loader created with batch size {self.config.batch_size}")
        
        self.logger.info(f"Dataset preparation completed in {time.time() - start_time:.2f} seconds")
        
        return (
            dataset, 
            processed_dataset, 
            data_loader, 
            cell_types, 
            type_descriptions, 
            text_embeddings, 
            type2idx
        )
    
    def classify(self) -> Dict:
        """
        Perform zero-shot cell type classification.
        
        Returns:
            Dictionary of results
        """
        # Load and prepare dataset
        self.logger.info("Starting zero-shot cell type classification")
        start_time = time.time()
        
        (
            dataset, 
            processed_dataset, 
            data_loader, 
            cell_types, 
            type_descriptions, 
            text_embeddings, 
            type2idx
        ) = self.load_dataset()
        
        dataset_size = len(dataset)
        num_types = len(cell_types)
        embedding_dim = self.config.embedding_dim
        device = self.config.device
        
        self.logger.info(f"Classification parameters: dataset_size={dataset_size}, num_types={num_types}, embedding_dim={embedding_dim}")
        
        # Initialize tensors for results
        cell_embeddings = torch.zeros(dataset_size, embedding_dim)
        sim_logits = torch.zeros(dataset_size, num_types)
        ctm_logits = torch.zeros(dataset_size, num_types)
        ensemble_logits = torch.zeros(dataset_size, num_types)
        predictions = torch.zeros(dataset_size)
        labels = torch.tensor(processed_dataset['label'])
        
        # Process batches
        self.logger.info("Processing batches for classification")
        classification_start = time.time()
        
        # Use a counter for logging interval to avoid excessive logs
        log_interval = max(1, len(data_loader) // 10)
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Classification", total=len(data_loader))):
                batch_start_time = time.time()
                start_idx = i * self.config.batch_size
                end_idx = min((i + 1) * self.config.batch_size, dataset_size)
                
                if i % log_interval == 0 or i == len(data_loader) - 1:
                    self.logger.info(f"Processing batch {i+1}/{len(data_loader)}, cells {start_idx}-{end_idx}")
                
                # Get cell embeddings
                cell_hidden, cell_embedding = self.model.cell_encode(
                    batch['input_ids'], 
                    batch.get('attention_mask', torch.ones_like(batch['input_ids']))
                )
                
                # Calculate similarity logits
                similarity_time = time.time()
                similarity = (cell_embedding @ text_embeddings) / self.config.similarity_temperature
                batch_sim_logits = F.softmax(similarity, dim=-1)
                
                if i % log_interval == 0:
                    self.logger.debug(f"Similarity calculation completed in {time.time() - similarity_time:.3f} seconds")
                
                # Calculate CTM logits
                ctm_time = time.time()
                batch_ctm_logits = self.model.batch_ctm(
                    type_descriptions,
                    cell_hidden,
                    batch.get('attention_mask', torch.ones_like(batch['input_ids']))
                )
                
                if i % log_interval == 0:
                    self.logger.debug(f"CTM calculation completed in {time.time() - ctm_time:.3f} seconds")
                
                # Combine logits with ensemble weight
                batch_ensemble_logits = (
                    self.config.alpha * batch_sim_logits + 
                    (1 - self.config.alpha) * batch_ctm_logits
                )
                batch_predictions = batch_ensemble_logits.argmax(dim=-1)
                
                # Store results
                cell_embeddings[start_idx:end_idx] = cell_embedding.cpu()
                sim_logits[start_idx:end_idx] = batch_sim_logits.cpu()
                ctm_logits[start_idx:end_idx] = batch_ctm_logits.cpu()
                ensemble_logits[start_idx:end_idx] = batch_ensemble_logits.cpu()
                predictions[start_idx:end_idx] = batch_predictions.cpu()
                
                if i % log_interval == 0:
                    self.logger.debug(f"Batch processing completed in {time.time() - batch_start_time:.3f} seconds")
        
        self.logger.info(f"Classification completed in {time.time() - classification_start:.2f} seconds")
        
        # Compute different prediction sets
        sim_predictions = sim_logits.argmax(dim=-1)
        ctm_predictions = ctm_logits.argmax(dim=-1)
        ensemble_predictions = ensemble_logits.argmax(dim=-1)
        
        self.logger.info("Results computed and prepared")
        
        # Check accuracy quickly
        accuracy = (ensemble_predictions == labels).float().mean().item()
        sim_accuracy = (sim_predictions == labels).float().mean().item()
        ctm_accuracy = (ctm_predictions == labels).float().mean().item()
        
        self.logger.info(f"Quick accuracy check: Similarity={sim_accuracy:.4f}, CTM={ctm_accuracy:.4f}, Ensemble={accuracy:.4f}")
        
        # Return all results
        results = {
            'cell_embeddings': cell_embeddings,
            'sim_logits': sim_logits, 
            'ctm_logits': ctm_logits, 
            'ensemble_logits': ensemble_logits,
            'sim_predictions': sim_predictions,
            'ctm_predictions': ctm_predictions,
            'ensemble_predictions': ensemble_predictions,
            'labels': labels,
            'cell_types': cell_types,
            'dataset': dataset
        }
        
        self.logger.info(f"Zero-shot classification completed in {time.time() - start_time:.2f} seconds")
        return results
    
    def evaluate(self, results: Dict) -> Dict:
        """
        Evaluate classification results.
        
        Args:
            results: Results from classification
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating classification results")
        start_time = time.time()
        
        labels = results['labels']
        ensemble_predictions = results['ensemble_predictions']
        cell_types = results['cell_types']
        
        # Compute confusion matrix
        self.logger.info("Computing confusion matrix")
        cm = confusion_matrix(labels, ensemble_predictions)
        
        # Compute classification report
        self.logger.info("Computing classification report")
        report = classification_report(
            labels, 
            ensemble_predictions, 
            target_names=cell_types, 
            digits=4, 
            output_dict=True
        )
        
        # Log macro metrics
        macro_precision = report['macro avg']['precision']
        macro_recall = report['macro avg']['recall']
        macro_f1 = report['macro avg']['f1-score']
        
        self.logger.info(f"Macro metrics - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
        
        # Compute normalized confusion matrix for visualization
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Compute per-class metrics
        self.logger.info("Computing per-class metrics")
        per_class_metrics = {}
        class_f1_scores = []
        
        for i, cell_type in enumerate(cell_types):
            class_indices = (labels == i)
            if sum(class_indices) > 0:
                true_positives = ((ensemble_predictions == i) & (labels == i)).sum().item()
                precision = true_positives / max(1, (ensemble_predictions == i).sum().item())
                recall = true_positives / max(1, (labels == i).sum().item())
                f1 = 2 * precision * recall / max(1e-8, precision + recall)
                
                per_class_metrics[cell_type] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': class_indices.sum().item()
                }
                class_f1_scores.append(f1)
                
                if f1 < 0.5:  # Log low-performing classes
                    self.logger.warning(f"Low performance for '{cell_type}' - F1: {f1:.4f}, Support: {class_indices.sum().item()}")
        
        # Log classes with best and worst performance
        sorted_f1_pairs = sorted([(ct, m['f1']) for ct, m in per_class_metrics.items()], 
                                 key=lambda x: x[1])
        worst_classes = sorted_f1_pairs[:3]
        best_classes = sorted_f1_pairs[-3:]
        
        self.logger.info(f"Best performing classes: {', '.join([f'{c}: {f:.4f}' for c, f in best_classes])}")
        self.logger.info(f"Worst performing classes: {', '.join([f'{c}: {f:.4f}' for c, f in worst_classes])}")
        
        # Compute overall metrics
        accuracy = (ensemble_predictions == labels).float().mean().item()
        self.logger.info(f"Overall accuracy: {accuracy:.4f}")
        
        # For multi-class ROC AUC, prepare one-hot encoded arrays
        self.logger.info("Computing ROC AUC scores")
        try:
            y_true_onehot = F.one_hot(labels.long(), num_classes=len(cell_types)).numpy()
            y_pred_probs = results['ensemble_logits'].numpy()
            
            macro_roc_auc = roc_auc_score(y_true_onehot, y_pred_probs, average='macro', multi_class='ovr')
            weighted_roc_auc = roc_auc_score(y_true_onehot, y_pred_probs, average='weighted', multi_class='ovr')
            
            self.logger.info(f"ROC AUC scores: Macro={macro_roc_auc:.4f}, Weighted={weighted_roc_auc:.4f}")
        except ValueError as e:
            # Handle case where some classes might not have both positive and negative samples
            self.logger.warning(f"Could not compute ROC AUC scores: {str(e)}")
            macro_roc_auc = np.nan
            weighted_roc_auc = np.nan
        
        evaluation = {
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'classification_report': report,
            'per_class_metrics': per_class_metrics,
            'accuracy': accuracy,
            'macro_roc_auc': macro_roc_auc,
            'weighted_roc_auc': weighted_roc_auc
        }
        
        self.logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        return evaluation
    
    def visualize(self, results: Dict, evaluation: Dict, output_dir: Path) -> None:
        """
        Visualize classification results and evaluation metrics.
        
        Args:
            results: Results from classification
            evaluation: Evaluation metrics
            output_dir: Directory to save plots
        """
        self.logger.info("Starting visualization of results")
        viz_start_time = time.time()
        
        # Create AnnData object for visualization
        self.logger.info("Creating AnnData object for visualization")
        cell_embeddings = results['cell_embeddings'].numpy()
        cell_types = results['cell_types']
        dataset = results['dataset']
        predictions = results['ensemble_predictions']
        
        try:
            adata = ad.AnnData(cell_embeddings)
            adata.obs['celltype'] = dataset['celltype']
            adata.obs['predictions'] = [cell_types[i] for i in predictions]
            
            # Add batch information if available
            if 'batch' in dataset.features.keys():
                self.logger.info("Adding batch information to visualization")
                adata.obs['batch'] = dataset['batch']
                adata.obs['batch'] = adata.obs['batch'].astype(str)
            
            # Add confidence scores
            logits = results['ensemble_logits'].numpy()
            prediction_confidence = logits.max(axis=1)
            adata.obs['confidence'] = prediction_confidence
            
            # Add prediction correctness
            adata.obs['correct_prediction'] = (
                results['ensemble_predictions'] == results['labels']
            ).numpy().astype(bool)
            
            # Compute UMAP embedding
            self.logger.info("Computing UMAP embedding for visualization")
            umap_start = time.time()
            sc.pp.neighbors(adata, use_rep='X', n_neighbors=30)
            sc.tl.umap(adata)
            self.logger.info(f"UMAP computation completed in {time.time() - umap_start:.2f} seconds")
            
            # Plot UMAP visualizations
            self.logger.info("Generating UMAP visualizations")
            plt.figure(figsize=(30, 30))
            sc.pl.umap(
                adata, 
                color=['celltype', 'predictions', 'correct_prediction', 'confidence'],
                title=['True Cell Types', 'Predicted Cell Types', 'Prediction Correctness', 'Confidence'],
                legend_loc='on data',
                legend_fontsize='x-small',
                size=30,
                wspace=0.5,
                ncols=1,
                show=False
            )
            
            self.logger.info("Saving UMAP visualization plot")
            plt.savefig(output_dir / 'umap_summary.png')
            self.logger.info("UMAP visualization saved to 'umap_summary.png'")
            
            # Plot confusion matrix
            self.logger.info("Generating confusion matrix visualization")
            plt.figure(figsize=(15, 15))
            cm_normalized = evaluation['confusion_matrix_normalized']
            df_cm = pd.DataFrame(
                cm_normalized, 
                index=cell_types, 
                columns=cell_types
            )
            sns.heatmap(df_cm, annot=False, cmap=plt.cm.Blues)
            plt.title('Normalized Confusion Matrix')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'langcell_results.png', dpi=300, bbox_inches='tight')
            self.logger.info("Confusion matrix visualization saved to 'langcell_results.png'")
        
            # Plot additional metrics
            self.logger.info("Generating metrics visualization")
            plt.figure(figsize=(15, 10))
            
            # Plot per-class F1 scores
            plt.subplot(2, 2, 1)
            per_class_metrics = evaluation['per_class_metrics']
            f1_scores = [metrics['f1'] for metrics in per_class_metrics.values()]
            cell_type_names = list(per_class_metrics.keys())
            
            # Sort by F1 score for better visualization
            sorted_indices = np.argsort(f1_scores)
            sorted_cell_types = [cell_type_names[i] for i in sorted_indices]
            sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
            
            plt.barh(sorted_cell_types, sorted_f1_scores)
            plt.xlabel('F1 Score')
            plt.title('F1 Score by Cell Type')
            plt.xlim(0, 1)
            
            # Plot class distribution
            plt.subplot(2, 2, 2)
            class_support = [metrics['support'] for metrics in per_class_metrics.values()]
            
            # Sort by class support
            sorted_indices = np.argsort(class_support)
            sorted_cell_types = [cell_type_names[i] for i in sorted_indices]
            sorted_support = [class_support[i] for i in sorted_indices]
            
            plt.barh(sorted_cell_types, sorted_support)
            plt.xlabel('Number of Cells')
            plt.title('Cell Type Distribution')
            
            # Plot comparison of methods (similarity vs CTM)
            plt.subplot(2, 2, 3)
            sim_accuracy = (results['sim_predictions'] == results['labels']).float().mean().item()
            ctm_accuracy = (results['ctm_predictions'] == results['labels']).float().mean().item()
            ensemble_accuracy = evaluation['accuracy']
            
            methods = ['Similarity', 'CTM', 'Ensemble']
            accuracies = [sim_accuracy, ctm_accuracy, ensemble_accuracy]
            
            plt.bar(methods, accuracies)
            plt.ylim(0, 1)
            plt.ylabel('Accuracy')
            plt.title('Comparison of Methods')
            
            # Plot misclassification analysis
            plt.subplot(2, 2, 4)
            incorrect_mask = (results['ensemble_predictions'] != results['labels']).numpy()
            if np.sum(incorrect_mask) > 0:
                incorrect_confidences = results['ensemble_logits'].numpy()[incorrect_mask].max(axis=1)
                plt.hist(incorrect_confidences, bins=20, alpha=0.5, label='Incorrect')
                
                correct_mask = ~incorrect_mask
                correct_confidences = results['ensemble_logits'].numpy()[correct_mask].max(axis=1)
                plt.hist(correct_confidences, bins=20, alpha=0.5, label='Correct')
                
                plt.xlabel('Confidence')
                plt.ylabel('Count')
                plt.title('Confidence Distribution')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'No misclassifications', ha='center', va='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'langcell_metrics.png', dpi=300, bbox_inches='tight')
            self.logger.info("Metrics visualization saved to 'langcell_metrics.png'")
            
            # Generate error analysis report
            self._generate_error_analysis(results, cell_types)
        except Exception as e:
            self.logger.error(f"Error during visualization: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
        self.logger.info(f"Visualization completed in {time.time() - viz_start_time:.2f} seconds")
    
    def _generate_error_analysis(self, results: Dict, cell_types: List[str]) -> None:
        """
        Generate detailed error analysis for misclassified cells.
        
        Args:
            results: Results from classification
            cell_types: List of cell type names
        """
        self.logger.info("Generating error analysis")
        start_time = time.time()
        
        labels = results['labels']
        predictions = results['ensemble_predictions']
        
        # Identify misclassified cells
        misclassified = (predictions != labels).numpy()
        misclassified_count = np.sum(misclassified)
        
        self.logger.info(f"Found {misclassified_count} misclassified cells ({misclassified_count/len(labels)*100:.2f}%)")
        
        if not np.any(misclassified):
            self.logger.info("No misclassifications found!")
            return
        
        # Get indices of misclassified cells
        misclassified_indices = np.where(misclassified)[0]
        
        # Analyze common error patterns
        error_pairs = []
        for idx in misclassified_indices:
            true_label = labels[idx].item()
            pred_label = predictions[idx].item()
            error_pairs.append((true_label, pred_label))
        
        # Count occurrences of each error pair
        error_counts = {}
        for true_label, pred_label in error_pairs:
            key = (true_label, pred_label)
            if key not in error_counts:
                error_counts[key] = 0
            error_counts[key] += 1
        
        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        self.logger.info("Top misclassification patterns:")
        self.logger.info("-------------------------------")
        for i, ((true_label, pred_label), count) in enumerate(sorted_errors[:10]):  # Show top 10
            true_type = cell_types[true_label]
            pred_type = cell_types[pred_label]
            error_pct = count / len(labels) * 100
            self.logger.info(f"{i+1}. {true_type} → {pred_type}: {count} cells ({error_pct:.2f}% of dataset)")
        
        # Analyze confidence of misclassifications
        misclassified_confidences = results['ensemble_logits'][misclassified].max(dim=1)[0].numpy()
        avg_misclassified_confidence = misclassified_confidences.mean()
        
        correct_confidences = results['ensemble_logits'][~misclassified].max(dim=1)[0].numpy()
        avg_correct_confidence = correct_confidences.mean()
        
        self.logger.info("\nConfidence analysis:")
        self.logger.info("--------------------")
        self.logger.info(f"Average confidence for correct classifications: {avg_correct_confidence:.4f}")
        self.logger.info(f"Average confidence for misclassifications: {avg_misclassified_confidence:.4f}")
        self.logger.info(f"Confidence gap: {avg_correct_confidence - avg_misclassified_confidence:.4f}")
        
        # Find high-confidence misclassifications (potential systematic errors)
        high_conf_threshold = np.percentile(correct_confidences, 75)  # 75th percentile
        high_conf_errors = misclassified_confidences > high_conf_threshold
        high_conf_error_count = np.sum(high_conf_errors)
        
        self.logger.info(f"High-confidence errors (above {high_conf_threshold:.4f}): {high_conf_error_count}")
        
        if np.any(high_conf_errors):
            high_conf_indices = misclassified_indices[high_conf_errors]
            self.logger.info("\nHigh-confidence misclassifications (potential systematic errors):")
            self.logger.info("---------------------------------------------------------------")
            
            for i, idx in enumerate(high_conf_indices[:5]):  # Show top 5
                true_label = labels[idx].item()
                pred_label = predictions[idx].item()
                confidence = results['ensemble_logits'][idx, pred_label].item()
                
                true_type = cell_types[true_label]
                pred_type = cell_types[pred_label]
                
                self.logger.info(f"Example {i+1}: {true_type} misclassified as {pred_type} with {confidence:.4f} confidence")
                
                # Compare similarity and CTM scores for this example
                sim_score = results['sim_logits'][idx, pred_label].item()
                ctm_score = results['ctm_logits'][idx, pred_label].item()
                
                self.logger.info(f"  Similarity score: {sim_score:.4f}, CTM score: {ctm_score:.4f}")
        
        # Analyze cell types with high error rates
        self.logger.info("\nCell types with highest error rates:")
        self.logger.info("------------------------------------")
        
        type_error_rates = {}
        for cell_type_idx, cell_type in enumerate(cell_types):
            type_indices = (labels == cell_type_idx)
            type_count = type_indices.sum().item()
            
            if type_count > 0:
                type_errors = ((predictions != labels) & type_indices).sum().item()
                error_rate = type_errors / type_count
                type_error_rates[cell_type] = (error_rate, type_count)
        
        # Sort by error rate
        sorted_error_rates = sorted(type_error_rates.items(), key=lambda x: x[1][0], reverse=True)
        
        for i, (cell_type, (error_rate, count)) in enumerate(sorted_error_rates[:5]):  # Show top 5
            self.logger.info(f"{i+1}. {cell_type}: {error_rate*100:.2f}% error rate ({count} cells)")
        
        self.logger.info(f"Error analysis completed in {time.time() - start_time:.2f} seconds")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='LangCell Zero-Shot Cell Type Classification')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--alpha', type=float, default=0.5, help='Ensemble weight (0=CTM only, 1=Similarity only)')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--save_results', action='store_true', default=True, help='Save classification results')
    parser.add_argument('--log_level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    return parser.parse_args()


def main():
    """Main function for running the LangCell zero-shot classification."""
    # Parse arguments
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(output_dir, level=log_level)
    
    # Log system information
    logger.info("=" * 80)
    logger.info("Starting LangCell Zero-Shot Classification")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Create configuration
        logger.info("Initializing configuration")
        config = Config(args.config, logger=logger)
        
        # Override config with command-line arguments
        if args.dataset:
            config.dataset_path = Path(args.dataset)
            logger.info(f"Overriding dataset path: {config.dataset_path}")
            
        if args.batch_size is not None:
            config.batch_size = args.batch_size
            logger.info(f"Overriding batch size: {config.batch_size}")
        
        if args.alpha is not None:
            config.alpha = args.alpha
            logger.info(f"Overriding ensemble alpha: {config.alpha}")
        
        # Initialize classifier
        logger.info("Initializing LangCell classifier")
        classifier = ZeroShotCellTypeClassifier(config, logger=logger)
        
        # Perform classification
        logger.info("Starting zero-shot classification")
        results = classifier.classify()
        
        # Evaluate results
        logger.info("Evaluating classification results")
        evaluation = classifier.evaluate(results)
        
        # Print summary
        logger.info("\nClassification Summary:")
        logger.info(f"Overall accuracy: {evaluation['accuracy']:.4f}")
        logger.info(f"Macro ROC AUC: {evaluation['macro_roc_auc']:.4f}")
        
        # Visualize results
        logger.info("Generating visualizations")
        classifier.visualize(results, evaluation, output_dir)
        
        # Save results if requested
        if args.save_results:
            logger.info(f"Saving results to {output_dir}")
            
            # Save raw results
            torch.save({
                'cell_embeddings': results['cell_embeddings'],
                'sim_logits': results['sim_logits'],
                'ctm_logits': results['ctm_logits'],
                'ensemble_logits': results['ensemble_logits'],
                'predictions': results['ensemble_predictions'],
                'labels': results['labels']
            }, output_dir / 'classification_results.pt')
            logger.info(f"Raw results saved to {output_dir / 'classification_results.pt'}")
            
            # Save evaluation metrics
            with open(output_dir / 'evaluation.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_eval = {}
                for key, value in evaluation.items():
                    if isinstance(value, np.ndarray):
                        serializable_eval[key] = value.tolist()
                    elif isinstance(value, dict):
                        # Handle nested dictionaries with numpy values
                        serializable_eval[key] = {}
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, np.ndarray):
                                serializable_eval[key][sub_key] = sub_value.tolist()
                            else:
                                serializable_eval[key][sub_key] = sub_value
                    else:
                        serializable_eval[key] = value
                
                json.dump(serializable_eval, f, indent=2)
            logger.info(f"Evaluation metrics saved to {output_dir / 'evaluation.json'}")
            
        total_time = time.time() - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info("LangCell zero-shot classification completed successfully")
    
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
