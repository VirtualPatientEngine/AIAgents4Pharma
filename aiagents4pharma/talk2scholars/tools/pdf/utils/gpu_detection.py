"""
GPU Detection Utility for Milvus Index Selection
Handle COSINE -> IP conversion for GPU indexes
"""

import logging
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def detect_nvidia_gpu(config=None) -> bool:
    """
    Detect if NVIDIA GPU is available and should be used.

    Args:
        config: Hydra config object that may contain force_cpu_mode flag

    Returns:
        bool: True if GPU should be used, False if CPU should be used
    """

    # Check for force CPU mode in config
    if config and hasattr(config, "gpu_detection"):
        force_cpu = getattr(config.gpu_detection, "force_cpu_mode", False)
        if force_cpu:
            logger.info(
                "Force CPU mode enabled in config - using CPU even though GPU may be available"
            )
            return False

    # Normal GPU detection logic
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_names = result.stdout.strip().split("\n")
            logger.info("Detected NVIDIA GPU(s): %s", gpu_names)
            logger.info("To force CPU mode, set 'force_cpu_mode: true' in config")
            return True

        logger.info("nvidia-smi command failed or no GPUs detected")
        return False

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.info("NVIDIA GPU detection failed: %s", e)
        return False


def get_optimal_index_config(
    has_gpu: bool, embedding_dim: int = 768, use_cosine: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Get optimal index and search parameters based on GPU availability.

    IMPORTANT: GPU indexes don't support COSINE distance. When using GPU with COSINE,
    vectors must be normalized and IP distance used instead.

    Args:
        has_gpu (bool): Whether NVIDIA GPU is available
        embedding_dim (int): Dimension of embeddings
        use_cosine (bool): Whether to use cosine similarity (will be converted to IP for GPU)

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: (index_params, search_params)
    """
    if has_gpu:
        logger.info("Configuring GPU_CAGRA index for NVIDIA GPU")

        # For GPU: COSINE is not supported, must use IP with normalized vectors
        if use_cosine:
            logger.warning(
                "GPU indexes don't support COSINE distance. "
                "Vectors will be normalized and IP distance will be used instead."
            )
            metric_type = "IP"  # Inner Product for normalized vectors = cosine similarity
        else:
            metric_type = "IP"  # Default to IP for GPU

        # GPU_CAGRA index parameters - optimized for performance
        index_params = {
            "index_type": "GPU_CAGRA",
            "metric_type": metric_type,
            "params": {
                "intermediate_graph_degree": 64,  # Higher for better recall
                "graph_degree": 32,  # Balanced performance/recall
                "build_algo": "IVF_PQ",  # Higher quality build
                "cache_dataset_on_device": "true",  # Cache for better recall
                "adapt_for_cpu": "false",  # Pure GPU mode
            },
        }

        # GPU_CAGRA search parameters
        search_params = {
            "metric_type": metric_type,
            "params": {
                "itopk_size": 128,  # Power of 2, good for intermediate results
                "search_width": 16,  # Balanced entry points
                "team_size": 16,  # Optimize for typical vector dimensions
            },
        }

    else:
        logger.info("Configuring CPU index (IVF_FLAT) - no NVIDIA GPU detected")

        # CPU supports COSINE directly
        metric_type = "COSINE" if use_cosine else "IP"

        # CPU IVF_FLAT index parameters
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": metric_type,
            "params": {
                "nlist": min(1024, max(64, embedding_dim // 8))  # Dynamic nlist based on dimension
            },
        }

        # CPU search parameters
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": 16},  # Slightly higher than original for better recall
        }

    return index_params, search_params


def log_index_configuration(
    index_params: dict[str, Any], search_params: dict[str, Any], use_cosine: bool = True
) -> None:
    """Log the selected index configuration for debugging."""
    index_type = index_params.get("index_type", "Unknown")
    metric_type = index_params.get("metric_type", "Unknown")

    logger.info("=== Milvus Index Configuration ===")
    logger.info("Index Type: %s", index_type)
    logger.info("Metric Type: %s", metric_type)

    if index_type == "GPU_CAGRA" and use_cosine and metric_type == "IP":
        logger.info("NOTE: Using IP with normalized vectors to simulate COSINE for GPU")

    logger.info("Index Params: %s", index_params.get("params", {}))
    logger.info("Search Params: %s", search_params.get("params", {}))
    logger.info("===================================")
