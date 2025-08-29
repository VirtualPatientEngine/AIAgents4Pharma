"""
Helper class for question and answer tool in PDF processing.
"""

import logging
from typing import Any

from .get_vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


class QAToolHelper:
    """
    Encapsulates helper routines for the PDF Question & Answer tool.
    Enhanced with automatic GPU/CPU detection and optimization.
    """

    def __init__(self) -> None:
        self.config: Any = None
        self.call_id: str = ""
        self.has_gpu: bool = False  # Track GPU availability
        logger.debug("Initialized QAToolHelper")

    def start_call(self, config: Any, call_id: str) -> None:
        """Initialize helper with current config and call identifier."""
        self.config = config
        self.call_id = call_id
        logger.debug("QAToolHelper started call %s", call_id)

    def get_state_models_and_data(self, state: dict) -> tuple[Any, Any, dict[str, Any]]:
        """Retrieve embedding model, LLM, and article data from agent state."""
        text_emb = state.get("text_embedding_model")
        if not text_emb:
            msg = "No text embedding model found in state."
            logger.error("%s: %s", self.call_id, msg)
            raise ValueError(msg)
        llm = state.get("llm_model")
        if not llm:
            msg = "No LLM model found in state."
            logger.error("%s: %s", self.call_id, msg)
            raise ValueError(msg)
        articles = state.get("article_data", {})
        if not articles:
            msg = "No article_data found in state."
            logger.error("%s: %s", self.call_id, msg)
            raise ValueError(msg)
        return text_emb, llm, articles

    def init_vector_store(self, emb_model: Any) -> Any:
        """Get the singleton Milvus vector store instance with GPU/CPU optimization."""
        logger.info(
            "%s: Getting singleton vector store instance with hardware optimization",
            self.call_id,
        )
        vs = get_vectorstore(embedding_model=emb_model, config=self.config)

        # Track GPU availability from vector store
        self.has_gpu = getattr(vs, "has_gpu", False)
        hardware_type = "GPU-accelerated" if self.has_gpu else "CPU-only"

        logger.info(
            "%s: Vector store initialized (%s mode)",
            self.call_id,
            hardware_type,
        )

        # Log hardware-specific configuration
        if hasattr(vs, "index_params"):
            index_type = vs.index_params.get("index_type", "Unknown")
            logger.info(
                "%s: Using %s index type for %s processing",
                self.call_id,
                index_type,
                hardware_type,
            )

        return vs

    def get_hardware_stats(self) -> dict[str, Any]:
        """Get current hardware configuration stats for monitoring."""
        return {
            "gpu_available": self.has_gpu,
            "hardware_mode": "GPU-accelerated" if self.has_gpu else "CPU-only",
            "call_id": self.call_id,
        }
