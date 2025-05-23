"""
Helper class for PDF Q&A tool orchestration: state validation, vectorstore init,
paper loading, reranking, and answer formatting.
"""

import logging
from typing import Any, Dict, List, Optional

from .generate_answer import generate_answer
from .nvidia_nim_reranker import rank_papers_by_query
from .vector_store import Vectorstore

logger = logging.getLogger(__name__)


class QAToolHelper:
    """Encapsulates helper routines for the PDF Question & Answer tool."""

    def __init__(self) -> None:
        self.prebuilt_vector_store: Optional[Vectorstore] = None
        self.config: Any = None
        self.call_id: str = ""
        logger.debug("Initialized QAToolHelper")

    def start_call(self, config: Any, call_id: str) -> None:
        """Initialize helper with current config and call identifier."""
        self.config = config
        self.call_id = call_id
        logger.debug("QAToolHelper started call %s", call_id)

    def get_state_models_and_data(self, state: dict) -> tuple[Any, Any, Dict[str, Any]]:
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

    def init_vector_store(self, emb_model: Any) -> Vectorstore:
        """Return shared or new Vectorstore instance."""
        if self.prebuilt_vector_store is not None:
            logger.info("Using shared pre-built vector store from memory")
            return self.prebuilt_vector_store
        vs = Vectorstore(embedding_model=emb_model, config=self.config)
        logger.info("Initialized new vector store with provided configuration")
        self.prebuilt_vector_store = vs
        return vs

    def load_candidate_papers(
        self,
        vs: Vectorstore,
        articles: Dict[str, Any],
        candidates: List[str],
    ) -> None:
        """Ensure each candidate paper is loaded into the vector store."""
        for pid in candidates:
            if pid not in vs.loaded_papers:
                pdf_url = articles.get(pid, {}).get("pdf_url")
                if not pdf_url:
                    continue
                try:
                    vs.add_paper(pid, pdf_url, articles[pid])
                except (IOError, ValueError) as exc:
                    logger.warning(
                        "%s: Error loading paper %s: %s", self.call_id, pid, exc
                    )

    def run_reranker(
        self,
        vs: Vectorstore,
        query: str,
        candidates: List[str],
    ) -> List[str]:
        """Rank papers by relevance and return filtered paper IDs."""
        try:
            ranked = rank_papers_by_query(
                vs, query, self.config, top_k=self.config.top_k_papers
            )
            logger.info("%s: Papers after NVIDIA reranking: %s", self.call_id, ranked)
            return [pid for pid in ranked if pid in candidates]
        except (ValueError, RuntimeError) as exc:
            logger.error("%s: NVIDIA reranker failed: %s", self.call_id, exc)
            logger.info(
                "%s: Falling back to all %d candidate papers",
                self.call_id,
                len(candidates),
            )
            return candidates

    def format_answer(
        self,
        question: str,
        chunks: List[Any],
        llm: Any,
        articles: Dict[str, Any],
    ) -> str:
        """Generate the final answer text with source attributions."""
        result = generate_answer(question, chunks, llm, self.config)
        answer = result.get("output_text", "No answer generated.")
        titles: Dict[str, str] = {}
        for pid in result.get("papers_used", []):
            if pid in articles:
                titles[pid] = articles[pid].get("Title", "Unknown paper")
        if titles:
            srcs = "\n\nSources:\n" + "\n".join(f"- {t}" for t in titles.values())
        else:
            srcs = ""
        logger.info(
            "%s: Generated answer using %d chunks from %d papers",
            self.call_id,
            len(chunks),
            len(titles),
        )
        return f"{answer}{srcs}"
