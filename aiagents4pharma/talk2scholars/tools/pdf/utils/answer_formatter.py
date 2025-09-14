"""
Format the final answer text with source attributions and hardware info.
"""

import logging
from typing import Any

from .generate_answer import generate_answer

logger = logging.getLogger(__name__)


def format_answer(
    question: str,
    chunks: list[Any],
    llm: Any,
    articles: dict[str, Any],
    config: Any,
    **kwargs: Any,
) -> str:
    """
    Generate the final answer text with source attributions and hardware info.

    Expects `call_id` and `has_gpu` in kwargs.
    """
    result = generate_answer(question, chunks, llm, config)
    answer = result.get("output_text", "No answer generated.")

    # Get unique paper titles for source attribution
    titles: dict[str, str] = {}
    for pid in result.get("papers_used", []):
        if pid in articles:
            titles[pid] = articles[pid].get("Title", "Unknown paper")

    # Format sources
    if titles:
        srcs = "\n\nSources:\n" + "\n".join(f"- {t}" for t in titles.values())
    else:
        srcs = ""

    # Extract logging metadata
    call_id = kwargs.get("call_id", "<no-call-id>")
    has_gpu = kwargs.get("has_gpu", False)
    hardware_info = "GPU-accelerated" if has_gpu else "CPU-processed"

    # Log final statistics with hardware info
    logger.info(
        "%s: Generated answer using %d chunks from %d papers (%s)",
        call_id,
        len(chunks),
        len(titles),
        hardware_info,
    )

    # Add subtle hardware info to logs but not to user output
    logger.debug(
        "%s: Answer generation completed with %s processing",
        call_id,
        hardware_info,
    )

    return f"{answer}{srcs}"
