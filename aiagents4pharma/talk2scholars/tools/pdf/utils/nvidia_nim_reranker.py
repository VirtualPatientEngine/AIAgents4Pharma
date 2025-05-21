"""
PDF Question & Answer Tool
"""

from typing import Any, List, Tuple

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank


def rank_papers_by_query(
    self, query: str, config: Any, top_k: int = 40
) -> List[Tuple[str, float]]:
    """
    Rank papers by relevance to the query using NVIDIA's off-the-shelf re-ranker.

    This function aggregates all chunks per paper, ranks them using the NVIDIA model,
    and returns the top-k papers.

    Args:
        query (str): The query string.
        config (Any): Configuration containing reranker settings (model, api_key).
        top_k (int): Number of top papers to return.

    Returns:
        List of tuples (paper_id, dummy_score) sorted by relevance.
    """

    # Aggregate all document chunks for each paper
    paper_texts = {}
    for doc in self.documents.values():
        paper_id = doc.metadata["paper_id"]
        paper_texts.setdefault(paper_id, []).append(doc.page_content)

    aggregated_documents = []
    for paper_id, texts in paper_texts.items():
        aggregated_text = " ".join(texts)
        aggregated_documents.append(
            Document(page_content=aggregated_text, metadata={"paper_id": paper_id})
        )

    # Instantiate the NVIDIA re-ranker client using provided config
    reranker = NVIDIARerank(
        model=config.reranker.model,
        api_key=config.reranker.api_key,
    )

    # Get the ranked list of documents based on the query
    response = reranker.compress_documents(query=query, documents=aggregated_documents)

    ranked_papers = [doc.metadata["paper_id"] for doc in response[:top_k]]
    return ranked_papers
