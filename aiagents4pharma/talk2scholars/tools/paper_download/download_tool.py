# tools/download_tool.py
"""
Tool to download paper from Arxiv,Biorxiv, Medrxiv
"""
import logging
from typing import Annotated, List
from pydantic import BaseModel, Field

from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools.base import InjectedToolCallId

from .download_arxiv_input import DownloadArxivPaperInput
from .download_biorxiv_input import DownloadBiorxivPaperInput
from .download_medrxiv_input import DownloadMedrxivPaperInput
from .download_pubmed_input import DownloadPubmedPaperInput
from .utils.summary_builder import build_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownloadSourceDetermination(BaseModel):
    """
    Structured output schema to determine the list of paper ids to be downloaded
    - arxiv_ids like ["arxiv_id:1212.0001"]
    - DOIs like ["10.1101/2025.06.22.660927"]
    - Pubmed ids like ["pmc: PMC123456"] or ["pmc: 123456"] 
    """
    arxiv_ids: List[str] = Field(
        ...,
        description=
        "The list of arxiv ids of the papers. For example: arxiv_id:1212.0001",
    )
    dois: List[str] = Field(
        ...,
        description=
        "The list of DOIs of the papers. For example: DOI:10.1101/2025.06.22.660927",
    )
    pubmed_ids: List[str] = Field(
        ...,
        description=
        "The list of pubmed ids of the papers. For example: pubmed: PMC123456",
    )

class DownloadPaperInput(BaseModel):
    """Input Schema to paper download tool"""
    paper_id: List[str] = Field(
        description="List of paper ids (e.g., arXiv ID, DOIs)")
    tool_call_id: Annotated[str, InjectedToolCallId]
    state: Annotated[dict, InjectedState]

@tool(args_schema=DownloadPaperInput)
def download_paper(
    paper_id: List[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
    ):
    """
    Download a paper given its ID and source (e.g., arXiv or Biorxiv).
    """
    logger.info("Inside paper download tool")
    llm_model = state.get("llm_model")
    structured_llm = llm_model.with_structured_output(DownloadSourceDetermination)
    collective_ids = structured_llm.invoke(
        [f"The paper ids are {paper_id}. Identify the arxiv ids and dois."]
        )
    arxiv_ids = collective_ids.arxiv_ids
    dois = collective_ids.dois
    pubmed_ids = collective_ids.pubmed_ids
    logger.info("arxiv ids: %s",arxiv_ids)
    logger.info("dois: %s",dois)
    logger.info("pubmed: %s",pubmed_ids)
    all_article_data = {}
    if arxiv_ids:
        logger.info("Getting into arxiv")
        retriever = DownloadArxivPaperInput()
        arxiv_cmd =  retriever.paper_retriever(paper_ids=arxiv_ids)
        if arxiv_cmd:
            all_article_data.update(arxiv_cmd["article_data"])
            logger.info("Recieved papers from arxiv")

    if pubmed_ids:
        logger.info("Getting into Pubmed")
        retriever = DownloadPubmedPaperInput()
        pubmed_cmd = retriever.paper_retriever(paper_ids=pubmed_ids)
        if pubmed_cmd:
            all_article_data.update(pubmed_cmd["article_data"])
            logger.info("Recieved papers from pubmed")

    if dois:
        try:
            logger.info("Getting into bioarxiv")
            retriever = DownloadBiorxivPaperInput()
            doi_cmd =  retriever.paper_retriever(paper_ids=dois)
            logger.info("Recieved papers from bioarxiv")
        except ValueError:
            logger.info("Getting into medrxiv")
            retriever = DownloadMedrxivPaperInput()
            doi_cmd = retriever.paper_retriever(paper_ids=dois)
            logger.info("Recieved papers from medarxiv")

        if doi_cmd:
            all_article_data.update(doi_cmd["article_data"])

    content = build_summary(all_article_data)

    logger.info("Download tool run completed")
    logger.info("Final ToolMessages: %s", content)

    return Command(
        update = {
            "article_data": all_article_data,
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                artifact=all_article_data,
            )
        ],
        }
    )
