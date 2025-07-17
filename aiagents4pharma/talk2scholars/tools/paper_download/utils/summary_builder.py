#!/usr/bin/env python3

"""
Utility for building summary of article data papers.
"""

from typing import Any

def _get_snippet(abstract: str) -> str:
    """Extract the first one or two sentences from an abstract."""
    if not abstract or abstract == "N/A":
        return ""
    sentences = abstract.split(". ")
    snippet_sentences = sentences[:2]
    snippet = ". ".join(snippet_sentences)
    if not snippet.endswith("."):
        snippet += "."
    return snippet


def build_summary(article_data: dict[str, Any]) -> str:
    """Build a summary string for up to three papers with snippets."""
    top = list(article_data.values())[:3]
    lines: list[str] = []
    for idx, paper in enumerate(top):
        title = paper.get("Title", "N/A")
        pub_date = paper.get("Publication Date", "N/A")
        url = paper.get("URL", "")
        snippet = _get_snippet(paper.get("Abstract", ""))
        line = f"{idx+1}. {title} ({pub_date})"
        if url:
            line += f"\n   View PDF: {url}"
        if snippet:
            line += f"\n   Abstract snippet: {snippet}"
        lines.append(line)
    summary = "\n".join(lines)
    return (
        "Download was successful. Papers metadata are attached as an artifact. "
        "Here is a summary of the results:\n"
        f"Number of papers found: {len(article_data)}\n"
        "Top 3 papers:\n" + summary
    )
