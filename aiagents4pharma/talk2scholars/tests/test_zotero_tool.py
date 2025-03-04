"""
Unit tests for Zotero search tool in zotero_read.py.
"""

from unittest.mock import patch
from langgraph.types import Command
from ..tools.zotero.zotero_read import zotero_search_tool


# Mock data for Zotero API response
MOCK_ZOTERO_RESPONSE = [
    {
        "data": {
            "key": "ABC123",
            "title": "Deep Learning in Medicine",
            "abstractNote": "An overview of deep learning applications in medicine.",
            "date": "2022",
            "url": "https://example.com/paper1",
            "itemType": "journalArticle",
        }
    },
    {
        "data": {
            "key": "XYZ789",
            "title": "Advances in AI",
            "abstractNote": "Recent advancements in AI research.",
            "date": "2023",
            "url": "https://example.com/paper2",
            "itemType": "conferencePaper",
        }
    },
]


class TestZoteroRead:
    """Unit tests for Zotero search tool"""

    @patch("pyzotero.zotero.Zotero.items")
    def test_zotero_success(self, mock_zotero):
        """Verifies successful retrieval of papers from Zotero"""
        mock_zotero.return_value = MOCK_ZOTERO_RESPONSE

        result = zotero_search_tool.invoke(
            input={
                "query": "deep learning",
                "only_articles": True,
                "limit": 2,
                "tool_call_id": "test123",
            }
        )

        assert isinstance(result, Command)
        assert "zotero_read" in result.update
        assert "messages" in result.update
        papers = result.update["zotero_read"]
        assert len(papers) == 2  # Should return 2 papers
        assert papers["ABC123"]["Title"] == "Deep Learning in Medicine"
        assert papers["XYZ789"]["Title"] == "Advances in AI"

    @patch("pyzotero.zotero.Zotero.items")
    def test_zotero_no_papers_found(self, mock_zotero):
        """Verifies Zotero tool behavior when no papers are found"""
        mock_zotero.return_value = []  # Simulating empty response

        result = zotero_search_tool.invoke(
            input={
                "query": "nonexistent topic",
                "only_articles": True,
                "limit": 2,
                "tool_call_id": "test123",
            }
        )

        assert isinstance(result, Command)
        assert "zotero_read" in result.update
        assert len(result.update["zotero_read"]) == 0  # No papers found
        assert "messages" in result.update
        assert "Number of papers found: 0" in result.update["messages"][0].content

    @patch("pyzotero.zotero.Zotero.items")
    def test_zotero_only_articles_filtering(self, mock_zotero):
        """Ensures only journal articles and conference papers are returned"""
        mock_response = [
            {
                "data": {
                    "key": "DEF456",
                    "title": "A Book on AI",
                    "abstractNote": "Book about AI advancements.",
                    "date": "2021",
                    "url": "https://example.com/book",
                    "itemType": "book",
                }
            },
            MOCK_ZOTERO_RESPONSE[0],  # Valid journal article
        ]
        mock_zotero.return_value = mock_response

        result = zotero_search_tool.invoke(
            input={
                "query": "AI",
                "only_articles": True,
                "limit": 2,
                "tool_call_id": "test123",
            }
        )

        assert isinstance(result, Command)
        assert "zotero_read" in result.update
        papers = result.update["zotero_read"]
        assert len(papers) == 1  # The book should be filtered out
        assert "ABC123" in papers  # Journal article should be included

    @patch("pyzotero.zotero.Zotero.items")
    def test_zotero_invalid_response(self, mock_zotero):
        """Tests handling of malformed API response"""
        mock_zotero.return_value = [
            {"data": None},  # Invalid response format
            {},  # Empty object
            {"data": {"title": "Missing Key", "itemType": "journalArticle"}},  # No key
        ]

        result = zotero_search_tool.invoke(
            input={
                "query": "AI ethics",
                "only_articles": True,
                "limit": 2,
                "tool_call_id": "test123",
            }
        )

        assert isinstance(result, Command)
        assert "zotero_read" in result.update
        assert (
            len(result.update["zotero_read"]) == 0
        )  # Should filter out invalid entries
        assert "messages" in result.update
        assert "Number of papers found: 0" in result.update["messages"][0].content

    @patch("pyzotero.zotero.Zotero.items")
    def test_zotero_handles_non_dict_items(self, mock_zotero):
        """Ensures that Zotero tool correctly skips non-dictionary items (covers line 86)"""

        # Simulate Zotero returning an invalid item (e.g., `None` and a string)
        mock_zotero.return_value = [
            None,
            "invalid_string",
            {
                "data": {
                    "key": "123",
                    "title": "Valid Paper",
                    "itemType": "journalArticle",
                }
            },
        ]

        result = zotero_search_tool.invoke(
            input={
                "query": "AI ethics",
                "only_articles": True,
                "limit": 2,
                "tool_call_id": "test123",
            }
        )

        assert isinstance(result, Command)
        assert "zotero_read" in result.update

        # Expect only valid items to be processed
        assert (
            len(result.update["zotero_read"]) == 1
        ), "Only valid dictionary items should be processed"
