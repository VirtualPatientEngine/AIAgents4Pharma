"""
Unit tests for medrXiv paper downloading functionality, including:
- download_medrxiv_paper tool function.
"""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import ToolMessage

from aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input import (
    download_medrxiv_paper,
)


class TestDownloadMedrxivPaper(unittest.TestCase):
    """Tests for the download_medrxiv_paper tool."""

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input.hydra.initialize"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input.hydra.compose"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input.requests.get"
    )
    def test_download_medrxiv_paper_success(self, mock_get, mock_compose, mock_initialize):
        """Test successful metadata and PDF URL retrieval."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_medrxiv_paper.api_url = "http://dummy.medrxiv.org/api"
        dummy_cfg.tools.download_medrxiv_paper.request_timeout = 10
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        doi = "10.1101/2025.04.25.25326432"

        dummy_response = MagicMock()
        dummy_response.status_code = 200
        dummy_response.raise_for_status = MagicMock()
        dummy_response.json.return_value = {
            "collection": [
                {
                    "title": "Sample Medrxiv Paper",
                    "authors": "Author One; Author Two",
                    "abstract": "This is a medRxiv abstract.",
                    "date": "2025-04-25",
                    "doi": doi,
                    "link": f"https://www.medrxiv.org/content/{doi}.full.pdf"
                }
            ]
        }
        mock_get.return_value = dummy_response

        tool_input = {"doi": doi, "tool_call_id": "test_tool_id"}
        result = download_medrxiv_paper.run(tool_input)
        update = result.update

        self.assertIn("article_data", update)
        self.assertIn(doi, update["article_data"])
        metadata = update["article_data"][doi]
        self.assertEqual(metadata["Title"], "Sample Medrxiv Paper")
        self.assertEqual(metadata["Authors"], "Author One; Author Two")
        self.assertEqual(metadata["Abstract"], "This is a medRxiv abstract.")
        self.assertEqual(metadata["Publication Date"], "2025-04-25")
        self.assertEqual(metadata["URL"], f"https://www.medrxiv.org/content/{doi}.full.pdf")
        self.assertEqual(metadata["pdf_url"], f"https://www.medrxiv.org/content/{doi}.full.pdf")
        self.assertEqual(metadata["filename"], f"{doi.rsplit('/', maxsplit=1)[-1]}.pdf")
        self.assertEqual(metadata["source"], "medrxiv")
        self.assertEqual(metadata["medrxiv_id"], doi)

        self.assertTrue(len(update["messages"]) >= 1)
        self.assertIsInstance(update["messages"][0], ToolMessage)
        self.assertIn("Successfully retrieved metadata and PDF URL", update["messages"][0].content)

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input.hydra.initialize"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input.hydra.compose"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input.requests.get"
    )
    def test_no_entry_found(self, mock_get, mock_compose, mock_initialize):
        """Test behavior when no 'entry' is in response."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_medrxiv_paper.api_url = "http://dummy.medrxiv.org/api"
        dummy_cfg.tools.download_medrxiv_paper.request_timeout = 10
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        dummy_response = MagicMock()
        dummy_response.status_code = 200
        dummy_response.raise_for_status = MagicMock()
        dummy_response.json.return_value = {}  # No entry
        mock_get.return_value = dummy_response

        doi = "10.1101/2025.04.25.25326432"
        tool_input = {"doi": doi, "tool_call_id": "test_tool_id"}

        with self.assertRaises(ValueError) as context:
            download_medrxiv_paper.run(tool_input)

        self.assertEqual(str(context.exception), f"No entry found for medRxiv ID {doi}")

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input.hydra.initialize"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input.hydra.compose"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_medrxiv_input.requests.get"
    )
    def test_no_pdf_url_found(self, mock_get, mock_compose, mock_initialize):
        """Test fallback to DOI-based PDF URL construction when 'link' is missing."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_medrxiv_paper.api_url = "http://dummy.medrxiv.org/api"
        dummy_cfg.tools.download_medrxiv_paper.request_timeout = 10
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        doi = "10.1101/2025.04.25.25326432"

        dummy_response = MagicMock()
        dummy_response.status_code = 200
        dummy_response.raise_for_status = MagicMock()
        dummy_response.json.return_value = {
            "collection": [
                {
                    "title": "Sample Medrxiv Paper",
                    "authors": "Author One; Author Two",
                    "abstract": "This is a medRxiv abstract.",
                    "date": "2025-04-25",
                    "doi": doi
                    # 'link' is intentionally omitted
                }
            ]
        }
        mock_get.return_value = dummy_response

        tool_input = {"doi": doi, "tool_call_id": "test_tool_id"}
        result = download_medrxiv_paper.run(tool_input)
        update = result.update
        metadata = update["article_data"][doi]

        # Assert that the PDF URL was constructed from DOI
        expected_suffix = doi.rsplit('/', maxsplit=1)[-1]
        expected_url = f"https://www.medrxiv.org/content/10.1101/{expected_suffix}.full.pdf"

        self.assertEqual(metadata["pdf_url"], expected_url)
