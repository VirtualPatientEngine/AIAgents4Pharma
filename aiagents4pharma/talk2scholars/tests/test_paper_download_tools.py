"""
Unit tests for arXiv paper downloading functionality, including:
- download_arxiv_paper tool function.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import ToolMessage

from aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input import (
    download_arxiv_paper,
)


class TestDownloadArxivPaper(unittest.TestCase):
    """tests for the download_arxiv_paper tool."""

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.hydra.initialize"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.hydra.compose"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.requests.get"
    )
    def test_download_arxiv_paper_success(
        self, mock_get, mock_compose, mock_initialize
    ):
        """test the download_arxiv_paper tool for successful retrieval of metadata and PDF URL."""
        # Set up a dummy Hydra config.
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_arxiv_paper.api_url = "http://dummy.arxiv.org/api"
        dummy_cfg.tools.download_arxiv_paper.request_timeout = 10
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        # Set up a dummy XML response with a valid entry including a pdf link.
        arxiv_id = "1234.56789"
        dummy_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Sample Paper Title</title>
                <author>
                    <name>Author One</name>
                </author>
                <author>
                    <name>Author Two</name>
                </author>
                <summary>This is a sample abstract.</summary>
                <published>2020-01-01T00:00:00Z</published>
                <link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}v1"/>
            </entry>
        </feed>
        """
        dummy_response = MagicMock()
        dummy_response.text = dummy_xml
        dummy_response.raise_for_status = MagicMock()
        mock_get.return_value = dummy_response

        tool_call_id = "test_tool_id"
        tool_input = {"arxiv_id": arxiv_id, "tool_call_id": tool_call_id}
        result = download_arxiv_paper.run(tool_input)
        update = result.update

        # Check that article_data was correctly set.
        self.assertIn("article_data", update)
        self.assertIn(arxiv_id, update["article_data"])
        metadata = update["article_data"][arxiv_id]
        self.assertEqual(metadata["Title"], "Sample Paper Title")
        self.assertEqual(metadata["Authors"], ["Author One", "Author Two"])
        self.assertEqual(metadata["Abstract"], "This is a sample abstract.")
        self.assertEqual(metadata["Publication Date"], "2020-01-01T00:00:00Z")
        self.assertEqual(metadata["URL"], f"http://arxiv.org/pdf/{arxiv_id}v1")
        self.assertEqual(metadata["pdf_url"], f"http://arxiv.org/pdf/{arxiv_id}v1")
        self.assertEqual(metadata["filename"], f"{arxiv_id}.pdf")
        self.assertEqual(metadata["source"], "arxiv")
        self.assertEqual(metadata["arxiv_id"], arxiv_id)

        # Check that the message content is as expected.
        messages = update["messages"]
        self.assertTrue(len(messages) >= 1)
        self.assertIsInstance(messages[0], ToolMessage)
        self.assertIn(
            f"Successfully retrieved metadata and PDF URL for arXiv ID {arxiv_id}",
            messages[0].content,
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.hydra.initialize"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.hydra.compose"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.requests.get"
    )
    def test_no_entry_found(self, mock_get, mock_compose, mock_initialize):
        """test the download_arxiv_paper tool for no entry found in XML response."""
        # Dummy config as before.
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_arxiv_paper.api_url = "http://dummy.arxiv.org/api"
        dummy_cfg.tools.download_arxiv_paper.request_timeout = 10
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        # Set up XML with no entry element.
        arxiv_id = "1234.56789"
        dummy_xml = (
            """<?xml version="1.0" encoding="UTF-8"?>"""
            """<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
        )
        dummy_response = MagicMock()
        dummy_response.text = dummy_xml
        dummy_response.raise_for_status = MagicMock()
        mock_get.return_value = dummy_response

        tool_call_id = "test_tool_id"
        tool_input = {"arxiv_id": arxiv_id, "tool_call_id": tool_call_id}
        with self.assertRaises(ValueError) as context:
            download_arxiv_paper.run(tool_input)
        self.assertEqual(
            str(context.exception), f"No entry found for arXiv ID {arxiv_id}"
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.hydra.initialize"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.hydra.compose"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.requests.get"
    )
    def test_no_pdf_url_found(self, mock_get, mock_compose, mock_initialize):
        """test the download_arxiv_paper tool for no PDF URL found in XML response."""
        # Dummy config.
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_arxiv_paper.api_url = "http://dummy.arxiv.org/api"
        dummy_cfg.tools.download_arxiv_paper.request_timeout = 10
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        # Set up XML with an entry that does not contain a pdf link.
        arxiv_id = "1234.56789"
        dummy_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Sample Paper Title</title>
                <author>
                    <name>Author One</name>
                </author>
                <summary>This is a sample abstract.</summary>
                <published>2020-01-01T00:00:00Z</published>
                <!-- Missing pdf link -->
            </entry>
        </feed>
        """
        dummy_response = MagicMock()
        dummy_response.text = dummy_xml
        dummy_response.raise_for_status = MagicMock()
        mock_get.return_value = dummy_response

        tool_call_id = "test_tool_id"
        tool_input = {"arxiv_id": arxiv_id, "tool_call_id": tool_call_id}
        with self.assertRaises(RuntimeError) as context:
            download_arxiv_paper.run(tool_input)
        self.assertEqual(
            str(context.exception), f"Could not find PDF URL for arXiv ID {arxiv_id}"
        )
