"""
Unit tests for arXiv paper downloading functionality, including:
- download_arxiv_paper tool function.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage

from aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input import (
    _get_snippet,
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
        dummy_response = MagicMock()
        dummy_response.text = (
            f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
        <feed xmlns=\"http://www.w3.org/2005/Atom\">"""
            f"            <entry>"
            f"<title>Sample Paper Title</title>"
            f"<author><name>Author One</name></author>"
            f"<author><name>Author Two</name></author>"
            f"<summary>This is a sample abstract.</summary>"
            f"<published>2020-01-01T00:00:00Z</published>"
            f'<link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}v1"/>'
            f"</entry></feed>"
        )
        dummy_response.raise_for_status = MagicMock()
        mock_get.return_value = dummy_response

        tool_call_id = "test_tool_id"
        tool_input = {"arxiv_ids": [arxiv_id], "tool_call_id": tool_call_id}
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

        # Check that the message content matches the new summary format
        messages = update["messages"]
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], ToolMessage)
        content = messages[0].content
        # Build expected summary
        expected = (
            "Download was successful. Papers metadata are attached as an artifact. "
            "Here is a summary of the results:\n"
            f"Number of papers found: 1\n"
            "Top 3 papers:\n"
            f"1. Sample Paper Title (2020-01-01T00:00:00Z)\n"
            f"   View PDF: http://arxiv.org/pdf/{arxiv_id}v1\n"
            "   Abstract snippet: This is a sample abstract."
        )
        self.assertEqual(content, expected)

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
        tool_input = {"arxiv_ids": [arxiv_id], "tool_call_id": tool_call_id}
        # No entry found should result in empty article_data and header-only summary
        result = download_arxiv_paper.run(tool_input)
        update = result.update
        self.assertIn("article_data", update)
        self.assertEqual(update["article_data"], {})
        messages = update.get("messages", [])
        self.assertEqual(len(messages), 1)
        content = messages[0].content
        expected = (
            "Download was successful. Papers metadata are attached as an artifact. "
            "Here is a summary of the results:\n"
            "Number of papers found: 0\n"
            "Top 3 papers:\n"
        )
        self.assertEqual(content, expected)

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
        tool_input = {"arxiv_ids": [arxiv_id], "tool_call_id": tool_call_id}
        with self.assertRaises(RuntimeError) as context:
            download_arxiv_paper.run(tool_input)
        self.assertEqual(
            str(context.exception), f"Could not find PDF URL for arXiv ID {arxiv_id}"
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.extract_metadata"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_"
        "arxiv_input.fetch_arxiv_metadata"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.hydra.compose"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.download_arxiv_input.hydra.initialize"
    )
    def test_summary_multiple_papers(
        self, mock_initialize, mock_compose, _mock_fetch, mock_extract
    ):
        """Test summary includes '...and N more papers.' when more than 3 papers."""
        # Dummy config
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_arxiv_paper.api_url = "http://dummy"
        dummy_cfg.tools.download_arxiv_paper.request_timeout = 5
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        # Simulate metadata extraction for multiple papers
        def dummy_meta(_entry, _ns, aid):
            """dummy metadata extraction function."""
            return {
                "Title": f"T{aid}",
                "Publication Date": "2020-01-01T00:00:00Z",
                "URL": f"u{aid}v1",
            }

        mock_extract.side_effect = dummy_meta
        # Prepare 5 paper IDs
        ids = [str(i) for i in range(5)]
        tool_input = {"arxiv_ids": ids, "tool_call_id": "tid"}
        result = download_arxiv_paper.run(tool_input)
        summary = result.update["messages"][0].content
        # Should report total count of 5 and list only top 3 without ellipsis
        assert "Number of papers found: 5" in summary
        assert "Top 3 papers:" in summary
        # Entries for first three IDs should include URL and no ellipsis
        assert "1. T0 (2020-01-01T00:00:00Z)" in summary
        assert "   View PDF: u0v1" in summary
        assert "3. T2 (2020-01-01T00:00:00Z)" in summary
        assert "...and" not in summary


@pytest.mark.parametrize(
    "input_text,expected",
    [
        ("", ""),
        ("N/A", ""),
        ("Just one sentence", "Just one sentence."),
        ("First. Second", "First. Second."),
        ("Hello. World.", "Hello. World."),
    ],
)
def test_get_snippet_various(input_text, expected):
    """Test _get_snippet behavior for various abstracts."""
    assert _get_snippet(input_text) == expected
