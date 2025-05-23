"""
Unit tests for pubmed paper downloading functionality, including:
- download_pubmedx_paper tool function.
"""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import ToolMessage
from aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper import (
    download_pubmedx_paper,
)

class TestDownloadPubMedXPaper(unittest.TestCase):
    """Tests for the download_pubmedx_paper tool."""
    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.hydra.initialize"
)
    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.hydra.compose"
    )
    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.requests.get"
    )
    def test_download_pubmedx_paper_success(self, mock_get, mock_compose, mock_initialize):
        """Test successful retrieval of metadata and PDF URL."""
        # Dummy config
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_pubmed_paper.metadata_url = "http://dummy.pubmed.org/api"
        dummy_cfg.tools.download_pubmed_paper.pdf_base_url = "http://dummy.pubmed.org/pdf/"
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        # PMC ID
        pmc_id = "PMC123456"

        # Dummy XML with required fields
        dummy_xml = """
        <article>
            <front>
                <article-meta>
                    <title-group>
                        <article-title>Sample PubMedX Title</article-title>
                    </title-group>
                    <abstract>This is a sample abstract.</abstract>
                    <contrib-group>
                        <contrib contrib-type=\"author\">
                            <name>
                                <given-names>John</given-names>
                                <surname>Doe</surname>
                            </name>
                        </contrib>
                    </contrib-group>
                    <pub-date>
                        <year>2020</year>
                    </pub-date>
                </article-meta>
            </front>
        </article>
        """
        dummy_response_metadata = MagicMock()
        dummy_response_metadata.text = dummy_xml
        dummy_response_metadata.raise_for_status = MagicMock()

        dummy_response_pdf = MagicMock()
        dummy_response_pdf.status_code = 200

        mock_get.side_effect = [dummy_response_metadata, dummy_response_pdf]

        tool_call_id = "test_tool_id"
        tool_input = {"pmc_id": pmc_id, "tool_call_id": tool_call_id}
        result = download_pubmedx_paper.run(tool_input)
        update = result.update

        self.assertIn("article_data", update)
        self.assertIn(pmc_id, update["article_data"])
        metadata = update["article_data"][pmc_id]
        self.assertEqual(metadata["Title"], "Sample PubMedX Title")
        self.assertIn("John Doe", metadata["Authors"])
        self.assertEqual(metadata["Abstract"], "This is a sample abstract.")
        self.assertEqual(metadata["URL"], f"http://dummy.pubmed.org/pdf/{pmc_id}?pdf=render")

        messages = update["messages"]
        self.assertTrue(len(messages) >= 1)
        self.assertIsInstance(messages[0], ToolMessage)
        self.assertIn(f"Successfully retrieved metadata for PMC ID {pmc_id}",messages[0].content)

    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.hydra.initialize"
    )
    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.hydra.compose"
    )
    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.requests.get"
    )
    def test_download_pubmedx_paper_empty_metadata(self, mock_get, mock_compose, mock_initialize):
        """Test case where the XML contains no article metadata."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_pubmed_paper.metadata_url = "http://dummy.pubmed.org/api"
        dummy_cfg.tools.download_pubmed_paper.pdf_base_url = "http://dummy.pubmed.org/pdf/"
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        pmc_id = "PMC999999"

        # XML without expected metadata fields
        dummy_xml = "<article><front></front></article>"
        dummy_response_metadata = MagicMock()
        dummy_response_metadata.text = dummy_xml
        dummy_response_metadata.raise_for_status = MagicMock()

        dummy_response_pdf = MagicMock()
        dummy_response_pdf.status_code = 200

        mock_get.side_effect = [dummy_response_metadata, dummy_response_pdf]

        tool_call_id = "test_tool_id"
        tool_input = {"pmc_id": pmc_id, "tool_call_id": tool_call_id}

        result = download_pubmedx_paper.run(tool_input)
        update = result.update
        metadata = update["article_data"][pmc_id]

        self.assertEqual(metadata["Title"], "N/A")
        self.assertEqual(metadata["Authors"], "N/A")
        self.assertEqual(metadata["Abstract"], "N/A")
        self.assertEqual(metadata["Publication Date"], "N/A")
        self.assertEqual(metadata["pdf_url"], f"http://dummy.pubmed.org/pdf/{pmc_id}?pdf=render")

    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.hydra.initialize"
    )
    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.hydra.compose"
    )
    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.requests.get"
    )
    def test_pdf_not_found(self, mock_get, mock_compose, mock_initialize):
        """Test the case when the PDF cannot be retrieved."""
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_pubmed_paper.metadata_url = "http://dummy.pubmed.org/api"
        dummy_cfg.tools.download_pubmed_paper.pdf_base_url = "http://dummy.pubmed.org/pdf/"
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        pmc_id = "PMC123456"

        dummy_xml = """
        <article>
            <front>
                <article-meta>
                    <title-group>
                        <article-title>Sample PubMedX Title</article-title>
                    </title-group>
                </article-meta>
            </front>
        </article>
        """
        dummy_response_metadata = MagicMock()
        dummy_response_metadata.text = dummy_xml
        dummy_response_metadata.raise_for_status = MagicMock()

        dummy_response_pdf = MagicMock()
        dummy_response_pdf.status_code = 404

        mock_get.side_effect = [dummy_response_metadata, dummy_response_pdf]

        tool_call_id = "test_tool_id"
        tool_input = {"pmc_id": pmc_id, "tool_call_id": tool_call_id}

        with self.assertRaises(RuntimeError) as context:
            download_pubmedx_paper.run(tool_input)

        self.assertIn("No PDF found or access denied", str(context.exception))

    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.hydra.initialize"
    )
    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.hydra.compose"
    )
    @patch(
    "aiagents4pharma.talk2scholars.tools.paper_download.download_pubmed_paper.requests.get"
    )
    def test_download_with_mapping_to_pmc_id(self, mock_get, mock_compose, mock_initialize):
        """Test case where the id cannot be mapped to pmc id."""
        # Dummy config
        dummy_cfg = MagicMock()
        dummy_cfg.tools.download_pubmed_paper.metadata_url = "http://dummy.pubmed.org/api"
        dummy_cfg.tools.download_pubmed_paper.pdf_base_url = "http://dummy.pubmed.org/pdf/"
        dummy_cfg.tools.download_pubmed_paper.map_url = "http://dummy.pubmed.org/map"
        mock_compose.return_value = dummy_cfg
        mock_initialize.return_value.__enter__.return_value = None

        # Input for successful mapping
        non_pmc_input_id = "10.1000/test-doi"
        resolved_pmc_id = "PMC123456"

        # --- Mock `map_ids` XML response ---
        dummy_map_response = MagicMock()
        dummy_map_response.status_code = 200
        dummy_map_response.text = f"""
        <eSummaryResult>
            <record pmcid="{resolved_pmc_id}"/>
        </eSummaryResult>
        """

        # --- Mock metadata XML response ---
        dummy_metadata_response = MagicMock()
        dummy_metadata_response.status_code = 200
        dummy_metadata_response.text = """
        <article>
            <front>
                <article-meta>
                    <title-group>
                        <article-title>Mapped Paper Title</article-title>
                    </title-group>
                    <abstract>Mapped abstract here</abstract>
                    <contrib-group>
                        <contrib contrib-type="author">
                            <name>
                                <given-names>Jane</given-names>
                                <surname>Smith</surname>
                            </name>
                        </contrib>
                    </contrib-group>
                    <pub-date><year>2022</year></pub-date>
                </article-meta>
            </front>
        </article>
        """

        # --- Mock PDF availability check ---
        dummy_pdf_response = MagicMock()
        dummy_pdf_response.status_code = 200

        # --- CASE 1: Success path ---
        mock_get.side_effect = [
            dummy_map_response,          # map_ids
            dummy_metadata_response,     # fetch_metadata
            dummy_pdf_response           # PDF check
        ]

        result = download_pubmedx_paper.run(
            {"pmc_id": non_pmc_input_id,
            "tool_call_id": "mock_tool_id"}
        )

        metadata = result.update["article_data"][resolved_pmc_id]
        self.assertEqual(metadata["Title"], "Mapped Paper Title")
        self.assertEqual(metadata["Abstract"], "Mapped abstract here")
        self.assertIn("Jane Smith", metadata["Authors"])
        self.assertEqual(metadata["Publication Date"], "N/A")
        self.assertTrue(metadata["URL"].startswith("http://dummy.pubmed.org/pdf/PMC123456"))

        # --- CASE 2: Mapping fails — test RuntimeError ---
        failed_map_response = MagicMock()
        failed_map_response.status_code = 200
        failed_map_response.text = """
        <eSummaryResult>
            <!-- No record with pmcid -->
        </eSummaryResult>
        """

        mock_get.side_effect = [failed_map_response]

        with self.assertRaises(RuntimeError) as context:
            download_pubmedx_paper.run(
                {"pmc_id": "10.1000/unknown-doi",
                "tool_call_id": "mock_tool_id"}
            )
        self.assertIn("PMC id not found for 10.1000/unknown-doi", str(context.exception))
