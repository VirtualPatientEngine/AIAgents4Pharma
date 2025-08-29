"""
Unit tests for ArxivDownloader.
Tests XML parsing, PDF URL construction, and metadata extraction.
"""

import unittest
import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch

import requests

from aiagents4pharma.talk2scholars.tools.paper_download.utils.arxiv_downloader import (
    ArxivDownloader,
)


class ArxivDownloaderTestShim(ArxivDownloader):
    """Public wrappers to exercise protected helpers without W0212."""

    def extract_basic_metadata_public(self, entry, ns):
        """extract_basic_metadata_public"""
        return self._extract_basic_metadata(entry, ns)

    def extract_title_public(self, entry, ns):
        """extract_title_public"""
        return self._extract_title(entry, ns)

    def extract_authors_public(self, entry, ns):
        """extract_authors_public"""
        return self._extract_authors(entry, ns)

    def extract_abstract_public(self, entry, ns):
        """extract_authors_public"""
        return self._extract_abstract(entry, ns)

    def extract_publication_date_public(self, entry, ns):
        """extract_publication_date_public"""
        return self._extract_publication_date(entry, ns)

    def extract_pdf_metadata_public(self, pdf_result, identifier):
        """extract_pdf_metadata_public"""
        return self._extract_pdf_metadata(pdf_result, identifier)

    def get_paper_identifier_info_public(self, paper):
        """get_paper_identifier_info_public"""
        return self._get_paper_identifier_info(paper)

    def add_service_identifier_public(self, entry, identifier):
        """add_service_identifier_public"""
        self._add_service_identifier(entry, identifier)


class TestArxivDownloader(unittest.TestCase):
    """Tests for the ArxivDownloader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.api_url = "http://export.arxiv.org/api/query"
        self.mock_config.pdf_base_url = "https://arxiv.org/pdf"
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192
        self.mock_config.xml_namespace = {"atom": "http://www.w3.org/2005/Atom"}

        # Use the testable subclass to avoid W0212 while still covering helpers
        self.downloader = ArxivDownloaderTestShim(self.mock_config)

        # Sample arXiv XML response
        self.sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678v1</id>
                <updated>2023-01-01T12:00:00Z</updated>
                <published>2023-01-01T12:00:00Z</published>
                <title>Test Paper Title</title>
                <summary>This is a test abstract for the paper.</summary>
                <author>
                    <name>John Doe</name>
                </author>
                <author>
                    <name>Jane Smith</name>
                </author>
                <link href="http://arxiv.org/abs/1234.5678v1" rel="alternate" type="text/html"/>
                <link href="http://arxiv.org/pdf/1234.5678v1.pdf"
                    rel="related"
                    type="application/pdf"
                    title="pdf"/>
            </entry>
        </feed>"""

    def test_initialization(self):
        """Test ArxivDownloader initialization."""
        self.assertEqual(self.downloader.api_url, "http://export.arxiv.org/api/query")
        self.assertEqual(self.downloader.pdf_base_url, "https://arxiv.org/pdf")
        self.assertEqual(self.downloader.request_timeout, 30)
        self.assertEqual(self.downloader.chunk_size, 8192)

    @patch("requests.get")
    def test_fetch_metadata_success(self, mock_get):
        """Test successful metadata fetching from arXiv API."""
        mock_response = Mock()
        mock_response.text = self.sample_xml
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.downloader.fetch_metadata("1234.5678")

        # Verify API call - it uses query string format, not params
        expected_url = (
            "http://export.arxiv.org/api/query?search_query=id:1234.5678&start=0&max_results=1"
        )
        mock_get.assert_called_once_with(expected_url, timeout=30)
        mock_response.raise_for_status.assert_called_once()

        # Verify XML parsing
        self.assertIsInstance(result, ET.Element)
        self.assertEqual(result.tag, "{http://www.w3.org/2005/Atom}feed")

    @patch("requests.get")
    def test_fetch_metadata_request_error(self, mock_get):
        """Test fetch_metadata with request error."""
        mock_get.side_effect = requests.RequestException("Network error")

        with self.assertRaises(requests.RequestException):
            self.downloader.fetch_metadata("1234.5678")

    @patch("requests.get")
    def test_fetch_metadata_invalid_xml(self, mock_get):
        """Test fetch_metadata with invalid XML response."""
        mock_response = Mock()
        mock_response.text = "Invalid XML content"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with self.assertRaises(ET.ParseError):
            self.downloader.fetch_metadata("1234.5678")

    @patch("requests.get")
    def test_fetch_metadata_no_entry_found(self, mock_get):
        """Test fetch_metadata when no entry is found in arXiv API response."""
        # XML response without any entry - note the namespace declarations
        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
            <title>ArXiv Query: search_query=all:1234.5678</title>
            <id>http://arxiv.org/api/query?search_query=all:1234.5678</id>
            <opensearch:totalResults>0</opensearch:totalResults>
            <opensearch:startIndex>0</opensearch:startIndex>
        </feed>"""

        mock_response = Mock()
        mock_response.text = empty_xml
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with self.assertRaises(RuntimeError) as context:
            self.downloader.fetch_metadata("1234.5678")

        self.assertIn("No entry found in arXiv API response", str(context.exception))

    def test_construct_pdf_url_from_metadata(self):
        """Test PDF URL construction from metadata."""
        metadata = ET.fromstring(self.sample_xml)

        result = self.downloader.construct_pdf_url(metadata, "1234.5678")

        # Should extract PDF URL from the link with title="pdf"
        self.assertEqual(result, "http://arxiv.org/pdf/1234.5678v1.pdf")

    def test_construct_pdf_url_fallback(self):
        """Test PDF URL construction fallback when not found in metadata."""
        # XML without PDF link
        xml_no_pdf = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678v1</id>
                <title>Test Paper Title</title>
                <link href="http://arxiv.org/abs/1234.5678v1" rel="alternate" type="text/html"/>
            </entry>
        </feed>"""

        metadata = ET.fromstring(xml_no_pdf)

        result = self.downloader.construct_pdf_url(metadata, "1234.5678")

        # Should fallback to constructed URL
        self.assertEqual(result, "https://arxiv.org/pdf/1234.5678.pdf")

    def test_construct_pdf_url_no_entry(self):
        """Test PDF URL construction with no entry in metadata."""
        xml_no_entry = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        metadata = ET.fromstring(xml_no_entry)

        result = self.downloader.construct_pdf_url(metadata, "1234.5678")

        # Should return empty string when no entry found
        self.assertEqual(result, "")

    def test_extract_paper_metadata_success(self):
        """Test successful paper metadata extraction."""
        metadata = ET.fromstring(self.sample_xml)
        pdf_result = ("/tmp/test.pdf", "test_paper.pdf")

        result = self.downloader.extract_paper_metadata(metadata, "1234.5678", pdf_result)

        # Verify extracted metadata
        expected_metadata = {
            "Title": "Test Paper Title",
            "Authors": ["John Doe", "Jane Smith"],
            "Abstract": "This is a test abstract for the paper.",
            "Publication Date": "2023-01-01T12:00:00Z",
            "URL": "/tmp/test.pdf",
            "pdf_url": "/tmp/test.pdf",
            "filename": "test_paper.pdf",
            "source": "arxiv",
            "arxiv_id": "1234.5678",
            "access_type": "open_access_downloaded",
            "temp_file_path": "/tmp/test.pdf",
        }

        self.assertEqual(result, expected_metadata)

    def test_extract_paper_metadata_no_pdf(self):
        """Test metadata extraction without PDF download."""
        metadata = ET.fromstring(self.sample_xml)

        with patch.object(self.downloader, "get_default_filename", return_value="1234.5678.pdf"):
            result = self.downloader.extract_paper_metadata(metadata, "1234.5678", None)

        # Verify metadata without PDF
        self.assertEqual(result["Title"], "Test Paper Title")
        self.assertEqual(result["URL"], "")
        self.assertEqual(result["pdf_url"], "")
        self.assertEqual(result["filename"], "1234.5678.pdf")
        self.assertEqual(result["access_type"], "download_failed")
        self.assertEqual(result["temp_file_path"], "")

    def test_extract_paper_metadata_no_entry(self):
        """Test metadata extraction with no entry in XML."""
        xml_no_entry = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>"""

        metadata = ET.fromstring(xml_no_entry)

        with self.assertRaises(RuntimeError) as context:
            self.downloader.extract_paper_metadata(metadata, "1234.5678", None)

        self.assertIn("No entry found in metadata", str(context.exception))

    def test_extract_basic_metadata(self):
        """Test basic metadata extraction helper method."""
        metadata = ET.fromstring(self.sample_xml)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = metadata.find("atom:entry", ns)

        result = self.downloader.extract_basic_metadata_public(entry, ns)

        expected = {
            "Title": "Test Paper Title",
            "Authors": ["John Doe", "Jane Smith"],
            "Abstract": "This is a test abstract for the paper.",
            "Publication Date": "2023-01-01T12:00:00Z",
        }
        self.assertEqual(result, expected)

    def test_extract_title_variants(self):
        """Title extraction for present and missing cases."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # Case 1: Title present
        metadata1 = ET.fromstring(self.sample_xml)
        entry1 = metadata1.find("atom:entry", ns)
        self.assertEqual(self.downloader.extract_title_public(entry1, ns), "Test Paper Title")

        # Case 2: Title missing
        xml_no_title = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678v1</id>
            </entry>
        </feed>"""
        metadata2 = ET.fromstring(xml_no_title)
        entry2 = metadata2.find("atom:entry", ns)
        self.assertEqual(self.downloader.extract_title_public(entry2, ns), "N/A")

    def test_extract_authors_variants(self):
        """Authors extraction for present and empty cases."""
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # Case 1: Authors present
        metadata1 = ET.fromstring(self.sample_xml)
        entry1 = metadata1.find("atom:entry", ns)
        self.assertEqual(
            self.downloader.extract_authors_public(entry1, ns),
            ["John Doe", "Jane Smith"],
        )

        # Case 2: Authors missing
        xml_no_authors = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678v1</id>
                <title>Test Paper Title</title>
            </entry>
        </feed>"""
        metadata2 = ET.fromstring(xml_no_authors)
        entry2 = metadata2.find("atom:entry", ns)
        self.assertEqual(self.downloader.extract_authors_public(entry2, ns), [])

    def test_extract_abstract_and_publication_date(self):
        """Abstract and publication date extraction."""
        metadata = ET.fromstring(self.sample_xml)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = metadata.find("atom:entry", ns)

        self.assertEqual(
            self.downloader.extract_abstract_public(entry, ns),
            "This is a test abstract for the paper.",
        )
        self.assertEqual(
            self.downloader.extract_publication_date_public(entry, ns),
            "2023-01-01T12:00:00Z",
        )

    def test_extract_pdf_metadata_variants(self):
        """PDF metadata extraction with and without a download result."""
        # With result
        pdf_result = ("/tmp/test.pdf", "paper.pdf")
        expected_with = {
            "URL": "/tmp/test.pdf",
            "pdf_url": "/tmp/test.pdf",
            "filename": "paper.pdf",
            "access_type": "open_access_downloaded",
            "temp_file_path": "/tmp/test.pdf",
        }
        self.assertEqual(
            self.downloader.extract_pdf_metadata_public(pdf_result, "1234.5678"),
            expected_with,
        )

        # Without result
        with patch.object(self.downloader, "get_default_filename", return_value="default.pdf"):
            expected_without = {
                "URL": "",
                "pdf_url": "",
                "filename": "default.pdf",
                "access_type": "download_failed",
                "temp_file_path": "",
            }
            self.assertEqual(
                self.downloader.extract_pdf_metadata_public(None, "1234.5678"),
                expected_without,
            )

    def test_service_and_identifier_helpers(self):
        """Service name, identifier name, and default filename helpers."""
        self.assertEqual(self.downloader.get_service_name(), "arXiv")
        self.assertEqual(self.downloader.get_identifier_name(), "arXiv ID")
        self.assertEqual(self.downloader.get_default_filename("1234.5678"), "1234.5678.pdf")

    def test_get_paper_identifier_info(self):
        """Test _get_paper_identifier_info method."""
        paper = {"arxiv_id": "1234.5678", "Publication Date": "2023-01-01T12:00:00Z"}

        result = self.downloader.get_paper_identifier_info_public(paper)

        self.assertIn("1234.5678", result)
        self.assertIn("2023-01-01", result)

    def test_add_service_identifier(self):
        """Test _add_service_identifier method."""
        entry = {}

        self.downloader.add_service_identifier_public(entry, "1234.5678")

        self.assertEqual(entry["arxiv_id"], "1234.5678")


class TestArxivDownloaderIntegration(unittest.TestCase):
    """Integration tests for ArxivDownloader with mocked external dependencies."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_config = Mock()
        self.mock_config.api_url = "http://export.arxiv.org/api/query"
        self.mock_config.pdf_base_url = "https://arxiv.org/pdf"
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192
        self.mock_config.xml_namespace = {"atom": "http://www.w3.org/2005/Atom"}

        self.downloader = ArxivDownloaderTestShim(self.mock_config)

        self.sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.5678v1</id>
                <published>2023-01-01T12:00:00Z</published>
                <title>Integration Test Paper</title>
                <summary>This is a test abstract.</summary>
                <author>
                    <name>Test Author</name>
                </author>
                <link href="http://arxiv.org/pdf/1234.5678v1.pdf"
                    rel="related"
                    type="application/pdf"
                    title="pdf"/>
            </entry>
        </feed>"""

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.utils."
        "arxiv_downloader.ArxivDownloader.download_pdf_to_temp"
    )
    @patch("requests.get")
    def test_full_paper_processing_workflow(self, mock_get, mock_download):
        """Test the complete workflow from identifier to processed paper data."""
        # Mock API response
        mock_response = Mock()
        mock_response.text = self.sample_xml
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Mock PDF download
        mock_download.return_value = ("/tmp/paper.pdf", "1234.5678.pdf")

        # Simulate the workflow
        identifiers = ["1234.5678"]
        results = {}

        for identifier in identifiers:
            # Step 1: Fetch metadata
            metadata = self.downloader.fetch_metadata(identifier)

            # Step 2: Construct PDF URL
            pdf_url = self.downloader.construct_pdf_url(metadata, identifier)

            # Step 3: Download PDF
            pdf_result = self.downloader.download_pdf_to_temp(pdf_url, identifier)

            # Step 4: Extract metadata
            paper_data = self.downloader.extract_paper_metadata(metadata, identifier, pdf_result)

            results[identifier] = paper_data

        # Verify the complete workflow
        self.assertIn("1234.5678", results)
        paper = results["1234.5678"]

        self.assertEqual(paper["Title"], "Integration Test Paper")
        self.assertEqual(paper["Authors"], ["Test Author"])
        self.assertEqual(paper["access_type"], "open_access_downloaded")
        self.assertEqual(paper["filename"], "1234.5678.pdf")
        self.assertEqual(paper["temp_file_path"], "/tmp/paper.pdf")

        # Verify method calls
        mock_get.assert_called_once()
        mock_download.assert_called_once_with("http://arxiv.org/pdf/1234.5678v1.pdf", "1234.5678")

    @patch("requests.get")
    def test_error_handling_workflow(self, mock_get):
        """Test error handling in the workflow."""
        # Mock network error
        mock_get.side_effect = requests.RequestException("Network error")

        with self.assertRaises(requests.RequestException):
            self.downloader.fetch_metadata("1234.5678")
