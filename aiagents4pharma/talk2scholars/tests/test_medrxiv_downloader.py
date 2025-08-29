"""
Unit tests for MedrxivDownloader.
Tests JSON API interaction, PDF URL construction, and metadata extraction.
"""

import json
import unittest
from unittest.mock import Mock, patch

import requests

from aiagents4pharma.talk2scholars.tools.paper_download.utils.medrxiv_downloader import (
    MedrxivDownloader,
)


# ---- Test-only shim to access protected helpers without pylint W0212 ----
class MedrxivDownloaderTestShim(MedrxivDownloader):
    """mock class to access protected methods for testing."""

    __test__ = False  # prevent pytest collection

    def extract_basic_metadata_public(self, paper, identifier):
        """extract basic metadata from a paper."""
        return self._extract_basic_metadata(paper, identifier)

    def extract_authors_public(self, authors_str):
        """extract authors from a semicolon-separated string."""
        return self._extract_authors(authors_str)

    def extract_pdf_metadata_public(self, pdf_result, identifier):
        """extract PDF metadata from the download result."""
        return self._extract_pdf_metadata(pdf_result, identifier)

    def get_paper_identifier_info_public(self, paper):
        """get paper identifier info for public use."""
        return self._get_paper_identifier_info(paper)

    def add_service_identifier_public(self, entry, identifier):
        """add service identifier to an entry."""
        self._add_service_identifier(entry, identifier)


class TestMedrxivDownloader(unittest.TestCase):
    """Tests for the MedrxivDownloader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.api_url = "https://api.medrxiv.org/details"
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192
        self.mock_config.pdf_url_template = (
            "https://www.medrxiv.org/content/{identifier}v{version}.full.pdf"
        )
        self.mock_config.default_version = "1"

        self.downloader = MedrxivDownloaderTestShim(self.mock_config)

        # Sample medRxiv API response
        self.sample_json_response = {
            "collection": [
                {
                    "title": "Test MedRxiv Paper",
                    "authors": "John Doe; Jane Smith",
                    "abstract": "This is a test abstract for medRxiv paper.",
                    "date": "2023-01-01",
                    "category": "Infectious Diseases",
                    "version": "1",
                    "doi": "10.1101/2023.01.01.123456",
                }
            ]
        }

    def test_initialization(self):
        """Test MedrxivDownloader initialization."""
        self.assertEqual(self.downloader.api_url, "https://api.medrxiv.org/details")
        self.assertEqual(self.downloader.request_timeout, 30)
        self.assertEqual(self.downloader.chunk_size, 8192)

    @patch("requests.get")
    def test_fetch_metadata_success(self, mock_get):
        """Test successful metadata fetching from medRxiv API."""
        mock_response = Mock()
        mock_response.json.return_value = self.sample_json_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.downloader.fetch_metadata("10.1101/2023.01.01.123456")

        # Verify API call - should include /medrxiv/ and /na/json
        expected_url = "https://api.medrxiv.org/details/medrxiv/10.1101/2023.01.01.123456/na/json"
        mock_get.assert_called_once_with(expected_url, timeout=30)
        mock_response.raise_for_status.assert_called_once()

        # Verify JSON parsing
        self.assertEqual(result, self.sample_json_response)

    @patch("requests.get")
    def test_fetch_metadata_network_error(self, mock_get):
        """Test fetch_metadata with network error."""
        mock_get.side_effect = requests.RequestException("Network error")

        with self.assertRaises(requests.RequestException):
            self.downloader.fetch_metadata("10.1101/2023.01.01.123456")

    @patch("requests.get")
    def test_fetch_metadata_json_decode_error(self, mock_get):
        """Test fetch_metadata with JSON decode error."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        with self.assertRaises(json.JSONDecodeError):
            self.downloader.fetch_metadata("10.1101/2023.01.01.123456")

    def test_construct_pdf_url_variants(self):
        """PDF URL construction: normal, missing/empty collection, custom version."""
        # Success
        self.assertEqual(
            self.downloader.construct_pdf_url(
                self.sample_json_response, "10.1101/2023.01.01.123456"
            ),
            "https://www.medrxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf",
        )
        # No collection
        self.assertEqual(
            self.downloader.construct_pdf_url({}, "10.1101/2023.01.01.123456"),
            "",
        )
        # Empty collection
        self.assertEqual(
            self.downloader.construct_pdf_url({"collection": []}, "10.1101/2023.01.01.123456"),
            "",
        )
        # Custom version
        self.assertEqual(
            self.downloader.construct_pdf_url(
                {"collection": [{"title": "Test Paper", "version": "3"}]},
                "10.1101/2023.01.01.123456",
            ),
            "https://www.medrxiv.org/content/10.1101/2023.01.01.123456v3.full.pdf",
        )

    def test_extract_paper_metadata_success(self):
        """Test successful paper metadata extraction."""
        metadata = self.sample_json_response
        pdf_result = ("/tmp/paper.pdf", "medrxiv_paper.pdf")

        result = self.downloader.extract_paper_metadata(
            metadata, "10.1101/2023.01.01.123456", pdf_result
        )

        expected = {
            "Title": "Test MedRxiv Paper",
            "Authors": ["John Doe", "Jane Smith"],
            "Abstract": "This is a test abstract for medRxiv paper.",
            "Publication Date": "2023-01-01",
            "DOI": "10.1101/2023.01.01.123456",
            "Category": "Infectious Diseases",
            "Version": "1",
            "source": "medrxiv",
            "server": "medrxiv",
            "URL": "/tmp/paper.pdf",
            "pdf_url": "/tmp/paper.pdf",
            "filename": "medrxiv_paper.pdf",
            "access_type": "open_access_downloaded",
            "temp_file_path": "/tmp/paper.pdf",
        }

        self.assertEqual(result, expected)

    def test_extract_paper_metadata_no_pdf(self):
        """Test metadata extraction without PDF download."""
        metadata = self.sample_json_response

        with patch.object(self.downloader, "get_default_filename", return_value="default.pdf"):
            result = self.downloader.extract_paper_metadata(
                metadata, "10.1101/2023.01.01.123456", None
            )

        self.assertEqual(result["Title"], "Test MedRxiv Paper")
        self.assertEqual(result["URL"], "")
        self.assertEqual(result["access_type"], "download_failed")
        self.assertEqual(result["filename"], "default.pdf")

    def test_extract_paper_metadata_no_collection(self):
        """Test metadata extraction with missing collection."""
        metadata = {}

        with self.assertRaises(RuntimeError) as context:
            self.downloader.extract_paper_metadata(metadata, "10.1101/2023.01.01.123456", None)

        self.assertIn("No collection data found", str(context.exception))

    def test_extract_basic_metadata_variants(self):
        """Basic metadata extraction: complete and missing fields."""
        # Complete
        paper_full = self.sample_json_response["collection"][0]
        got_full = self.downloader.extract_basic_metadata_public(
            paper_full, "10.1101/2023.01.01.123456"
        )
        expected_full = {
            "Title": "Test MedRxiv Paper",
            "Authors": ["John Doe", "Jane Smith"],
            "Abstract": "This is a test abstract for medRxiv paper.",
            "Publication Date": "2023-01-01",
            "DOI": "10.1101/2023.01.01.123456",
            "Category": "Infectious Diseases",
            "Version": "1",
            "source": "medrxiv",
            "server": "medrxiv",
        }
        self.assertEqual(got_full, expected_full)

        # Missing fields
        paper_missing = {"title": "Test Paper"}  # Missing others
        got_missing = self.downloader.extract_basic_metadata_public(paper_missing, "10.1101/test")
        self.assertEqual(got_missing["Title"], "Test Paper")
        self.assertEqual(got_missing["Authors"], [])
        self.assertEqual(got_missing["Abstract"], "N/A")
        self.assertEqual(got_missing["Category"], "N/A")

    def test_extract_authors_variants(self):
        """Author parsing from semicolon string, empty, and whitespace-heavy inputs."""
        self.assertEqual(
            self.downloader.extract_authors_public("John Doe; Jane Smith; Bob Johnson"),
            ["John Doe", "Jane Smith", "Bob Johnson"],
        )
        self.assertEqual(self.downloader.extract_authors_public(""), [])
        self.assertEqual(
            self.downloader.extract_authors_public("  John Doe  ;  Jane Smith  ; "),
            ["John Doe", "Jane Smith"],
        )

    def test_extract_pdf_metadata_variants(self):
        """PDF metadata: with and without download result."""
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
            self.downloader.extract_pdf_metadata_public(pdf_result, "10.1101/test"),
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
                self.downloader.extract_pdf_metadata_public(None, "10.1101/test"),
                expected_without,
            )

    def test_service_and_identifier_helpers(self):
        """Service, identifier, and default filename helpers."""
        self.assertEqual(self.downloader.get_service_name(), "medRxiv")
        self.assertEqual(self.downloader.get_identifier_name(), "DOI")
        self.assertEqual(
            self.downloader.get_default_filename("10.1101/2023.01.01.123456"),
            "10_1101_2023_01_01_123456.pdf",
        )

    def test_get_paper_identifier_info(self):
        """Test _get_paper_identifier_info method."""
        paper = {
            "DOI": "10.1101/2023.01.01.123456",
            "Publication Date": "2023-01-01",
            "Category": "Medicine",
        }

        result = self.downloader.get_paper_identifier_info_public(paper)

        self.assertIn("10.1101/2023.01.01.123456", result)
        self.assertIn("2023-01-01", result)
        self.assertIn("Medicine", result)

    def test_add_service_identifier(self):
        """Test _add_service_identifier method."""
        entry = {}
        self.downloader.add_service_identifier_public(entry, "10.1101/2023.01.01.123456")
        self.assertEqual(entry["DOI"], "10.1101/2023.01.01.123456")
        self.assertEqual(entry["server"], "medrxiv")


class TestMedrxivDownloaderIntegration(unittest.TestCase):
    """Integration tests for MedrxivDownloader workflow."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_config = Mock()
        self.mock_config.api_url = "https://api.medrxiv.org/details"
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192
        self.mock_config.pdf_url_template = (
            "https://www.medrxiv.org/content/{identifier}v{version}.full.pdf"
        )
        self.mock_config.default_version = "1"

        self.downloader = MedrxivDownloaderTestShim(self.mock_config)

        self.sample_response = {
            "collection": [
                {
                    "title": "Integration Test Paper",
                    "authors": "Test Author",
                    "abstract": "Integration test abstract.",
                    "date": "2023-01-01",
                    "category": "Medicine",
                    "version": "2",
                    "doi": "10.1101/2023.01.01.123456",
                }
            ]
        }

    @patch(
        "aiagents4pharma.talk2scholars.tools.paper_download.utils."
        "medrxiv_downloader.MedrxivDownloader.download_pdf_to_temp"
    )
    @patch("requests.get")
    def test_full_paper_processing_workflow(self, mock_get, mock_download):
        """Test the complete workflow from DOI to processed paper data."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = self.sample_response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Mock PDF download
        mock_download.return_value = ("/tmp/paper.pdf", "medrxiv_paper.pdf")

        # Simulate the workflow
        identifier = "10.1101/2023.01.01.123456"

        # Step 1: Fetch metadata
        metadata = self.downloader.fetch_metadata(identifier)

        # Step 2: Construct PDF URL
        pdf_url = self.downloader.construct_pdf_url(metadata, identifier)

        # Step 3: Download PDF
        pdf_result = self.downloader.download_pdf_to_temp(pdf_url, identifier)

        # Step 4: Extract metadata
        paper_data = self.downloader.extract_paper_metadata(metadata, identifier, pdf_result)

        # Verify the complete workflow
        self.assertEqual(paper_data["Title"], "Integration Test Paper")
        self.assertEqual(paper_data["Authors"], ["Test Author"])
        self.assertEqual(paper_data["access_type"], "open_access_downloaded")
        self.assertEqual(paper_data["filename"], "medrxiv_paper.pdf")
        self.assertEqual(paper_data["temp_file_path"], "/tmp/paper.pdf")

        # Verify method calls
        mock_get.assert_called_once_with(
            "https://api.medrxiv.org/details/medrxiv/10.1101/2023.01.01.123456/na/json",
            timeout=30,
        )
        expected_pdf_url = "https://www.medrxiv.org/content/10.1101/2023.01.01.123456v2.full.pdf"
        mock_download.assert_called_once_with(expected_pdf_url, identifier)

    @patch("requests.get")
    def test_error_handling_workflow(self, mock_get):
        """Test error handling in the workflow."""
        # Mock API error
        mock_get.side_effect = requests.RequestException("API error")

        with self.assertRaises(requests.RequestException):
            self.downloader.fetch_metadata("10.1101/2023.01.01.123456")

    @patch("requests.get")
    def test_workflow_with_empty_collection(self, mock_get):
        """Test workflow with empty collection response."""
        # Mock API response with empty collection - this should raise error in fetch_metadata
        mock_response = Mock()
        mock_response.json.return_value = {"collection": []}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        identifier = "10.1101/2023.01.01.123456"

        # Should raise error in fetch_metadata when collection is empty
        with self.assertRaises(RuntimeError) as context:
            self.downloader.fetch_metadata(identifier)

        self.assertIn("No collection data found in medRxiv API response", str(context.exception))

    @patch("requests.get")
    def test_multiple_identifiers_workflow(self, mock_get):
        """Test processing multiple identifiers."""
        # Mock different responses for different DOIs
        responses = [
            {"collection": [{"title": "Paper 1", "version": "1", "authors": "Author 1"}]},
            {"collection": [{"title": "Paper 2", "version": "2", "authors": "Author 2"}]},
        ]

        mock_responses = []
        for response in responses:
            mock_resp = Mock()
            mock_resp.json.return_value = response
            mock_resp.raise_for_status = Mock()
            mock_responses.append(mock_resp)

        mock_get.side_effect = mock_responses

        identifiers = ["10.1101/2023.01.01.111111", "10.1101/2023.01.01.222222"]
        results = {}

        for identifier in identifiers:
            metadata = self.downloader.fetch_metadata(identifier)
            _ = self.downloader.construct_pdf_url(metadata, identifier)  # ensure path covered
            paper_data = self.downloader.extract_paper_metadata(metadata, identifier, None)
            results[identifier] = paper_data

        # Verify both papers were processed
        self.assertEqual(len(results), 2)
        self.assertEqual(results["10.1101/2023.01.01.111111"]["Title"], "Paper 1")
        self.assertEqual(results["10.1101/2023.01.01.222222"]["Title"], "Paper 2")

        # Verify API calls with correct URLs
        self.assertEqual(mock_get.call_count, 2)
        expected_calls = [
            "https://api.medrxiv.org/details/medrxiv/10.1101/2023.01.01.111111/na/json",
            "https://api.medrxiv.org/details/medrxiv/10.1101/2023.01.01.222222/na/json",
        ]
        actual_urls = [call[0][0] for call in mock_get.call_args_list]
        for expected_url in expected_calls:
            self.assertIn(expected_url, actual_urls)


class TestMedrxivSpecialCases(unittest.TestCase):
    """Tests for special cases and edge conditions."""

    def setUp(self):
        """Set up test fixtures for special cases."""
        self.mock_config = Mock()
        self.mock_config.api_url = "https://api.medrxiv.org/details"
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192
        self.mock_config.pdf_url_template = (
            "https://www.medrxiv.org/content/{identifier}v{version}.full.pdf"
        )
        self.mock_config.default_version = "1"

        self.downloader = MedrxivDownloaderTestShim(self.mock_config)

    def test_filename_generation_special_characters(self):
        """Test filename generation with special characters in DOI."""
        doi_with_special_chars = "10.1101/2023.01.01.123456/special-chars_test"

        result = self.downloader.get_default_filename(doi_with_special_chars)

        # Should replace problematic characters
        self.assertEqual(result, "10_1101_2023_01_01_123456_special-chars_test.pdf")

    def test_version_handling_edge_cases(self):
        """Test PDF URL construction with various version formats."""
        test_cases = [
            ({"collection": [{"version": ""}]}, "v.full.pdf"),  # Empty version
            ({"collection": [{"version": None}]}, "vNone.full.pdf"),  # None version
            ({"collection": [{}]}, "v1.full.pdf"),  # Missing version key defaults to 1
        ]

        for metadata, expected_suffix in test_cases:
            result = self.downloader.construct_pdf_url(metadata, "10.1101/test")
            self.assertTrue(result.endswith(expected_suffix))

    def test_metadata_extraction_unicode_handling(self):
        """Test metadata extraction with Unicode characters."""
        metadata = {
            "collection": [
                {
                    "title": "Título com acentos é símbolos especiais",
                    "authors": "José María; François Müller",
                    "abstract": "Resumo com çaracteres especiais ñ símbolos",
                    "date": "2023-01-01",
                    "category": "Médecine",
                    "version": "1",
                }
            ]
        }

        result = self.downloader.extract_paper_metadata(metadata, "10.1101/test", None)

        # Should handle Unicode properly
        self.assertEqual(result["Title"], "Título com acentos é símbolos especiais")
        self.assertEqual(result["Authors"], ["José María", "François Müller"])
        self.assertEqual(result["Abstract"], "Resumo com çaracteres especiais ñ símbolos")
