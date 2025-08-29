"""
Unit tests for BiorxivDownloader.
Tests CloudScraper integration, JSON API interaction, and PDF download with CloudFlare protection.
"""

import unittest
from unittest.mock import Mock, patch

import requests

from aiagents4pharma.talk2scholars.tools.paper_download.utils.biorxiv_downloader import (
    BiorxivDownloader,
)


class BiorxivDownloaderTestShim(BiorxivDownloader):
    """biorxiv_downloader test shim to expose protected methods."""

    __test__ = False

    def set_scraper(self, scraper):
        """set_scraper is a public method to set the scraper."""
        self._scraper = scraper

    def get_scraper_public(self):
        """get_scraper_public is a public method to access the scraper."""
        return self._get_scraper()

    def visit_landing_page_public(self, scraper, pdf_url, identifier):
        """call visit_landing_page with public access."""
        return self._visit_landing_page(scraper, pdf_url, identifier)

    def save_pdf_to_temp_public(self, response):
        """save_pdf_to_temp_public is a public method to save PDF response."""
        return self._save_pdf_to_temp(response)

    def extract_filename_public(self, response, identifier):
        """extract_filename_public is a public method to extract filename from response."""
        return self._extract_filename(response, identifier)

    def extract_basic_metadata_public(self, paper, identifier):
        """extract_basic_metadata_public is a public method to extract basic metadata."""
        return self._extract_basic_metadata(paper, identifier)

    def extract_authors_public(self, authors_str):
        """extract_authors_public is a public method to extract authors from a string."""
        return self._extract_authors(authors_str)

    def get_paper_identifier_info_public(self, paper):
        """get_paper_identifier_info_public is a public method to get paper identifier info."""
        return self._get_paper_identifier_info(paper)

    def add_service_identifier_public(self, entry, identifier):
        """add_service_identifier_public is a public method to add service identifier."""
        self._add_service_identifier(entry, identifier)


class TestBiorxivDownloader(unittest.TestCase):
    """Tests for the BiorxivDownloader class."""

    @patch("cloudscraper.create_scraper")
    def setUp(self, mock_create_scraper):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.api_url = "https://api.biorxiv.org/details"
        self.mock_config.pdf_url_template = (
            "https://www.biorxiv.org/content/{doi}v{version}.full.pdf"
        )
        self.mock_config.user_agent = "test-agent"
        self.mock_config.cf_clearance_timeout = 10
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192
        self.mock_config.session_reuse = True
        self.mock_config.default_version = "1"
        self.mock_config.browser_config = {"type": "custom"}

        # Mock the scraper creation during initialization
        mock_scraper = Mock()
        mock_create_scraper.return_value = mock_scraper

        self.downloader = BiorxivDownloaderTestShim(self.mock_config)
        self.initial_scraper = mock_scraper
        self.downloader.set_scraper(mock_scraper)

        # Sample bioRxiv API response
        self.sample_json_response = {
            "collection": [
                {
                    "title": "Test BioRxiv Paper",
                    "authors": "John Doe; Jane Smith",
                    "abstract": "This is a test abstract for bioRxiv paper.",
                    "date": "2023-01-01",
                    "category": "Biochemistry",
                    "version": "2",
                    "doi": "10.1101/2023.01.01.123456",
                }
            ]
        }

    def test_initialization(self):
        """Test BiorxivDownloader initialization."""
        self.assertEqual(self.downloader.api_url, "https://api.biorxiv.org/details")
        self.assertEqual(
            self.downloader.pdf_url_template,
            "https://www.biorxiv.org/content/{doi}v{version}.full.pdf",
        )
        self.assertEqual(self.downloader.user_agent, "test-agent")
        self.assertEqual(self.downloader.cf_clearance_timeout, 10)
        self.assertIsNotNone(self.downloader.get_scraper_public())

    def test_fetch_metadata_success(self):
        """Test successful metadata fetching from bioRxiv API."""
        mock_scraper = Mock()
        mock_response = Mock()
        mock_response.json.return_value = self.sample_json_response
        mock_response.raise_for_status = Mock()
        mock_scraper.get.return_value = mock_response

        # Mock the existing scraper
        self.downloader.set_scraper(mock_scraper)

        result = self.downloader.fetch_metadata("10.1101/2023.01.01.123456")

        # Verify API call
        expected_url = "https://api.biorxiv.org/details/biorxiv/10.1101/2023.01.01.123456/na/json"
        mock_scraper.get.assert_called_once_with(expected_url, timeout=30)
        mock_response.raise_for_status.assert_called_once()

        # Verify JSON parsing
        self.assertEqual(result, self.sample_json_response)

    def test_fetch_metadata_network_error(self):
        """Test fetch_metadata with network error."""
        mock_scraper = Mock()
        mock_scraper.get.side_effect = requests.RequestException("Network error")
        self.downloader.set_scraper(mock_scraper)

        with self.assertRaises(requests.RequestException):
            self.downloader.fetch_metadata("10.1101/2023.01.01.123456")

    def test_fetch_metadata_no_collection_data(self):
        """Test fetch_metadata when API response has no collection data."""
        mock_scraper = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {}  # Empty response
        mock_response.raise_for_status = Mock()
        mock_scraper.get.return_value = mock_response
        self.downloader.set_scraper(mock_scraper)

        with self.assertRaises(RuntimeError) as context:
            self.downloader.fetch_metadata("10.1101/2023.01.01.123456")

        self.assertIn("No collection data found", str(context.exception))

    def test_construct_pdf_url_variants(self):
        """PDF URL construction: normal, missing collection, default version."""
        # Success
        self.assertEqual(
            self.downloader.construct_pdf_url(
                self.sample_json_response, "10.1101/2023.01.01.123456"
            ),
            "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v2.full.pdf",
        )
        # No collection
        self.assertEqual(
            self.downloader.construct_pdf_url({}, "10.1101/2023.01.01.123456"),
            "",
        )
        # Default version
        meta_default = {"collection": [{"title": "Test Paper"}]}
        self.assertEqual(
            self.downloader.construct_pdf_url(meta_default, "10.1101/2023.01.01.123456"),
            "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf",
        )

    @patch("tempfile.NamedTemporaryFile")
    def test_download_pdf_to_temp_success(self, mock_tempfile):
        """Test successful PDF download with CloudScraper."""
        # Setup mock scraper
        mock_scraper = Mock()
        self.downloader.set_scraper(mock_scraper)

        # Mock landing page response
        mock_landing_response = Mock()
        mock_landing_response.raise_for_status = Mock()

        # Mock PDF download response
        mock_pdf_response = Mock()
        mock_pdf_response.raise_for_status = Mock()
        mock_pdf_response.iter_content.return_value = [
            b"PDF content chunk 1",
            b"PDF content chunk 2",
        ]
        mock_pdf_response.headers = {"Content-Disposition": 'attachment; filename="paper.pdf"'}

        mock_scraper.get.side_effect = [mock_landing_response, mock_pdf_response]

        # Mock temporary file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test.pdf"
        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_file.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp_file

        pdf_url = "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf"
        result = self.downloader.download_pdf_to_temp(pdf_url, "10.1101/2023.01.01.123456")

        # Verify result
        self.assertEqual(result, ("/tmp/test.pdf", "paper.pdf"))

        # Verify landing page visit
        landing_url = "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1"
        mock_scraper.get.assert_any_call(landing_url, timeout=30)

        # Verify PDF download
        mock_scraper.get.assert_any_call(pdf_url, timeout=30, stream=True)

        # Verify file writing
        mock_temp_file.write.assert_any_call(b"PDF content chunk 1")
        mock_temp_file.write.assert_any_call(b"PDF content chunk 2")

    def test_download_pdf_to_temp_error_variants(self):
        """Download errors: empty URL and network failure."""
        # Empty URL
        self.assertIsNone(self.downloader.download_pdf_to_temp("", "10.1101/x"))

        # Network error
        mock_scraper = Mock()
        mock_scraper.get.side_effect = requests.RequestException("Network error")
        self.downloader.set_scraper(mock_scraper)
        url = "https://www.biorxiv.org/content/10.1101/xv1.full.pdf"
        self.assertIsNone(self.downloader.download_pdf_to_temp(url, "10.1101/x"))

    @patch("cloudscraper.create_scraper")
    def test_get_scraper_new_and_existing(self, mock_create):
        """_get_scraper creates when missing and reuses when present."""
        # New scraper
        self.downloader.set_scraper(None)
        new_scraper = Mock()
        mock_create.return_value = new_scraper
        got = self.downloader.get_scraper_public()
        self.assertIs(got, new_scraper)
        mock_create.assert_called_once_with(browser={"custom": "test-agent"}, delay=10)

        # Existing scraper
        self.downloader.set_scraper(new_scraper)
        got2 = self.downloader.get_scraper_public()
        self.assertIs(got2, new_scraper)

    def test_visit_landing_page_variants(self):
        """Landing page visit happens only for .full.pdf URLs."""
        mock_scraper = Mock()
        ok = Mock()
        ok.raise_for_status = Mock()
        mock_scraper.get.return_value = ok

        # Case 1: with .full.pdf -> should visit landing
        pdf_url_full = "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf"
        self.downloader.visit_landing_page_public(
            mock_scraper, pdf_url_full, "10.1101/2023.01.01.123456"
        )
        expected = "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1"
        mock_scraper.get.assert_called_with(expected, timeout=30)

        # Case 2: no .full.pdf -> no call
        mock_scraper.get.reset_mock()
        pdf_url_plain = "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1"
        self.downloader.visit_landing_page_public(
            mock_scraper, pdf_url_plain, "10.1101/2023.01.01.123456"
        )
        mock_scraper.get.assert_not_called()

    @patch("tempfile.NamedTemporaryFile")
    def test_save_pdf_to_temp(self, mock_tempfile):
        """Test saving PDF response to temporary file."""
        mock_response = Mock()
        mock_response.iter_content.return_value = [
            b"chunk1",
            b"chunk2",
            None,
            b"chunk3",
        ]  # Include None chunk

        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/saved.pdf"
        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_file.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp_file

        result = self.downloader.save_pdf_to_temp_public(mock_response)

        self.assertEqual(result, "/tmp/saved.pdf")

        # Verify chunks were written (None chunk should be skipped)
        mock_temp_file.write.assert_any_call(b"chunk1")
        mock_temp_file.write.assert_any_call(b"chunk2")
        mock_temp_file.write.assert_any_call(b"chunk3")
        self.assertEqual(mock_temp_file.write.call_count, 3)

    def test_extract_filename_variants(self):
        """Filename extraction across header variants and regex-exception path."""
        cases = [
            (
                {"Content-Disposition": 'attachment; filename="test-paper.pdf"'},
                "test-paper.pdf",
                False,
            ),
            ({}, "default.pdf", False),
            ({"Content-Disposition": "invalid header format"}, "default.pdf", False),
            (
                {"Content-Disposition": 'attachment; filename="test.pdf"'},
                "default.pdf",
                True,
            ),  # trigger exception path
        ]
        for headers, expected, raise_regex in cases:
            with self.subTest(headers=headers, expected=expected, raise_regex=raise_regex):
                resp = Mock()
                resp.headers = headers
                if raise_regex:
                    with patch(
                        "re.search",
                        side_effect=requests.RequestException("Regex error"),
                    ):
                        with patch.object(
                            self.downloader,
                            "get_default_filename",
                            return_value="default.pdf",
                        ):
                            got = self.downloader.extract_filename_public(resp, "10.1101/test")
                else:
                    with patch.object(
                        self.downloader,
                        "get_default_filename",
                        return_value="default.pdf",
                    ):
                        got = self.downloader.extract_filename_public(resp, "10.1101/test")
                self.assertEqual(got, expected)

    def test_extract_paper_metadata_success(self):
        """Test successful paper metadata extraction."""
        metadata = self.sample_json_response
        pdf_result = ("/tmp/paper.pdf", "biorxiv_paper.pdf")

        result = self.downloader.extract_paper_metadata(
            metadata, "10.1101/2023.01.01.123456", pdf_result
        )

        expected = {
            "Title": "Test BioRxiv Paper",
            "Authors": ["John Doe", "Jane Smith"],
            "Abstract": "This is a test abstract for bioRxiv paper.",
            "Publication Date": "2023-01-01",
            "DOI": "10.1101/2023.01.01.123456",
            "Category": "Biochemistry",
            "Version": "2",
            "source": "biorxiv",
            "server": "biorxiv",
            "URL": "/tmp/paper.pdf",
            "pdf_url": "/tmp/paper.pdf",
            "filename": "biorxiv_paper.pdf",
            "access_type": "open_access_downloaded",
            "temp_file_path": "/tmp/paper.pdf",
        }

        self.assertEqual(result, expected)

    def test_extract_paper_metadata_no_pdf_result(self):
        """Test metadata extraction when PDF download failed."""
        metadata = self.sample_json_response
        pdf_result = None  # No PDF download result

        result = self.downloader.extract_paper_metadata(
            metadata, "10.1101/2023.01.01.123456", pdf_result
        )

        # Should still have basic metadata but with download_failed access type
        self.assertEqual(result["Title"], "Test BioRxiv Paper")
        self.assertEqual(result["access_type"], "download_failed")
        self.assertEqual(result["URL"], "")
        self.assertEqual(result["pdf_url"], "")
        self.assertEqual(result["temp_file_path"], "")
        self.assertEqual(result["filename"], "10_1101_2023_01_01_123456.pdf")  # Default filename

    def test_extract_paper_metadata_no_collection(self):
        """Test metadata extraction with missing collection."""
        metadata = {}

        with self.assertRaises(RuntimeError) as context:
            self.downloader.extract_paper_metadata(metadata, "10.1101/2023.01.01.123456", None)

        self.assertIn("No collection data found", str(context.exception))

    def test_extract_basic_metadata(self):
        """Test basic metadata extraction helper method."""
        paper = self.sample_json_response["collection"][0]

        result = self.downloader.extract_basic_metadata_public(paper, "10.1101/2023.01.01.123456")

        expected = {
            "Title": "Test BioRxiv Paper",
            "Authors": ["John Doe", "Jane Smith"],
            "Abstract": "This is a test abstract for bioRxiv paper.",
            "Publication Date": "2023-01-01",
            "DOI": "10.1101/2023.01.01.123456",
            "Category": "Biochemistry",
            "Version": "2",
            "source": "biorxiv",
            "server": "biorxiv",
        }

        self.assertEqual(result, expected)

    def test_extract_authors_variants(self):
        """Author parsing for semicolon list and empty string."""
        self.assertEqual(
            self.downloader.extract_authors_public("John Doe; Jane Smith; Bob Johnson"),
            ["John Doe", "Jane Smith", "Bob Johnson"],
        )
        self.assertEqual(self.downloader.extract_authors_public(""), [])

    def test_service_and_identifier_helpers(self):
        """Service name, identifier name, and default filename."""
        self.assertEqual(self.downloader.get_service_name(), "bioRxiv")
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
            "Category": "Biology",
        }

        result = self.downloader.get_paper_identifier_info_public(paper)

        self.assertIn("10.1101/2023.01.01.123456", result)
        self.assertIn("2023-01-01", result)
        self.assertIn("Biology", result)

    def test_add_service_identifier(self):
        """Test _add_service_identifier method."""
        entry = {}

        self.downloader.add_service_identifier_public(entry, "10.1101/2023.01.01.123456")

        self.assertEqual(entry["DOI"], "10.1101/2023.01.01.123456")
        self.assertEqual(entry["server"], "biorxiv")


class TestBiorxivDownloaderIntegration(unittest.TestCase):
    """Integration tests for BiorxivDownloader workflow."""

    @patch("cloudscraper.create_scraper")
    def setUp(self, mock_create_scraper):
        """Set up integration test fixtures."""
        self.mock_config = Mock()
        self.mock_config.api_url = "https://api.biorxiv.org/details"
        self.mock_config.pdf_url_template = (
            "https://www.biorxiv.org/content/{doi}v{version}.full.pdf"
        )
        self.mock_config.user_agent = "test-agent"
        self.mock_config.cf_clearance_timeout = 10
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192
        self.mock_config.session_reuse = True
        self.mock_config.default_version = "1"
        self.mock_config.browser_config = {"type": "custom"}

        # Mock the scraper creation during initialization
        mock_scraper = Mock()
        mock_create_scraper.return_value = mock_scraper

        self.downloader = BiorxivDownloaderTestShim(self.mock_config)

        self.sample_response = {
            "collection": [
                {
                    "title": "Integration Test Paper",
                    "authors": "Test Author",
                    "abstract": "Integration test abstract.",
                    "date": "2023-01-01",
                    "category": "Biology",
                    "version": "1",
                    "doi": "10.1101/2023.01.01.123456",
                }
            ]
        }

    @patch("tempfile.NamedTemporaryFile")
    def test_full_paper_processing_workflow(self, mock_tempfile):
        """Test the complete workflow from DOI to processed paper data."""
        # Mock scraper responses
        mock_scraper = Mock()
        mock_metadata_response = Mock()
        mock_metadata_response.json.return_value = self.sample_response
        mock_metadata_response.raise_for_status = Mock()

        # Mock landing page and PDF responses for download
        mock_landing_response = Mock()
        mock_landing_response.raise_for_status = Mock()

        mock_pdf_response = Mock()
        mock_pdf_response.raise_for_status = Mock()
        mock_pdf_response.iter_content.return_value = [b"PDF data"]
        mock_pdf_response.headers = {}

        # First call for metadata, then landing page, then PDF download
        mock_scraper.get.side_effect = [
            mock_metadata_response,
            mock_landing_response,
            mock_pdf_response,
        ]
        self.downloader.set_scraper(mock_scraper)

        # Mock temporary file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/integration.pdf"
        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_file.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp_file

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
        self.assertEqual(paper_data["temp_file_path"], "/tmp/integration.pdf")

        expected_pdf_url = "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf"
        self.assertEqual(pdf_url, expected_pdf_url)

        # Verify 3 calls: metadata, landing page, PDF
        self.assertEqual(mock_scraper.get.call_count, 3)

    def test_workflow_with_existing_scraper(self):
        """Test workflow reusing existing scraper instance."""
        # Set existing scraper
        existing_scraper = Mock()

        # Mock API response for metadata
        mock_response = Mock()
        mock_response.json.return_value = self.sample_response
        mock_response.raise_for_status = Mock()
        existing_scraper.get.return_value = mock_response

        self.downloader.set_scraper(existing_scraper)

        identifier = "10.1101/2023.01.01.123456"
        metadata = self.downloader.fetch_metadata(identifier)
        pdf_url = self.downloader.construct_pdf_url(metadata, identifier)

        # Try to download (will use existing scraper)
        with patch("tempfile.NamedTemporaryFile"):
            # Reset the mock and set up responses for landing + PDF
            existing_scraper.reset_mock()
            mock_landing = Mock()
            mock_landing.raise_for_status = Mock()
            mock_pdf = Mock()
            mock_pdf.raise_for_status = Mock()
            mock_pdf.iter_content.return_value = [b"data"]
            mock_pdf.headers = {}
            existing_scraper.get.side_effect = [mock_landing, mock_pdf]

            self.downloader.download_pdf_to_temp(pdf_url, identifier)

        # Should have used existing scraper for landing + PDF (2 calls)
        self.assertEqual(existing_scraper.get.call_count, 2)


class TestBiorxivCloudFlareHandling(unittest.TestCase):
    """Tests specific to CloudFlare protection handling."""

    @patch("cloudscraper.create_scraper")
    def setUp(self, mock_create_scraper):
        """Set up CloudFlare handling test fixtures."""
        self.mock_config = Mock()
        self.mock_config.api_url = "https://api.biorxiv.org/details"
        self.mock_config.pdf_url_template = (
            "https://www.biorxiv.org/content/{doi}v{version}.full.pdf"
        )
        self.mock_config.user_agent = "Mozilla/5.0 (compatible; test-agent)"
        self.mock_config.cf_clearance_timeout = 15
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192
        self.mock_config.session_reuse = True
        self.mock_config.default_version = "1"
        self.mock_config.browser_config = {"type": "custom"}

        # Mock the scraper creation during initialization
        mock_scraper = Mock()
        mock_create_scraper.return_value = mock_scraper

        self.downloader = BiorxivDownloaderTestShim(self.mock_config)

    @patch("cloudscraper.create_scraper")
    def test_cloudscraper_configuration(self, mock_create_scraper):
        """Test CloudScraper is configured with proper parameters."""
        # Set scraper to None so we create a new one
        self.downloader.set_scraper(None)
        mock_scraper = Mock()
        mock_create_scraper.return_value = mock_scraper

        scraper = self.downloader.get_scraper_public()

        mock_create_scraper.assert_called_once_with(
            browser={"custom": "Mozilla/5.0 (compatible; test-agent)"}, delay=15
        )
        self.assertEqual(scraper, mock_scraper)

    @patch("tempfile.NamedTemporaryFile")
    def test_landing_page_visit_before_pdf_download(self, mock_tempfile):
        """Test that landing page is visited before PDF download for CloudFlare bypass."""
        mock_scraper = Mock()
        self.downloader.set_scraper(mock_scraper)

        # Mock responses
        mock_landing_response = Mock()
        mock_landing_response.raise_for_status = Mock()

        mock_pdf_response = Mock()
        mock_pdf_response.raise_for_status = Mock()
        mock_pdf_response.iter_content.return_value = [b"PDF content"]
        mock_pdf_response.headers = {}

        mock_scraper.get.side_effect = [mock_landing_response, mock_pdf_response]

        # Mock temp file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test.pdf"
        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_file.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp_file

        pdf_url = "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf"
        self.downloader.download_pdf_to_temp(pdf_url, "10.1101/2023.01.01.123456")

        # Verify landing page was visited first
        landing_url = "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1"

        calls = mock_scraper.get.call_args_list
        self.assertEqual(len(calls), 2)

        # First call should be to landing page
        self.assertEqual(calls[0][0][0], landing_url)
        self.assertEqual(calls[0][1]["timeout"], 30)

        # Second call should be to PDF URL
        self.assertEqual(calls[1][0][0], pdf_url)
        self.assertEqual(calls[1][1]["timeout"], 30)
        self.assertEqual(calls[1][1]["stream"], True)
