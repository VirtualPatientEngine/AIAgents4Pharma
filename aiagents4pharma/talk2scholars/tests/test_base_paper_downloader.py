"""
Unit tests for BasePaperDownloader.
Tests the abstract base class functionality and common methods.
"""

import inspect
import unittest
from typing import Any
from unittest.mock import Mock, patch

import requests

from aiagents4pharma.talk2scholars.tools.paper_download.utils.base_paper_downloader import (
    BasePaperDownloader,
)


class ConcretePaperDownloader(BasePaperDownloader):
    """Concrete implementation of BasePaperDownloader for testing."""

    def __init__(self, config: Any):
        super().__init__(config)
        self.test_metadata = {"test": "data"}

    def fetch_metadata(self, identifier: str) -> Any:
        """Concrete implementation for testing."""
        return self.test_metadata

    def construct_pdf_url(self, metadata: Any, identifier: str) -> str:
        """Concrete implementation for testing."""
        return f"https://test.com/{identifier}.pdf"

    def extract_paper_metadata(
        self, metadata: Any, identifier: str, pdf_result: tuple[str, str] | None
    ) -> dict[str, Any]:
        """Concrete implementation for testing."""
        return {
            "Title": f"Test Paper {identifier}",
            "Authors": ["Test Author"],
            "identifier": identifier,
            "metadata_source": metadata,
        }

    def get_service_name(self) -> str:
        """Concrete implementation for testing."""
        return "TestService"

    def get_identifier_name(self) -> str:
        """Concrete implementation for testing."""
        return "Test ID"

    def get_default_filename(self, identifier: str) -> str:
        """Concrete implementation for testing."""
        return f"test_{identifier}.pdf"

    def _get_paper_identifier_info(self, paper: dict[str, Any]) -> str:
        """Concrete implementation for testing."""
        return f" ({paper.get('identifier', 'unknown')})"

    def _add_service_identifier(self, entry: dict[str, Any], identifier: str) -> None:
        """Concrete implementation for testing."""
        entry["test_id"] = identifier

    def get_paper_identifier_info_public(self, paper: dict[str, Any]) -> str:
        """Public wrapper to access protected identifier info for tests."""
        return self._get_paper_identifier_info(paper)

    def add_service_identifier_public(self, entry: dict[str, Any], identifier: str) -> None:
        """Public wrapper to access protected service identifier for tests."""
        self._add_service_identifier(entry, identifier)


class TestBasePaperDownloader(unittest.TestCase):
    """Tests for the BasePaperDownloader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192

        self.downloader = ConcretePaperDownloader(self.mock_config)

    def test_initialization(self):
        """Test BasePaperDownloader initialization."""
        self.assertEqual(self.downloader.request_timeout, 30)
        self.assertEqual(self.downloader.chunk_size, 8192)

    def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract methods are unimplemented in an incomplete subclass."""

        # Create an intentionally incomplete subclass **without** instantiating it
        # (avoid E0110) and without a pointless 'pass' (avoid W0107).
        class IncompleteDownloader(BasePaperDownloader):
            """Intentionally incomplete concrete subclass for introspection only."""

            __test__ = False  # not a test class

        # Assert it's abstract instead of trying to instantiate
        self.assertTrue(inspect.isabstract(IncompleteDownloader))

    @patch("tempfile.NamedTemporaryFile")
    @patch("requests.get")
    def test_download_pdf_to_temp_success(self, mock_get, mock_tempfile):
        """Test successful PDF download to temporary file."""
        # Mock response
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content.return_value = [b"PDF chunk 1", b"PDF chunk 2"]
        mock_response.headers = {"Content-Disposition": 'attachment; filename="paper.pdf"'}
        mock_get.return_value = mock_response

        # Mock temporary file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test.pdf"
        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_file.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp_file

        result = self.downloader.download_pdf_to_temp("https://test.com/paper.pdf", "12345")

        # Verify result
        self.assertEqual(result, ("/tmp/test.pdf", "paper.pdf"))

        # Verify HTTP request - includes headers with User-Agent
        expected_headers = {"User-Agent": self.downloader.user_agent}
        mock_get.assert_called_once_with(
            "https://test.com/paper.pdf",
            headers=expected_headers,
            timeout=30,
            stream=True,
        )
        mock_response.raise_for_status.assert_called_once()

        # Verify file writing
        mock_temp_file.write.assert_any_call(b"PDF chunk 1")
        mock_temp_file.write.assert_any_call(b"PDF chunk 2")

    def test_download_pdf_to_temp_empty_url(self):
        """Test PDF download with empty URL."""
        result = self.downloader.download_pdf_to_temp("", "12345")

        self.assertIsNone(result)

    @patch("requests.get")
    def test_download_pdf_to_temp_network_error(self, mock_get):
        """Test PDF download with network error."""
        mock_get.side_effect = requests.RequestException("Network error")

        result = self.downloader.download_pdf_to_temp("https://test.com/paper.pdf", "12345")

        self.assertIsNone(result)

    @patch("tempfile.NamedTemporaryFile")
    @patch("requests.get")
    def test_download_pdf_to_temp_filename_extraction(self, mock_get, mock_tempfile):
        """Test filename extraction from Content-Disposition header."""
        # Mock response with various header formats
        test_cases = [
            ('attachment; filename="test-paper.pdf"', "test-paper.pdf"),
            ("attachment; filename=simple.pdf", "simple.pdf"),
            (
                "attachment; filename*=UTF-8''encoded%20file.pdf",
                "12345.pdf",
            ),  # Complex header format falls back to default
            ('inline; filename="quoted file.pdf"', "quoted file.pdf"),
            ("", "12345.pdf"),  # No header, should use default
        ]

        for header_value, expected_filename in test_cases:
            with self.subTest(header=header_value):
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.iter_content.return_value = [b"PDF data"]
                mock_response.headers = (
                    {"Content-Disposition": header_value} if header_value else {}
                )
                mock_get.return_value = mock_response

                # Mock get_default_filename for fallback case
                with patch.object(
                    self.downloader, "get_default_filename", return_value="12345.pdf"
                ):
                    # Mock temporary file
                    mock_temp_file = Mock()
                    mock_temp_file.name = "/tmp/test.pdf"
                    mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                    mock_temp_file.__exit__ = Mock(return_value=None)
                    mock_tempfile.return_value = mock_temp_file

                    result = self.downloader.download_pdf_to_temp(
                        "https://test.com/paper.pdf", "12345"
                    )

                self.assertEqual(result[1], expected_filename)

    def test_process_identifiers_success(self):
        """Test successful processing of multiple identifiers."""
        identifiers = ["12345", "67890"]

        # Mock download_pdf_to_temp to return different results
        with patch.object(self.downloader, "download_pdf_to_temp") as mock_download:
            mock_download.side_effect = [
                ("/tmp/paper1.pdf", "paper1.pdf"),  # First paper succeeds
                None,  # Second paper fails
            ]

            result = self.downloader.process_identifiers(identifiers)

        # Verify results
        self.assertIn("12345", result)
        self.assertIn("67890", result)

        # First paper should have PDF data
        self.assertEqual(result["12345"]["Title"], "Test Paper 12345")
        self.assertEqual(result["12345"]["Authors"], ["Test Author"])

        # Second paper should also be processed (but without PDF)
        self.assertEqual(result["67890"]["Title"], "Test Paper 67890")

    def test_process_identifiers_with_errors(self):
        """Test processing identifiers with various errors."""
        identifiers = ["valid", "fetch_error"]

        def mock_fetch_metadata(identifier):
            if identifier == "fetch_error":
                raise requests.RequestException("Fetch failed")
            return {"test": identifier}

        with patch.object(self.downloader, "fetch_metadata", side_effect=mock_fetch_metadata):
            with patch.object(self.downloader, "download_pdf_to_temp", return_value=None):
                result = self.downloader.process_identifiers(identifiers)

        # Valid identifier should succeed
        self.assertIn("valid", result)
        self.assertEqual(result["valid"]["Title"], "Test Paper valid")

        # Error cases should create error entries (not be excluded)
        self.assertIn("fetch_error", result)
        self.assertEqual(result["fetch_error"]["Title"], "Error fetching paper")
        self.assertIn("Fetch failed", result["fetch_error"]["Abstract"])
        self.assertEqual(result["fetch_error"]["access_type"], "error")

    def test_build_summary_success(self):
        """Test building summary for successful downloads."""
        article_data = {
            "paper1": {"Title": "Paper 1", "access_type": "open_access_downloaded"},
            "paper2": {"Title": "Paper 2", "access_type": "download_failed"},
            "paper3": {"Title": "Paper 3", "access_type": "open_access_downloaded"},
        }

        result = self.downloader.build_summary(article_data)

        # Should include count of papers and successful downloads
        self.assertIn("3", result)  # Total papers
        self.assertIn("2", result)  # Successful downloads
        self.assertIn("TestService", result)  # Service name

    def test_build_summary_no_papers(self):
        """Test building summary with no papers."""
        result = self.downloader.build_summary({})

        self.assertIn("0", result)
        self.assertIn("TestService", result)

    def test_build_summary_all_failed(self):
        """Test building summary with all failed downloads."""
        article_data = {
            "paper1": {"Title": "Paper 1", "access_type": "download_failed"},
            "paper2": {"Title": "Paper 2", "access_type": "download_failed"},
        }

        result = self.downloader.build_summary(article_data)

        self.assertIn("2", result)  # Total papers
        self.assertIn("0", result)  # Successful downloads (should be 0)

    def test_build_summary_with_papers(self):
        """Test building summary with paper list."""
        article_data = {
            "123": {
                "Title": "Paper 1",
                "identifier": "123",
                "access_type": "open_access_downloaded",
                "Abstract": "Test abstract.",
            },
            "456": {
                "Title": "Paper 2",
                "identifier": "456",
                "access_type": "download_failed",
                "Abstract": "Another abstract.",
            },
        }

        result = self.downloader.build_summary(article_data)

        self.assertIn("Paper 1", result)
        self.assertIn("Paper 2", result)
        self.assertIn("TestService", result)
        self.assertIn("2", result)  # Total papers
        self.assertIn("1", result)  # Successfully downloaded

    def test_build_summary_truncated_list(self):
        """Test building summary with long list (should show only top 3)."""
        article_data = {}
        for i in range(5):  # More than 3
            article_data[f"{i + 1}"] = {
                "Title": f"Paper {i + 1}",
                "identifier": f"{i + 1}",
                "access_type": "open_access_downloaded",
                "Abstract": f"Abstract {i + 1}",
            }

        result = self.downloader.build_summary(article_data)

        # Should include first 3 papers only
        self.assertIn("Paper 1", result)
        self.assertIn("Paper 2", result)
        self.assertIn("Paper 3", result)

        # Should not include papers 4 and 5
        self.assertNotIn("Paper 4", result)
        self.assertNotIn("Paper 5", result)

        # Should show total count
        self.assertIn("5", result)  # Total papers

    def test_concrete_implementation_methods(self):
        """Test that concrete implementations work correctly."""
        # Test fetch_metadata
        metadata = self.downloader.fetch_metadata("test123")
        self.assertEqual(metadata, {"test": "data"})

        # Test construct_pdf_url
        pdf_url = self.downloader.construct_pdf_url(metadata, "test123")
        self.assertEqual(pdf_url, "https://test.com/test123.pdf")

        # Test extract_paper_metadata
        paper_data = self.downloader.extract_paper_metadata(metadata, "test123", None)
        self.assertEqual(paper_data["Title"], "Test Paper test123")
        self.assertEqual(paper_data["Authors"], ["Test Author"])

        # Test get_service_name
        service_name = self.downloader.get_service_name()
        self.assertEqual(service_name, "TestService")

        # Test get_identifier_name
        identifier_name = self.downloader.get_identifier_name()
        self.assertEqual(identifier_name, "Test ID")

        # Test get_default_filename
        filename = self.downloader.get_default_filename("test123")
        self.assertEqual(filename, "test_test123.pdf")

    def test_helper_methods(self):
        """Test helper methods."""
        # Test _get_paper_identifier_info via public wrapper
        paper = {"identifier": "test123"}
        info = self.downloader.get_paper_identifier_info_public(paper)
        self.assertEqual(info, " (test123)")

        # Test _add_service_identifier via public wrapper
        entry = {}
        self.downloader.add_service_identifier_public(entry, "test123")
        self.assertEqual(entry["test_id"], "test123")

    def test_abstract_methods_raise_not_implemented_direct_call(self):
        """Test that base-class abstract methods raise NotImplementedError when called."""
        # Use the already-imported BasePaperDownloader (no reimport/redefinition).

        # Public abstract methods: call directly on the base to hit the NotImplementedError paths.
        with self.assertRaises(NotImplementedError):
            BasePaperDownloader.fetch_metadata(self.downloader, "test")

        with self.assertRaises(NotImplementedError):
            BasePaperDownloader.construct_pdf_url(self.downloader, {}, "test")

        with self.assertRaises(NotImplementedError):
            BasePaperDownloader.extract_paper_metadata(self.downloader, {}, "test", None)

        with self.assertRaises(NotImplementedError):
            BasePaperDownloader.get_service_name(self.downloader)

        with self.assertRaises(NotImplementedError):
            BasePaperDownloader.get_identifier_name(self.downloader)

        with self.assertRaises(NotImplementedError):
            BasePaperDownloader.get_default_filename(self.downloader, "test")

        # Protected abstract methods: call via getattr to avoid W0212 while still executing code.
        method_name_1 = "_get_paper_identifier_info"
        with self.assertRaises(NotImplementedError):
            getattr(BasePaperDownloader, method_name_1)(self.downloader, {})

        method_name_2 = "_add_service_identifier"
        with self.assertRaises(NotImplementedError):
            getattr(BasePaperDownloader, method_name_2)(self.downloader, {}, "test")

    @patch("tempfile.NamedTemporaryFile")
    @patch("requests.get")
    def test_filename_extraction_exception_handling(self, mock_get, mock_tempfile):
        """Test exception handling during filename extraction."""
        # Mock response that will cause an exception in filename extraction
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content.return_value = [b"PDF data"]
        mock_response.headers = {"Content-Disposition": 'attachment; filename="paper.pdf"'}
        mock_get.return_value = mock_response

        # Mock temporary file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test.pdf"
        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_file.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp_file

        # Patch re.search to raise an exception during filename extraction
        with patch("re.search", side_effect=requests.RequestException("Regex error")):
            result = self.downloader.download_pdf_to_temp("https://test.com/paper.pdf", "12345")

        # Should still succeed but use default filename due to exception
        self.assertEqual(result, ("/tmp/test.pdf", "test_12345.pdf"))

    def test_build_summary_with_temp_file_path(self):
        """Test build_summary with papers that have temp_file_path."""
        article_data = {
            "paper1": {
                "Title": "Paper 1",
                "access_type": "open_access_downloaded",
                "Abstract": "This is a test abstract with multiple sentences."
                "It should be truncated.",
                "temp_file_path": "/tmp/paper1.pdf",
            },
            "paper2": {
                "Title": "Paper 2",
                "access_type": "download_failed",
                "Abstract": "Short abstract.",
                "temp_file_path": "",  # Empty temp_file_path
            },
        }

        result = self.downloader.build_summary(article_data)

        # Should include temp file path for paper1
        self.assertIn("/tmp/paper1.pdf", result)
        self.assertIn("Downloaded to:", result)
        self.assertIn("Abstract snippet:", result)

        # Should include count information
        self.assertIn("2", result)  # Total papers
        self.assertIn("1", result)  # Successfully downloaded


class TestBasePaperDownloaderEdgeCases(unittest.TestCase):
    """Tests for edge cases and error conditions."""

    def setUp(self):
        """Set up edge case test fixtures."""
        self.mock_config = Mock()
        self.mock_config.request_timeout = 30
        self.mock_config.chunk_size = 8192

        self.downloader = ConcretePaperDownloader(self.mock_config)

    @patch("tempfile.NamedTemporaryFile")
    @patch("requests.get")
    def test_download_pdf_chunk_filtering(self, mock_get, mock_tempfile):
        """Test that empty chunks are filtered out during download."""
        # Mock response with mixed chunks including None/empty ones
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_content.return_value = [
            b"chunk1",
            None,  # Should be filtered out
            b"",  # Empty chunk, should be filtered out
            b"chunk2",
            None,
            b"chunk3",
        ]
        mock_response.headers = {}
        mock_get.return_value = mock_response

        # Mock temporary file
        mock_temp_file = Mock()
        mock_temp_file.name = "/tmp/test.pdf"
        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
        mock_temp_file.__exit__ = Mock(return_value=None)
        mock_tempfile.return_value = mock_temp_file

        with patch.object(self.downloader, "get_default_filename", return_value="default.pdf"):
            # Call without assigning to avoid 'unused-variable'
            self.downloader.download_pdf_to_temp("https://test.com/paper.pdf", "12345")

        # Should only write non-empty chunks
        self.assertEqual(mock_temp_file.write.call_count, 3)
        mock_temp_file.write.assert_any_call(b"chunk1")
        mock_temp_file.write.assert_any_call(b"chunk2")
        mock_temp_file.write.assert_any_call(b"chunk3")

    def test_filename_extraction_regex_edge_cases(self):
        """Test filename extraction with various regex edge cases."""
        test_headers = [
            # Various quote combinations
            ('filename="file with spaces.pdf"', "file with spaces.pdf"),
            (
                "filename='single_quotes.pdf'",
                "default.pdf",
            ),  # Single quotes don't match regex
            ("filename=no_quotes.pdf", "no_quotes.pdf"),
            # Unicode and special characters
            ('filename="файл.pdf"', "файл.pdf"),
            (
                'filename="file-with-dashes_and_underscores.pdf"',
                "file-with-dashes_and_underscores.pdf",
            ),
            # Edge cases
            ('filename=""', "default.pdf"),  # Empty filename falls back to default
            ("filename=", "default.pdf"),  # No value falls back to default
            (
                'other_param=value; filename="actual.pdf"',
                "actual.pdf",
            ),  # Mixed parameters
            # Invalid cases (should fall back to default)
            ("invalid_header_format", None),
            ("filename=not_a_pdf.txt", "default.pdf"),  # Non-PDF falls back to default
        ]

        for header_value, expected in test_headers:
            with self.subTest(header=header_value):
                with patch("requests.get") as mock_get:
                    mock_response = Mock()
                    mock_response.raise_for_status = Mock()
                    mock_response.iter_content.return_value = [b"data"]
                    mock_response.headers = {"Content-Disposition": header_value}
                    mock_get.return_value = mock_response

                    with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                        mock_temp_file = Mock()
                        mock_temp_file.name = "/tmp/test.pdf"
                        mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
                        mock_temp_file.__exit__ = Mock(return_value=None)
                        mock_tempfile.return_value = mock_temp_file

                        with patch.object(
                            self.downloader,
                            "get_default_filename",
                            return_value="default.pdf",
                        ):
                            result = self.downloader.download_pdf_to_temp(
                                "https://test.com/paper.pdf", "12345"
                            )

                if expected is None:
                    # Should fall back to default
                    self.assertEqual(result[1], "default.pdf")
                else:
                    self.assertEqual(result[1], expected)

    def test_process_identifiers_empty_list(self):
        """Test processing empty identifier list."""
        result = self.downloader.process_identifiers([])

        self.assertEqual(result, {})

    def test_process_identifiers_duplicate_handling(self):
        """Test processing list with duplicate identifiers."""
        identifiers = ["12345", "67890", "12345"]  # Duplicate 12345

        with patch.object(self.downloader, "download_pdf_to_temp", return_value=None):
            result = self.downloader.process_identifiers(identifiers)

        # Should only have unique entries
        self.assertEqual(len(result), 2)
        self.assertIn("12345", result)
        self.assertIn("67890", result)


class TestBasePaperDownloaderAbstractMethods(unittest.TestCase):
    """Test abstract method behavior."""

    def test_abstract_class_cannot_be_instantiated(self):
        """BasePaperDownloader should be abstract (non-instantiable)."""

        self.assertTrue(inspect.isabstract(BasePaperDownloader))

    def test_complete_implementation_succeeds(self):
        """Test that complete implementations work."""
        # ConcretePaperDownloader from setUp should work
        config = Mock()
        config.request_timeout = 30
        config.chunk_size = 8192

        downloader = ConcretePaperDownloader(config)

        # Should be able to call all methods
        self.assertEqual(downloader.get_service_name(), "TestService")
        self.assertEqual(downloader.get_identifier_name(), "Test ID")
        self.assertEqual(downloader.get_default_filename("test"), "test_test.pdf")
