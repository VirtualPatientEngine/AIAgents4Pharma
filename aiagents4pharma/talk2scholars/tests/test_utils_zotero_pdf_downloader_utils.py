"""
Unit tests for Zotero PDF downloader utilities.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import requests

from aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_pdf_downloader import (
    download_pdfs_in_parallel,
    download_zotero_pdf,
)


class TestZoteroPDFDownloaderUtils(unittest.TestCase):
    """Tests for zotero_pdf_downloader module."""

    @patch("requests.Session.get")
    def test_download_zotero_pdf_default_filename(self, mock_get):
        """Test download_zotero_pdf returns default filename when header has no filename."""
        # Mock response without Content-Disposition filename
        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.iter_content = lambda chunk_size: [b"fakepdf"]
        mock_response.headers = {}
        mock_get.return_value = mock_response

        session = requests.Session()
        result = download_zotero_pdf(session, "user123", "apikey", "attach123")
        # Should return a tuple (file_path, filename)
        self.assertIsNotNone(result)
        file_path, filename = result
        # File should exist
        self.assertTrue(os.path.isfile(file_path))
        # Filename should default to 'downloaded.pdf'
        self.assertEqual(filename, "downloaded.pdf")
        # Clean up temp file
        os.remove(file_path)

    def test_download_pdfs_in_parallel_empty(self):
        """Test that download_pdfs_in_parallel returns empty dict on empty input."""
        session = requests.Session()
        result = download_pdfs_in_parallel(session, "user123", "apikey", {})
        self.assertEqual(result, {})
