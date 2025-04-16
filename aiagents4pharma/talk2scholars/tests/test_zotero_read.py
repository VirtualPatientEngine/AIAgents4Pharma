"""
Unit tests for Zotero search tool in zotero_read.py.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
from langgraph.types import Command
from aiagents4pharma.talk2scholars.tools.zotero.zotero_read import zotero_read
from aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper import (
    ZoteroSearchData,
)

# pylint: disable=protected-access
# pylint: disable=protected-access, too-many-arguments, too-many-positional-arguments

# Dummy Hydra configuration to be used in tests
dummy_zotero_read_config = SimpleNamespace(
    user_id="dummy_user",
    library_type="user",
    api_key="dummy_api_key",
    zotero=SimpleNamespace(
        max_limit=5,
        filter_item_types=["journalArticle", "conferencePaper"],
        filter_excluded_types=["attachment", "note"],
    ),
)
dummy_cfg = SimpleNamespace(tools=SimpleNamespace(zotero_read=dummy_zotero_read_config))


class TestZoteroSearchTool(unittest.TestCase):
    """Tests for Zotero search tool."""

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_valid_query(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test valid query returns correct Command output."""
        # Setup Hydra mocks
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        # Create a fake Zotero client that returns two valid items
        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                }
            },
            {
                "data": {
                    "key": "paper2",
                    "title": "Paper 2",
                    "abstractNote": "Abstract 2",
                    "date": "2022",
                    "url": "http://example2.com",
                    "itemType": "conferencePaper",
                }
            },
        ]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot

        # Fake mapping for collection paths
        mock_get_item_collections.return_value = {
            "paper1": ["/Test Collection"],
            "paper2": ["/Test Collection"],
        }

        # Call the tool with a valid query using .run() with a dictionary input
        tool_call_id = "test_id_1"
        tool_input = {
            "query": "test",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        result = zotero_read.run(tool_input)

        # Verify the Command update structure and contents
        self.assertIsInstance(result, Command)
        update = result.update
        self.assertIn("article_data", update)
        self.assertIn("last_displayed_papers", update)
        self.assertIn("messages", update)

        filtered_papers = update["article_data"]
        self.assertIn("paper1", filtered_papers)
        self.assertIn("paper2", filtered_papers)
        message_content = update["messages"][0].content
        self.assertIn("Query: test", message_content)
        self.assertIn("Number of papers found: 2", message_content)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_empty_query_fetch_all_items(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test empty query fetches all items."""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                }
            },
        ]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper1": ["/Test Collection"]}

        tool_call_id = "test_id_2"
        tool_input = {
            "query": "  ",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        result = zotero_read.run(tool_input)

        update = result.update
        filtered_papers = update["article_data"]
        self.assertIn("paper1", filtered_papers)
        fake_zot.items.assert_called_with(
            limit=dummy_cfg.tools.zotero_read.zotero.max_limit
        )

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_no_items_returned(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test no items returned from Zotero."""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_zot.items.return_value = []
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_id_3"
        tool_input = {
            "query": "nonexistent",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_read.run(tool_input)
        self.assertIn("No items returned from Zotero", str(context.exception))

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper."
        "ZoteroSearchData._download_pdfs_in_parallel"
    )
    def test_filtering_no_matching_papers(
        self,
        mock_batch_download,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Testing filtering when no paper matching"""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "attachment",
                    "contentType": "application/pdf",  # orphaned
                    "filename": "paper1.pdf",
                }
            },
            {
                "data": {
                    "key": "paper2",
                    "title": "Paper 2",
                    "abstractNote": "Abstract 2",
                    "date": "2022",
                    "url": "http://example2.com",
                    "itemType": "note",
                }
            },
        ]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {
            "paper1": ["/Test Collection"],
            "paper2": ["/Test Collection"],
        }

        mock_batch_download.return_value = {
            "paper1": ("/tmp/fake_path.pdf", "paper1.pdf", "paper1")
        }

        tool_input = {
            "query": "test",
            "only_articles": False,
            "tool_call_id": "test_id_4",
            "limit": 2,
        }

        result = zotero_read.run(tool_input)
        filtered_papers = result.update["article_data"]

        self.assertIn("paper1", filtered_papers)
        self.assertIn("paper2", filtered_papers)
        self.assertEqual(filtered_papers["paper1"]["filename"], "paper1.pdf")
        self.assertEqual(filtered_papers["paper1"]["pdf_url"], "/tmp/fake_path.pdf")
        self.assertEqual(filtered_papers["paper1"]["source"], "zotero")

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_items_api_exception(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test items API exception is properly raised."""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None
        mock_get_item_collections.return_value = {}

        fake_zot = MagicMock()
        fake_zot.items.side_effect = Exception("API error")
        mock_zotero_class.return_value = fake_zot

        tool_call_id = "test_id_5"
        tool_input = {
            "query": "test",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_read.run(tool_input)
        self.assertIn("Failed to fetch items from Zotero", str(context.exception))

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_missing_key_in_item(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """
        Test that an item with a valid 'data' structure but missing the 'key' field is skipped.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "title": "No Key Paper",
                    "abstractNote": "Abstract",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                }
            },  # Missing 'key' field
            {
                "data": {
                    "key": "paper_valid",
                    "title": "Valid Paper",
                    "abstractNote": "Valid Abstract",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                }
            },
        ]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper_valid": ["/Test Collection"]}

        tool_call_id = "test_id_6"
        tool_input = {
            "query": "dummy",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        result = zotero_read.run(tool_input)

        update = result.update
        filtered_papers = update["article_data"]
        self.assertIn("paper_valid", filtered_papers)
        self.assertEqual(len(filtered_papers), 1)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_item_not_dict(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """
        Test that if the items list contains an element that is not a dict, it is skipped.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        # Supply one item that is not a dict.
        fake_items = ["this is not a dict"]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        # Mapping doesn't matter here.
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_id_7"
        tool_input = {
            "query": "dummy",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 2,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_read.run(tool_input)
        self.assertIn("No matching papers returned from Zotero", str(context.exception))

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_data_not_dict(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test for no dict"""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        # Make the item itself non-dict (not just `data`)
        fake_items = ["this is not a dict"]
        fake_zot.items.return_value = fake_items
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_input = {
            "query": "dummy",
            "only_articles": True,
            "tool_call_id": "test_id_8",
            "limit": 2,
        }

        with self.assertRaises(RuntimeError) as context:
            zotero_read.run(tool_input)
        self.assertIn("No matching papers returned from Zotero", str(context.exception))

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.requests.Session.get"
    )
    def test_pdf_attachment_success(
        self,
        mock_session_get,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test for pdf attachment success"""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_zot.items.return_value = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                    "creators": [
                        {
                            "firstName": "John",
                            "lastName": "Doe",
                            "creatorType": "author",
                        }
                    ],
                }
            }
        ]

        fake_pdf_child = {
            "data": {
                "key": "attachment1",
                "filename": "file1.pdf",
                "contentType": "application/pdf",
            }
        }
        fake_zot.children.return_value = [fake_pdf_child]
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper1": ["/Test Collection"]}

        # Mock successful PDF download via session
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = lambda chunk_size: [b"fake pdf content"]
        mock_response.headers = {
            "Content-Disposition": 'attachment; filename="file1.pdf"'
        }
        mock_response.raise_for_status = lambda: None
        mock_session_get.return_value = mock_response

        tool_input = {
            "query": "pdf test",
            "only_articles": True,
            "tool_call_id": "test_pdf_success",
            "limit": 1,
        }

        result = zotero_read.run(tool_input)
        paper = result.update["article_data"]["paper1"]

        self.assertIn("pdf_url", paper)
        self.assertTrue(paper["pdf_url"].endswith(".pdf"))
        self.assertEqual(paper["filename"], "file1.pdf")
        self.assertEqual(paper["attachment_key"], "attachment1")

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_pdf_attachment_children_exception(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test that when children() raises an exception, PDF info is not added."""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                    "creators": [
                        {
                            "firstName": "John",
                            "lastName": "Doe",
                            "creatorType": "author",
                        }
                    ],
                }
            },
        ]
        fake_zot.items.return_value = fake_items

        # Simulate children() raising an exception
        fake_zot.children.side_effect = Exception("Child fetch error")
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper1": ["/Test Collection"]}

        tool_call_id = "test_pdf_children_exception"
        tool_input = {
            "query": "pdf test exception",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 1,
        }
        result = zotero_read.run(tool_input)
        filtered_papers = result.update["article_data"]

        # Ensure no PDF-related keys are added
        self.assertIn("paper1", filtered_papers)
        self.assertNotIn("pdf_url", filtered_papers["paper1"])
        self.assertNotIn("filename", filtered_papers["paper1"])
        self.assertNotIn("attachment_key", filtered_papers["paper1"])

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_pdf_attachment_missing_key(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test for pdf attachment missing"""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_zot.items.return_value = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                    "creators": [
                        {
                            "firstName": "Alice",
                            "lastName": "Smith",
                            "creatorType": "author",
                        }
                    ],
                }
            },
        ]

        fake_pdf_child = {
            "data": {
                "filename": "no_key.pdf",
                "contentType": "application/pdf",
            }
        }
        fake_zot.children.return_value = [fake_pdf_child]
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper1": ["/Test Collection"]}

        tool_input = {
            "query": "missing key test",
            "only_articles": True,
            "tool_call_id": "test_pdf_missing_key",
            "limit": 1,
        }

        result = zotero_read.run(tool_input)
        paper = result.update["article_data"]["paper1"]

        self.assertNotIn("pdf_url", paper)
        self.assertNotIn("attachment_key", paper)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_pdf_attachment_outer_exception(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test that if children() returns a non-iterable (causing an exception),
        PDF info is not added."""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_items = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Paper 1",
                    "abstractNote": "Abstract 1",
                    "date": "2021",
                    "url": "http://example.com",
                    "itemType": "journalArticle",
                    "creators": [
                        {
                            "firstName": "Bob",
                            "lastName": "Jones",
                            "creatorType": "author",
                        }
                    ],
                }
            },
        ]
        fake_zot.items.return_value = fake_items

        # Simulate children() returning None to trigger an exception in list comprehension.
        fake_zot.children.return_value = None

        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper1": ["/Test Collection"]}

        tool_call_id = "test_pdf_outer_exception"
        tool_input = {
            "query": "outer exception test",
            "only_articles": True,
            "tool_call_id": tool_call_id,
            "limit": 1,
        }
        result = zotero_read.run(tool_input)
        filtered_papers = result.update["article_data"]

        # Ensure no PDF-related keys are added if an exception occurs
        self.assertIn("paper1", filtered_papers)
        self.assertNotIn("pdf_url", filtered_papers["paper1"])
        self.assertNotIn("filename", filtered_papers["paper1"])
        self.assertNotIn("attachment_key", filtered_papers["paper1"])

    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.requests.get")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_download_zotero_pdf_exception(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
        mock_requests_get,
    ):
        """Test that _download_zotero_pdf returns None and logs error on request exception."""
        # Setup mocks for config and Zotero client
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None
        mock_zotero_class.return_value = MagicMock()
        mock_get_item_collections.return_value = {}

        # Simulate a request exception during PDF download
        mock_requests_get.side_effect = Exception("Simulated download failure")

        zotero_search = ZoteroSearchData(
            query="test", only_articles=False, limit=1, tool_call_id="test123"
        )

        result = zotero_search._download_zotero_pdf("FAKE_ATTACHMENT_KEY")

        self.assertIsNone(result)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.zotero.Zotero")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.compose")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper.hydra.initialize"
    )
    def test_download_pdf_exception_logging(
        self,
        mock_hydra_init,
        mock_hydra_compose,
        mock_zotero_class,
        mock_get_item_collections,
    ):
        """Test that a failed download logs the error and does not break the pipeline."""
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        fake_zot = MagicMock()
        fake_zot.items.return_value = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Fake Title",
                    "itemType": "journalArticle",
                }
            }
        ]
        # Simulate an attachment
        fake_zot.children.return_value = [
            {
                "data": {
                    "key": "attachment1",
                    "filename": "file1.pdf",
                    "contentType": "application/pdf",
                }
            }
        ]

        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {"paper1": ["/Fake Collection"]}

        # Patch just the internal _download_zotero_pdf to raise an exception
        with patch(
            "aiagents4pharma.talk2scholars.tools.zotero.utils.read_helper."
            "ZoteroSearchData._download_zotero_pdf"
        ) as mock_download_pdf:
            mock_download_pdf.side_effect = Exception("Simulated download error")

            search = ZoteroSearchData(
                query="failure test",
                only_articles=True,
                limit=1,
                tool_call_id="fail_test",
            )
            search.process_search()

            article_data = search.get_search_results()["article_data"]
            assert "paper1" in article_data
            assert "pdf_url" not in article_data["paper1"]  # download failed, no URL
