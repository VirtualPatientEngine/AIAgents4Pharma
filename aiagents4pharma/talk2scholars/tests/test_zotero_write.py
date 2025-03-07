"""
Unit tests for Zotero write tool in zotero_write.py.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock
from langgraph.types import Command
from aiagents4pharma.talk2scholars.tools.zotero.zotero_write import (
    zotero_save_tool,
)

# Dummy Hydra configuration for the Zotero write tool
dummy_zotero_write_config = SimpleNamespace(
    user_id="dummy_user_write",
    library_type="user",
    api_key="dummy_api_key_write",
)
dummy_cfg = SimpleNamespace(
    tools=SimpleNamespace(zotero_write=dummy_zotero_write_config)
)


class TestZoteroSaveTool(unittest.TestCase):
    """a test class for the Zotero save tool"""

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_successful_save_direct_state(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test successful saving when the fetched papers are directly provided in the state.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        state = {
            "last_displayed_papers": {
                "paper1": {
                    "Title": "Test Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                },
                "paper2": {
                    "Title": "Test Paper 2",
                    "Abstract": "Abstract 2",
                    "Date": "2022",
                    "URL": "http://example2.com",
                    "Citations": "1",
                },
            },
            "zotero_read": {"paper1": ["/Test Collection"]},
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        fake_zot.collections.return_value = [
            {"key": "col1", "data": {"name": "Test Collection"}}
        ]
        fake_zot.create_items.return_value = {"success": True}
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_call_1"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/Test Collection",
            "state": state,
        }
        result = zotero_save_tool.run(tool_input)

        self.assertIsInstance(result, Command)
        messages = result.update.get("messages", [])
        self.assertTrue(len(messages) > 0)
        content = messages[0].content
        self.assertIn("Save was successful", content)
        self.assertIn("Test Collection", content)

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_successful_save_state_key(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test successful saving when the state's last_displayed_papers is a key referencing
        the actual fetched papers.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        state = {
            "last_displayed_papers": "papers_key",
            "papers_key": {
                "paper1": {
                    "Title": "Test Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                }
            },
            "zotero_read": {"paper1": ["/Test Collection"]},
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        fake_zot.collections.return_value = [
            {"key": "col1", "data": {"name": "Test Collection"}}
        ]
        fake_zot.create_items.return_value = {"success": True}
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_call_2"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/Test Collection",
            "state": state,
        }
        result = zotero_save_tool.run(tool_input)
        self.assertIsInstance(result, Command)
        messages = result.update.get("messages", [])
        self.assertTrue(len(messages) > 0)
        content = messages[0].content
        self.assertIn("Save was successful", content)
        self.assertIn("Test Collection", content)

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_no_fetched_papers(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test that a RuntimeError is raised when there are no fetched papers in the state.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_get_item_collections.return_value = {}
        mock_hydra_init.return_value.__enter__.return_value = None

        state = {
            "last_displayed_papers": {},
            "zotero_read": {"paper1": ["/Test Collection"]},
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        fake_zot.collections.return_value = [
            {"key": "col1", "data": {"name": "Test Collection"}}
        ]
        mock_zotero_class.return_value = fake_zot

        tool_call_id = "test_call_3"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/Test Collection",
            "state": state,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_save_tool.run(tool_input)
        self.assertIn("No fetched papers were found to save", str(context.exception))

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_fallback_get_item_collections(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test that if 'zotero_read' in the state is empty, the fallback
        using get_item_collections is used.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        state = {
            "last_displayed_papers": {
                "paper1": {
                    "Title": "Test Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                }
            },
            "zotero_read": {},  # empty mapping triggers fallback
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        fake_zot.collections.return_value = [
            {"key": "col1", "data": {"name": "Test Collection"}}
        ]
        fake_zot.create_items.return_value = {"success": True}
        mock_zotero_class.return_value = fake_zot

        # Simulate get_item_collections returning a mapping that includes a match.
        mock_get_item_collections.return_value = {"paper1": ["/Test Collection"]}

        tool_call_id = "test_call_4"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/Test Collection",
            "state": state,
        }
        result = zotero_save_tool.run(tool_input)
        messages = result.update.get("messages", [])
        self.assertTrue(len(messages) > 0)
        self.assertIn("Save was successful", messages[0].content)

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_invalid_collection_path(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test that a RuntimeError is raised when the provided collection
        path does not match any collection.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        state = {
            "last_displayed_papers": {
                "paper1": {
                    "Title": "Test Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                }
            },
            "zotero_read": {},  # empty mapping; no match available
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        fake_zot.collections.return_value = [
            {"key": "col1", "data": {"name": "Test Collection"}}
        ]
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_call_5"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/Nonexistent",
            "state": state,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_save_tool.run(tool_input)
        self.assertIn("does not exist in Zotero", str(context.exception))

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_save_failure(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test that if the Zotero client raises an exception during
        create_items, a RuntimeError is raised.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        state = {
            "last_displayed_papers": {
                "paper1": {
                    "Title": "Test Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                }
            },
            "zotero_read": {"paper1": ["/Test Collection"]},
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        fake_zot.collections.return_value = [
            {"key": "col1", "data": {"name": "Test Collection"}}
        ]
        fake_zot.create_items.side_effect = Exception("Creation error")
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_call_6"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/Test Collection",
            "state": state,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_save_tool.run(tool_input)
        self.assertIn("Error saving papers to Zotero", str(context.exception))

    # --- Additional tests to cover missing lines ---

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_get_item_collections_exception(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test that if get_item_collections raises an exception, the fallback branch
        raises a RuntimeError.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        # Provide valid fetched papers so we bypass earlier error
        state = {
            "last_displayed_papers": {
                "paper1": {
                    "Title": "Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                }
            },
            "zotero_read": {},  # empty so fallback is triggered
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        fake_zot.collections.return_value = [
            {"key": "col1", "data": {"name": "Test Collection"}}
        ]
        mock_zotero_class.return_value = fake_zot

        # Simulate exception in get_item_collections
        mock_get_item_collections.side_effect = Exception("Mapping error")

        tool_call_id = "test_call_7"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/Test Collection",
            "state": state,
        }
        with self.assertRaises(RuntimeError) as context:
            zotero_save_tool.run(tool_input)
        self.assertIn(
            "Failed to generate collection path mappings", str(context.exception)
        )

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_zotero_read_string_match(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test that if an entry in zotero_read is a string that matches the normalized path,
        it is used as the collection key
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        state = {
            "last_displayed_papers": {
                "paper1": {
                    "Title": "Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                }
            },
            # zotero_read entry is a string, not a list
            "zotero_read": {"match_key": "/test collection"},
            "query": "dummy query",
        }

        # Return a collection with key "match_key" to simulate a successful match.
        fake_zot = MagicMock()
        fake_zot.collections.return_value = [
            {"key": "match_key", "data": {"name": "Test Collection"}}
        ]
        fake_zot.create_items.return_value = {"success": True}
        mock_zotero_class.return_value = fake_zot
        # get_item_collections is not used in this branch.
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_call_8"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/test collection",
            "state": state,
        }
        result = zotero_save_tool.run(tool_input)
        messages = result.update.get("messages", [])
        self.assertTrue(len(messages) > 0)
        # Check for the correct substring in the returned message.
        self.assertIn(
            "Papers have been saved to Zotero collection", messages[0].content
        )

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_direct_match_by_collection_name(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test that if zotero_read does not yield a match, the tool finds
        a direct match by collection name
        in the collections list
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        state = {
            "last_displayed_papers": {
                "paper1": {
                    "Title": "Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                }
            },
            # Non-matching zotero_read data
            "zotero_read": {"dummy": ["/other"]},
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        # Collection with name "Test Collection" should match because
        # f"/Test Collection".lower() equals normalized path.
        fake_zot.collections.return_value = [
            {"key": "col1", "data": {"name": "Test Collection"}}
        ]
        fake_zot.create_items.return_value = {"success": True}
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_call_9"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/Test Collection",
            "state": state,
        }
        result = zotero_save_tool.run(tool_input)
        messages = result.update.get("messages", [])
        self.assertTrue(len(messages) > 0)
        self.assertIn("Test Collection", messages[0].content)

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_match_by_stripped_name(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test that if no direct match is found, a match is found by comparing
        the stripped collection path.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        # Provide state with non-matching zotero_read
        state = {
            "last_displayed_papers": {
                "paper1": {
                    "Title": "Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                }
            },
            "zotero_read": {"dummy": ["/other"]},
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        # Set collection_path without leading slash so that direct matching fails,
        # but normalized_path.lstrip("/") yields "test", which matches the collection name.
        fake_zot.collections.return_value = [{"key": "colX", "data": {"name": "test"}}]
        fake_zot.create_items.return_value = {"success": True}
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_call_10"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "test",  # no leading slash
            "state": state,
        }
        result = zotero_save_tool.run(tool_input)
        messages = result.update.get("messages", [])
        self.assertTrue(len(messages) > 0)
        self.assertIn("test", messages[0].content)

    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.initialize")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.hydra.compose")
    @patch("aiagents4pharma.talk2scholars.tools.zotero.zotero_write.zotero.Zotero")
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.zotero_write.get_item_collections"
    )
    def test_match_by_path_component(
        self,
        mock_get_item_collections,
        mock_zotero_class,
        mock_hydra_compose,
        mock_hydra_init,
    ):
        """
        Test that if no full-string match is found, the tool can match
        by one of the path components.
        """
        mock_hydra_compose.return_value = dummy_cfg
        mock_hydra_init.return_value.__enter__.return_value = None

        state = {
            "last_displayed_papers": {
                "paper1": {
                    "Title": "Paper 1",
                    "Abstract": "Abstract 1",
                    "Date": "2021",
                    "URL": "http://example.com",
                    "Citations": "0",
                }
            },
            "zotero_read": {"dummy": ["/other"]},
            "query": "dummy query",
        }

        fake_zot = MagicMock()
        # Collection name "bar" should be found via a path component when
        # collection_path is "/foo/bar"
        fake_zot.collections.return_value = [{"key": "colBar", "data": {"name": "bar"}}]
        fake_zot.create_items.return_value = {"success": True}
        mock_zotero_class.return_value = fake_zot
        mock_get_item_collections.return_value = {}

        tool_call_id = "test_call_11"
        tool_input = {
            "tool_call_id": tool_call_id,
            "collection_path": "/foo/bar",
            "state": state,
        }
        result = zotero_save_tool.run(tool_input)
        messages = result.update.get("messages", [])
        self.assertTrue(len(messages) > 0)
        self.assertIn("bar", messages[0].content)
