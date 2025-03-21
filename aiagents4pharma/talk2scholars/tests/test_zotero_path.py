"""
Unit tests for Zotero path utility in zotero_path.py.
"""

import unittest
from unittest.mock import MagicMock
from aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path import (
    get_item_collections,
)
from aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path import (
    find_or_create_collection,
)


class TestGetItemCollections(unittest.TestCase):
    """Unit tests for the get_item_collections function."""

    def test_basic_collections_mapping(self):
        """test_basic_collections_mapping"""
        # Define fake collections with one parent-child relationship and one independent collection.
        fake_collections = [
            {"key": "A", "data": {"name": "Parent", "parentCollection": None}},
            {"key": "B", "data": {"name": "Child", "parentCollection": "A"}},
            {"key": "C", "data": {"name": "Independent", "parentCollection": None}},
        ]
        # Define fake collection items for each collection:
        # - Collection A returns one item with key "item1"
        # - Collection B returns one item with key "item2"
        # - Collection C returns two items: one duplicate ("item1") and one new ("item3")
        fake_collection_items = {
            "A": [{"data": {"key": "item1"}}],
            "B": [{"data": {"key": "item2"}}],
            "C": [{"data": {"key": "item1"}}, {"data": {"key": "item3"}}],
        }
        fake_zot = MagicMock()
        fake_zot.collections.return_value = fake_collections

        # When collection_items is called, return the appropriate list based on collection key.
        def fake_collection_items_func(collection_key):
            return fake_collection_items.get(collection_key, [])

        fake_zot.collection_items.side_effect = fake_collection_items_func

        # Expected full collection paths:
        # - Collection A: "/Parent"
        # - Collection B: "/Parent/Child"   (child of A)
        # - Collection C: "/Independent"
        #
        # Expected mapping for items:
        # - "item1" appears in collections A and C → ["/Parent", "/Independent"]
        # - "item2" appears in B → ["/Parent/Child"]
        # - "item3" appears in C → ["/Independent"]
        expected_mapping = {
            "item1": ["/Parent", "/Independent"],
            "item2": ["/Parent/Child"],
            "item3": ["/Independent"],
        }

        result = get_item_collections(fake_zot)
        self.assertEqual(result, expected_mapping)


class TestFindOrCreateCollectionExtra(unittest.TestCase):
    def setUp(self):
        # Set up a fake Zotero client with some default collections.
        self.fake_zot = MagicMock()
        self.fake_zot.collections.return_value = [
            {"key": "parent1", "data": {"name": "Parent", "parentCollection": None}},
            {"key": "child1", "data": {"name": "Child", "parentCollection": "parent1"}},
        ]

    def test_empty_path(self):
        """Test that an empty path returns None."""
        result = find_or_create_collection(self.fake_zot, "", create_missing=False)
        self.assertIsNone(result)

    def test_create_collection_with_success_key(self):
        """
        Test that when create_missing is True and the response contains a "success" key,
        the function returns the new collection key.
        """
        # Simulate no existing collections (so direct match fails)
        self.fake_zot.collections.return_value = []
        # Simulate create_collection returning a dict with a "success" key.
        self.fake_zot.create_collection.return_value = {
            "success": {"0": "new_key_success"}
        }
        result = find_or_create_collection(
            self.fake_zot, "/NewCollection", create_missing=True
        )
        self.assertEqual(result, "new_key_success")
        # Verify payload formatting: for a simple (non-nested) path, no parentCollection.
        args, _ = self.fake_zot.create_collection.call_args
        payload = args[0]
        self.assertEqual(payload["name"], "newcollection")
        self.assertNotIn("parentCollection", payload)

    def test_create_collection_with_successful_key(self):
        """
        Test that when create_missing is True and the response contains a "successful" key,
        the function returns the new collection key.
        """
        self.fake_zot.collections.return_value = []
        self.fake_zot.create_collection.return_value = {
            "successful": {"0": {"data": {"key": "new_key_successful"}}}
        }
        result = find_or_create_collection(
            self.fake_zot, "/NewCollection", create_missing=True
        )
        self.assertEqual(result, "new_key_successful")

    def test_create_collection_exception(self):
        """
        Test that if create_collection raises an exception, the function logs the error and returns None.
        """
        self.fake_zot.collections.return_value = []
        self.fake_zot.create_collection.side_effect = Exception("Creation error")
        result = find_or_create_collection(
            self.fake_zot, "/NewCollection", create_missing=True
        )
        self.assertIsNone(result)
