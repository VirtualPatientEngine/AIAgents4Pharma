"""
Unit tests for Zotero path utility in zotero_path.py.
"""

import unittest
from unittest.mock import MagicMock
from aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path import (
    get_item_collections,
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
