#!/usr/bin/env python3

"""
Utility functions for Zotero tools.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_item_collections(zot):
    """
    Fetch all Zotero collections and map item keys to their full collection paths.

    Args:
        zot (Zotero): An initialized Zotero client.

    Returns:
        dict: A dictionary mapping item keys to a list of full collection paths.
    """
    logger.info("Fetching Zotero collections...")

    # Fetch all collections
    collections = zot.collections()

    # Create mappings: collection key → name and collection key → parent key
    collection_map = {col["key"]: col["data"]["name"] for col in collections}
    parent_map = {
        col["key"]: col["data"].get("parentCollection") for col in collections
    }

    # Build full paths for collections
    def build_collection_path(col_key):
        path = []
        while col_key:
            path.insert(0, collection_map.get(col_key, "Unknown"))
            col_key = parent_map.get(col_key)
        return "/" + "/".join(path)  # Convert to "/path/to/collection"

    collection_paths = {key: build_collection_path(key) for key in collection_map}

    # Manually create an item-to-collection mapping with full paths
    item_to_collections = {}

    for collection in collections:
        collection_key = collection["key"]
        collection_items = zot.collection_items(
            collection_key
        )  # Fetch items in the collection

        for item in collection_items:
            item_key = item["data"]["key"]
            if item_key in item_to_collections:
                item_to_collections[item_key].append(collection_paths[collection_key])
            else:
                item_to_collections[item_key] = [collection_paths[collection_key]]

    logger.info("Successfully mapped items to collection paths.")

    return item_to_collections
