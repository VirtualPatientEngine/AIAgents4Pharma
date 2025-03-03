#!/usr/bin/env python3

"""
Utility functions for Zotero tools.
"""

import logging
from pyzotero import zotero

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_item_collections(zot: zotero.Zotero):
    """
    Fetch all Zotero collections and map item keys to their collection names.

    Args:
        zot (Zotero): An initialized Zotero client.

    Returns:
        dict: A dictionary mapping item keys to a list of collection names.
    """
    logger.info("Fetching Zotero collections...")

    # Fetch all collections
    collections = zot.collections()
    collection_map = {col["key"]: col["data"]["name"] for col in collections}

    # Manually create an item-to-collection mapping
    item_to_collections = {}

    for collection in collections:
        collection_key = collection["key"]
        collection_items = zot.collection_items(
            collection_key
        )  # Fetch items in the collection

        for item in collection_items:
            item_key = item["data"]["key"]
            if item_key in item_to_collections:
                item_to_collections[item_key].append(collection_map[collection_key])
            else:
                item_to_collections[item_key] = [collection_map[collection_key]]

    logger.info("Successfully mapped items to collections.")

    return item_to_collections
