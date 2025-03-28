#!/usr/bin/env python3

"""
Utility functions for Zotero path operations.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# pylint: disable=broad-exception-caught


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
        """build collection path from collection key"""
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


def find_or_create_collection(zot, path, create_missing=False):
    """find collection or create if missing"""
    logger.info(
        "Finding collection for path: %s (create_missing=%s)", path, create_missing
    )
    # Normalize path: remove leading/trailing slashes and convert to lowercase
    normalized = path.strip("/").lower()
    path_parts = normalized.split("/") if normalized else []

    if not path_parts:
        logger.warning("Empty path provided")
        return None

    # Get all collections from Zotero
    all_collections = zot.collections()
    logger.info("Found %d collections in Zotero", len(all_collections))

    # Determine target name (last part) and, if nested, find the parent's key
    target_name = path_parts[-1]
    parent_key = None
    if len(path_parts) > 1:
        parent_name = path_parts[-2]
        # Look for a collection with name matching the parent (case-insensitive)
        for col in all_collections:
            if col["data"]["name"].lower() == parent_name:
                parent_key = col["key"]
                break

    # Try to find an existing collection by direct match (ignoring hierarchy)
    for col in all_collections:
        if col["data"]["name"].lower() == target_name:
            logger.info("Found direct match for %s: %s", target_name, col["key"])
            return col["key"]

    # No match found: create one if allowed
    if create_missing:
        payload = {"name": target_name}
        if parent_key:
            payload["parentCollection"] = parent_key
        try:
            result = zot.create_collection(payload)
            # Interpret result based on structure
            if "success" in result:
                new_key = result["success"]["0"]
            else:
                new_key = result["successful"]["0"]["data"]["key"]
            logger.info("Created collection %s with key %s", target_name, new_key)
            return new_key
        except Exception as e:
            logger.error("Failed to create collection: %s", e)
            return None
    else:
        logger.warning("No matching collection found for %s", target_name)
        return None


def get_all_collection_paths(zot):
    """
    Get all available collection paths in Zotero.

    Args:
        zot (Zotero): An initialized Zotero client.

    Returns:
        list: List of all available collection paths
    """
    logger.info("Getting all collection paths")
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
        return "/" + "/".join(path)

    collection_paths = [build_collection_path(key) for key in collection_map]
    logger.info("Found %d collection paths", len(collection_paths))
    return collection_paths


def fetch_papers_for_save(state):
    """
    Retrieve papers from the state for saving to Zotero.

    Args:
        state (dict): The state containing previously fetched papers.

    Returns:
        dict: Dictionary of papers to save, or None if no papers found
    """
    logger.info("Fetching papers from state for saving")

    # Retrieve last displayed papers from the agent state
    last_displayed_key = state.get("last_displayed_papers", "")

    if not last_displayed_key:
        logger.warning("No last_displayed_papers key in state")
        return None

    if isinstance(last_displayed_key, str):
        # If it's a string (key to another state object), get that object
        fetched_papers = state.get(last_displayed_key, {})
        logger.info("Using papers from '%s' state key", last_displayed_key)
    else:
        # If it's already the papers object
        fetched_papers = last_displayed_key
        logger.info("Using papers directly from last_displayed_papers")

    if not fetched_papers:
        logger.warning("No fetched papers found to save.")
        return None

    return fetched_papers
