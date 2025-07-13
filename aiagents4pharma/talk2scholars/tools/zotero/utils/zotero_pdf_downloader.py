#!/usr/bin/env python3
"""
Utility functions for downloading PDFs from Zotero.
"""

import logging
import tempfile
from typing import Optional, Tuple, Dict
import concurrent.futures
import requests

logger = logging.getLogger(__name__)


def download_zotero_pdf(
    session: requests.Session,
    user_id: str,
    api_key: str,
    attachment_key: str,
    **kwargs,
) -> Optional[Tuple[str, str]]:
    """
    Download a PDF from Zotero by attachment key.

    Args:
        session: requests.Session for HTTP requests.
        user_id: Zotero user ID.
        api_key: Zotero API key.
        attachment_key: Zotero attachment item key.
        kwargs:
            timeout (int): Request timeout in seconds (default: 10).
            chunk_size (int, optional): Chunk size for streaming.

    Returns:
        Tuple of (local_file_path, filename) if successful, else None.
    """
    # Extract optional parameters
    timeout = kwargs.get("timeout", 10)
    chunk_size = kwargs.get("chunk_size")
    # Log configured parameters for verification
    logger.info("download_zotero_pdf params -> timeout=%s, chunk_size=%s", timeout, chunk_size)
    # Log download start
    logger.info(
        "Downloading Zotero PDF for attachment %s from Zotero API", attachment_key
    )
    zotero_pdf_url = (
        f"https://api.zotero.org/users/{user_id}/items/" f"{attachment_key}/file"
    )
    headers = {"Zotero-API-Key": api_key}

    try:
        response = session.get(
            zotero_pdf_url, headers=headers, stream=True, timeout=timeout
        )
        response.raise_for_status()

        # Download to a temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        # Temp file written to %s
        logger.info("Zotero PDF downloaded to temporary file: %s", temp_file_path)

        # Determine filename from Content-Disposition header or default
        if "filename=" in response.headers.get("Content-Disposition", ""):
            filename = (
                response.headers.get("Content-Disposition", "")
                .split("filename=")[-1]
                .strip('"')
            )
        else:
            filename = "downloaded.pdf"

        return temp_file_path, filename

    except (requests.exceptions.RequestException, OSError) as e:
        logger.error(
            "Failed to download Zotero PDF for attachment %s: %s", attachment_key, e
        )
        return None


def download_pdfs_in_parallel(
    session: requests.Session,
    user_id: str,
    api_key: str,
    attachment_item_map: Dict[str, str],
    **kwargs,
) -> Dict[str, Tuple[str, str, str]]:
    """
    Download multiple PDFs in parallel using ThreadPoolExecutor.

    Args:
        session: requests.Session for HTTP requests.
        user_id: Zotero user ID.
        api_key: Zotero API key.
        attachment_item_map: Mapping of attachment_key to parent item_key.
        kwargs:
            max_workers (int, optional): Maximum number of worker threads (default: min(10, n)).
            chunk_size (int, optional): Chunk size for streaming.

    Returns:
        Mapping of parent item_key to (local_file_path, filename, attachment_key).
    """
    # Extract optional parameters
    max_workers = kwargs.get("max_workers")
    chunk_size = kwargs.get("chunk_size")
    # Log configured parameters for verification
    logger.info(
        "download_pdfs_in_parallel params -> max_workers=%s, chunk_size=%s", 
        max_workers,
        chunk_size,
    )
    results: Dict[str, Tuple[str, str, str]] = {}
    if not attachment_item_map:
        return results

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=(
            max_workers
            if max_workers is not None
            else min(10, len(attachment_item_map))
        )
    ) as executor:
        future_to_keys = {
            executor.submit(
                download_zotero_pdf,
                session,
                user_id,
                api_key,
                attachment_key,
                chunk_size=chunk_size,
            ): (attachment_key, item_key)
            for attachment_key, item_key in attachment_item_map.items()
        }

        for future in concurrent.futures.as_completed(future_to_keys):
            attachment_key, item_key = future_to_keys[future]
            try:
                res = future.result()
                if res:
                    results[item_key] = (*res, attachment_key)
            except (requests.exceptions.RequestException, OSError) as e:
                logger.error("Failed to download PDF for key %s: %s", attachment_key, e)

    return results
