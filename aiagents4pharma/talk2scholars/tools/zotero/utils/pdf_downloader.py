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
    timeout: int = 10,
) -> Optional[Tuple[str, str]]:
    """
    Download a PDF from Zotero by attachment key.

    Args:
        session: requests.Session for HTTP requests.
        user_id: Zotero user ID.
        api_key: Zotero API key.
        attachment_key: Zotero attachment item key.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (local_file_path, filename) if successful, else None.
    """
    zotero_pdf_url = (
        f"https://api.zotero.org/users/{user_id}/items/"
        f"{attachment_key}/file"
    )
    headers = {"Zotero-API-Key": api_key}

    try:
        response = session.get(
            zotero_pdf_url, headers=headers, stream=True, timeout=timeout
        )
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            for chunk in response.iter_content(chunk_size=16384):
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        content_disp = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip('"')
        else:
            filename = "downloaded.pdf"

        return temp_file_path, filename

    except Exception as e:
        logger.error(
            "Failed to download Zotero PDF for attachment %s: %s", attachment_key, e
        )
        return None


def download_pdfs_in_parallel(
    session: requests.Session,
    user_id: str,
    api_key: str,
    attachment_item_map: Dict[str, str],
    max_workers: Optional[int] = None,
) -> Dict[str, Tuple[str, str, str]]:
    """
    Download multiple PDFs in parallel using ThreadPoolExecutor.

    Args:
        session: requests.Session for HTTP requests.
        user_id: Zotero user ID.
        api_key: Zotero API key.
        attachment_item_map: Mapping of attachment_key to parent item_key.
        max_workers: Maximum number of worker threads (default: min(10, n)).

    Returns:
        Mapping of parent item_key to (local_file_path, filename, attachment_key).
    """
    results: Dict[str, Tuple[str, str, str]] = {}
    if not attachment_item_map:
        return results

    workers = min(10, len(attachment_item_map)) if max_workers is None else max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_keys = {
            executor.submit(
                download_zotero_pdf, session, user_id, api_key, attachment_key
            ): (attachment_key, item_key)
            for attachment_key, item_key in attachment_item_map.items()
        }

        for future in concurrent.futures.as_completed(future_to_keys):
            attachment_key, item_key = future_to_keys[future]
            try:
                result = future.result()
                if result:
                    temp_file_path, filename = result
                    results[item_key] = (temp_file_path, filename, attachment_key)
            except Exception as e:
                logger.error(
                    "Failed to download PDF for key %s: %s", attachment_key, e
                )

    return results