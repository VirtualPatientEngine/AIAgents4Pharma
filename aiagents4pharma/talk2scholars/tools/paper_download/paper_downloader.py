#!/usr/bin/env python3
"""
Unified paper download tool for LangGraph.
Supports downloading papers from arXiv, medRxiv, bioRxiv, and PubMed through a single interface.
"""

import logging
import threading
from typing import Annotated, Any, Literal

import hydra
from hydra.core.global_hydra import GlobalHydra
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.types import Command
from omegaconf import OmegaConf
from pydantic import BaseModel, Field

from .utils.arxiv_downloader import ArxivDownloader
from .utils.base_paper_downloader import BasePaperDownloader
from .utils.biorxiv_downloader import BiorxivDownloader
from .utils.medrxiv_downloader import MedrxivDownloader
from .utils.pubmed_downloader import PubmedDownloader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedPaperDownloadInput(BaseModel):
    """Input schema for the unified paper download tool."""

    service: Literal["arxiv", "medrxiv", "biorxiv", "pubmed"] | None = Field(
        default=None,
        description=(
            "Paper service to download from: 'arxiv', 'medrxiv', 'biorxiv', or 'pubmed'. "
            "If not specified, uses the configured default service."
        ),
    )
    identifiers: list[str] = Field(
        description=(
            "List of paper identifiers. Format depends on service:\n"
            "- arxiv: arXiv IDs (e.g., ['1234.5678', '2301.12345'])\n"
            "- medrxiv: DOIs (e.g., ['10.1101/2020.09.09.20191205'])\n"
            "- biorxiv: DOIs (e.g., ['10.1101/2020.09.09.20191205'])\n"
            "- pubmed: PMIDs (e.g., ['12345678', '87654321'])"
        )
    )
    tool_call_id: Annotated[str, InjectedToolCallId]


class PaperDownloaderFactory:
    """Factory class for creating paper downloader instances."""

    # Class-level cache for configuration
    _cached_config = None
    _config_lock = None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached configuration."""
        cls._cached_config = None

    @staticmethod
    def get_default_service() -> Literal["arxiv", "medrxiv", "biorxiv", "pubmed"]:
        """
        Get the default service from configuration.

        Returns:
            Default service name from config, fallback to 'pubmed'
        """
        config = PaperDownloaderFactory._get_unified_config()
        default_service = getattr(config.tool, "default_service", "pubmed")
        # Ensure the default service is valid and return with proper type
        if default_service == "arxiv":
            return "arxiv"
        if default_service == "medrxiv":
            return "medrxiv"
        if default_service == "biorxiv":
            return "biorxiv"
        if default_service == "pubmed":
            return "pubmed"
        logger.warning(
            "Invalid default service '%s' in config, falling back to 'pubmed'",
            default_service,
        )
        return "pubmed"

    @staticmethod
    def create(
        service: Literal["arxiv", "medrxiv", "biorxiv", "pubmed"],
    ) -> BasePaperDownloader:
        """
        Create appropriate downloader instance for the specified service.

        Args:
            service: Service name ('arxiv', 'medrxiv', 'biorxiv', 'pubmed')

        Returns:
            Configured downloader instance

        Raises:
            ValueError: If service is not supported
        """
        config = PaperDownloaderFactory._get_unified_config()
        service_config = PaperDownloaderFactory._build_service_config(config, service)

        if service == "arxiv":
            return ArxivDownloader(service_config)
        if service == "medrxiv":
            return MedrxivDownloader(service_config)
        if service == "biorxiv":
            return BiorxivDownloader(service_config)
        # service == "pubmed"
        return PubmedDownloader(service_config)

    @staticmethod
    def _get_unified_config() -> Any:
        """
        Load unified paper download configuration using Hydra with caching.
        This avoids the GlobalHydra reinitialization issue by caching the config.

        Returns:
            Unified configuration object
        """
        # Return cached config if available
        if PaperDownloaderFactory._cached_config is not None:
            return PaperDownloaderFactory._cached_config

        # Ensure lock exists and get a local reference
        lock = PaperDownloaderFactory._config_lock
        if lock is None:
            lock = threading.Lock()
            PaperDownloaderFactory._config_lock = lock

        # Thread-safe config loading with guaranteed non-None lock
        with lock:
            # Double-check pattern - another thread might have loaded it
            if PaperDownloaderFactory._cached_config is not None:
                return PaperDownloaderFactory._cached_config

            try:
                # Clear if already initialized
                if GlobalHydra().is_initialized():
                    logger.info("GlobalHydra already initialized, clearing for config load")
                    GlobalHydra.instance().clear()

                # Load configuration
                with hydra.initialize(version_base=None, config_path="../../configs"):
                    cfg = hydra.compose(
                        config_name="config", overrides=["tools/paper_download=default"]
                    )

                # Cache the configuration
                PaperDownloaderFactory._cached_config = cfg.tools.paper_download
                logger.info("Successfully loaded and cached paper download configuration")

                return PaperDownloaderFactory._cached_config

            except Exception as e:
                logger.error("Failed to load unified paper download configuration: %s", e)
                raise RuntimeError(f"Configuration loading failed: {e}") from e

    @staticmethod
    def _build_service_config(unified_config: Any, service: str) -> Any:
        """
        Build service-specific configuration by merging common and service settings.
        Handles Hydra's OmegaConf objects properly.

        Args:
            unified_config: The unified configuration object
            service: Service name

        Returns:
            Service-specific configuration object
        """
        if not hasattr(unified_config, "services") or service not in unified_config.services:
            raise ValueError(f"Service '{service}' not found in configuration")

        # Create a simple config object that combines common and service-specific settings
        class ServiceConfig:
            """Service-specific configuration holder."""

            def get_config_dict(self):
                """Return configuration as dictionary."""
                return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

            def has_attribute(self, name: str) -> bool:
                """Check if configuration has a specific attribute."""
                return hasattr(self, name)

        config_obj = ServiceConfig()

        # Handle common config (using helper method to reduce branches)
        PaperDownloaderFactory._apply_config(config_obj, unified_config.common, "common")

        # Handle service-specific config (using helper method to reduce branches)
        PaperDownloaderFactory._apply_config(config_obj, unified_config.services[service], service)

        return config_obj

    @staticmethod
    def _apply_config(config_obj: Any, source_config: Any, config_type: str) -> None:
        """
        Apply configuration from source to target object using multiple fallback methods.
        This preserves all the original logic but reduces branches in the main method.

        Args:
            config_obj: Target configuration object
            source_config: Source configuration to extract from
            config_type: Type description for logging
        """
        try:
            PaperDownloaderFactory._try_config_extraction(config_obj, source_config)
        except (AttributeError, TypeError, KeyError) as e:
            logger.warning("Failed to process %s config: %s", config_type, e)

    @staticmethod
    def _try_config_extraction(config_obj: Any, source_config: Any) -> None:
        """Try different methods to extract configuration data."""
        # Method 1: Try OmegaConf conversion
        if hasattr(source_config, "_content"):
            PaperDownloaderFactory._extract_from_omegaconf(config_obj, source_config)
            return

        # Method 2: Try direct attribute access
        if hasattr(source_config, "__dict__"):
            PaperDownloaderFactory._extract_from_dict(config_obj, source_config.__dict__)
            return

        # Method 3: Try items() method
        if hasattr(source_config, "items"):
            PaperDownloaderFactory._extract_from_items(config_obj, source_config)
            return

        # Method 4: Try dir() approach as fallback
        PaperDownloaderFactory._extract_from_dir(config_obj, source_config)

    @staticmethod
    def _extract_from_omegaconf(config_obj: Any, source_config: Any) -> None:
        """Extract configuration from OmegaConf object."""
        config_dict = OmegaConf.to_container(source_config, resolve=True)
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                if isinstance(key, str):  # Type guard for key
                    setattr(config_obj, key, value)

    @staticmethod
    def _extract_from_dict(config_obj: Any, config_dict: dict) -> None:
        """Extract configuration from dictionary."""
        for key, value in config_dict.items():
            if not key.startswith("_"):
                setattr(config_obj, key, value)

    @staticmethod
    def _extract_from_items(config_obj: Any, source_config: Any) -> None:
        """Extract configuration using items() method."""
        for key, value in source_config.items():
            if isinstance(key, str):  # Type guard for key
                setattr(config_obj, key, value)

    @staticmethod
    def _extract_from_dir(config_obj: Any, source_config: Any) -> None:
        """Extract configuration using dir() approach as fallback."""
        for key in dir(source_config):
            if not key.startswith("_"):
                value = getattr(source_config, key)
                if not callable(value):
                    setattr(config_obj, key, value)


@tool(
    args_schema=UnifiedPaperDownloadInput,
    parse_docstring=True,
)
def download_papers(
    service: Literal["arxiv", "medrxiv", "biorxiv", "pubmed"] | None,
    identifiers: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """
    Universal paper download tool supporting multiple academic paper services.

    Downloads paper metadata and PDFs from arXiv, medRxiv, bioRxiv, or PubMed and stores them
    in temporary files for further processing. The downloaded PDFs can be accessed
    using the temp_file_path in the returned metadata.

    Args:
        service: Paper service to download from (optional, uses configured default if not specified)
            - 'arxiv': For arXiv preprints (requires arXiv IDs)
            - 'medrxiv': For medRxiv preprints (requires DOIs)
            - 'biorxiv': For bioRxiv preprints (requires DOIs)
            - 'pubmed': For PubMed papers (requires PMIDs)
        identifiers: List of paper identifiers in the format expected by the service

    Returns:
        Command with article_data containing paper metadata and local file paths

    Examples:
        # Download from arXiv
        download_papers("arxiv", ["1234.5678", "2301.12345"])

        # Download from medRxiv
        download_papers("medrxiv", ["10.1101/2020.09.09.20191205"])

        # Download from bioRxiv
        download_papers("biorxiv", ["10.1101/2020.09.09.20191205"])

        # Download from PubMed
        download_papers("pubmed", ["12345678", "87654321"])

        # Use default service (configured in default.yaml)
        download_papers(None, ["12345678", "87654321"])
    """
    return _download_papers_impl(service, identifiers, tool_call_id)


# Convenience functions for backward compatibility (optional)
# These functions explicitly specify the service, bypassing the default service config
def download_arxiv_papers(
    arxiv_ids: list[str], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Convenience function for downloading arXiv papers (explicitly uses arXiv service)."""
    return _download_papers_impl("arxiv", arxiv_ids, tool_call_id)


def download_medrxiv_papers(
    dois: list[str], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Convenience function for downloading medRxiv papers (explicitly uses medRxiv service)."""
    return _download_papers_impl("medrxiv", dois, tool_call_id)


def download_biorxiv_papers(
    dois: list[str], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Convenience function for downloading bioRxiv papers (explicitly uses bioRxiv service)."""
    return _download_papers_impl("biorxiv", dois, tool_call_id)


def download_pubmed_papers(
    pmids: list[str], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command[Any]:
    """Convenience function for downloading PubMed papers (explicitly uses PubMed service)."""
    return _download_papers_impl("pubmed", pmids, tool_call_id)


def _download_papers_impl(
    service: Literal["arxiv", "medrxiv", "biorxiv", "pubmed"] | None,
    identifiers: list[str],
    tool_call_id: str,
) -> Command[Any]:
    """
    Internal implementation function that contains the actual download logic.
    This is called by both the decorated tool and the convenience functions.
    """
    # Resolve default service if not specified
    if service is None:
        service = PaperDownloaderFactory.get_default_service()
        logger.info("No service specified, using configured default: %s", service)
    logger.info(
        "Starting unified paper download for service '%s' with %d identifiers: %s",
        service,
        len(identifiers),
        identifiers,
    )

    try:
        # Step 1: Create appropriate downloader using factory
        downloader = PaperDownloaderFactory.create(service)
        logger.info("Created %s downloader successfully", downloader.get_service_name())

        # Step 2: Process all identifiers
        article_data = downloader.process_identifiers(identifiers)

        # Step 3: Build summary for user
        content = downloader.build_summary(article_data)

        # Step 4: Log results summary
        total_papers = len(article_data)
        successful_downloads = sum(
            1
            for paper in article_data.values()
            if paper.get("access_type") == "open_access_downloaded"
        )
        logger.info(
            "Download complete for %s: %d papers processed, %d PDFs downloaded",
            service,
            total_papers,
            successful_downloads,
        )

        # Step 5: Return command with results
        return Command(
            update={
                "article_data": article_data,
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        artifact=article_data,
                    )
                ],
            }
        )

    except ValueError as e:
        # Handle service/configuration errors
        error_msg = f"Service error for '{service}': {str(e)}"
        logger.error(error_msg)

        return Command(
            update={
                "article_data": {},
                "messages": [
                    ToolMessage(
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call_id,
                        artifact={},
                    )
                ],
            }
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        # Handle unexpected errors
        error_msg = f"Unexpected error during paper download: {str(e)}"
        logger.error(error_msg, exc_info=True)

        return Command(
            update={
                "article_data": {},
                "messages": [
                    ToolMessage(
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call_id,
                        artifact={},
                    )
                ],
            }
        )
