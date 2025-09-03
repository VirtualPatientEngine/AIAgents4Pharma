"""
Milvus Connection Manager for Talk2KnowledgeGraphs.

This module provides centralized connection management for Milvus database,
removing the dependency on frontend session state and enabling proper
separation of concerns between frontend and backend.
"""

import asyncio
import concurrent.futures
import logging
import threading
from dataclasses import dataclass
from typing import Any

import hydra
from pymilvus import AsyncMilvusClient, Collection, MilvusClient, connections, db
from pymilvus.exceptions import MilvusException

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchParams:
    """Parameters for search operations."""

    collection_name: str
    data: list
    anns_field: str
    search_params: dict
    limit: int
    output_fields: list | None = None


@dataclass
class QueryParams:
    """Parameters for query operations."""

    collection_name: str
    expr: str
    output_fields: list | None = None
    limit: int | None = None


class MilvusConnectionManager:
    """
    Centralized Milvus connection manager for backend tools with singleton pattern.

    This class handles:
    - Connection establishment and management
    - Database switching
    - Connection health checks
    - Graceful error handling
    - Thread-safe singleton pattern

    Args:
        cfg: Configuration object containing Milvus connection parameters
    """

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, cfg: dict[str, Any]):
        """
        Create singleton instance based on database configuration.

        Args:
            cfg: Configuration dictionary containing Milvus DB settings

        Returns:
            MilvusConnectionManager: Singleton instance for the given config
        """
        # Create a unique key based on connection parameters
        config_key = (
            cfg.milvus_db.host,
            int(cfg.milvus_db.port),
            cfg.milvus_db.user,
            cfg.milvus_db.database_name,
            cfg.milvus_db.alias,
        )

        if config_key not in cls._instances:
            with cls._lock:
                # Double-check locking pattern
                if config_key not in cls._instances:
                    instance = super().__new__(cls)
                    cls._instances[config_key] = instance
                    logger.info(
                        "Created new MilvusConnectionManager singleton for database: %s",
                        cfg.milvus_db.database_name,
                    )
        else:
            logger.debug(
                "Reusing existing MilvusConnectionManager singleton for database: %s",
                cfg.milvus_db.database_name,
            )

        return cls._instances[config_key]

    def __init__(self, cfg: dict[str, Any]):
        """
        Initialize the Milvus connection manager.

        Args:
            cfg: Configuration dictionary containing Milvus DB settings
        """
        # Prevent re-initialization of singleton instance
        if hasattr(self, "_initialized"):
            return

        self.cfg = cfg
        self.alias = cfg.milvus_db.alias
        self.host = cfg.milvus_db.host
        self.port = int(cfg.milvus_db.port)  # Ensure port is integer
        self.user = cfg.milvus_db.user
        self.password = cfg.milvus_db.password
        self.database_name = cfg.milvus_db.database_name

        # Thread lock for connection operations
        self._connection_lock = threading.Lock()

        # Initialize both sync and async clients
        self._sync_client = None
        self._async_client = None

        # Mark as initialized
        self._initialized = True

        logger.info("MilvusConnectionManager initialized for database: %s", self.database_name)

    def get_sync_client(self) -> MilvusClient:
        """
        Get or create a synchronous MilvusClient.

        Returns:
            MilvusClient: Configured synchronous client
        """
        if self._sync_client is None:
            self._sync_client = MilvusClient(
                uri=f"http://{self.host}:{self.port}",
                token=f"{self.user}:{self.password}",
                db_name=self.database_name,
            )
            logger.info("Created synchronous MilvusClient for database: %s", self.database_name)
        return self._sync_client

    def get_async_client(self) -> AsyncMilvusClient:
        """
        Get or create an asynchronous AsyncMilvusClient.

        Returns:
            AsyncMilvusClient: Configured asynchronous client
        """
        if self._async_client is None:
            try:
                self._async_client = AsyncMilvusClient(
                    uri=f"http://{self.host}:{self.port}",
                    token=f"{self.user}:{self.password}",
                    db_name=self.database_name,
                )
                logger.info(
                    "Created asynchronous AsyncMilvusClient for database: %s",
                    self.database_name,
                )
            except (MilvusException, RuntimeError, ConnectionError, OSError) as e:
                logger.error("Failed to create async client: %s", str(e))
                # Don't raise here, let the calling method handle the fallback
                return None
        return self._async_client

    def ensure_connection(self) -> bool:
        """
        Ensure Milvus connection exists, create if not.

        This method checks if a connection with the specified alias exists,
        and creates one if it doesn't. It also switches to the correct database.
        Thread-safe implementation with connection locking.

        Returns:
            bool: True if connection is established, False otherwise

        Raises:
            MilvusException: If connection cannot be established
        """
        with self._connection_lock:
            try:
                # Check if connection already exists
                if not connections.has_connection(self.alias):
                    logger.info("Creating new Milvus connection with alias: %s", self.alias)
                    connections.connect(
                        alias=self.alias,
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        password=self.password,
                    )
                    logger.info(
                        "Successfully connected to Milvus at %s:%s",
                        self.host,
                        self.port,
                    )
                else:
                    logger.debug("Milvus connection already exists with alias: %s", self.alias)

                # Switch to the correct database
                db.using_database(self.database_name)
                logger.debug("Using Milvus database: %s", self.database_name)

                return True

            except MilvusException as e:
                logger.error("Failed to establish Milvus connection: %s", str(e))
                raise
            except Exception as e:
                logger.error("Unexpected error during Milvus connection: %s", str(e))
                raise MilvusException(f"Connection failed: {str(e)}") from e

    def get_connection_info(self) -> dict[str, Any]:
        """
        Get current connection information.

        Returns:
            Dict containing connection details
        """
        try:
            if connections.has_connection(self.alias):
                conn_addr = connections.get_connection_addr(self.alias)
                return {
                    "alias": self.alias,
                    "host": self.host,
                    "port": self.port,
                    "database": self.database_name,
                    "connected": True,
                    "connection_address": conn_addr,
                }
            return {
                "alias": self.alias,
                "host": self.host,
                "port": self.port,
                "database": self.database_name,
                "connected": False,
                "connection_address": None,
            }
        except (MilvusException, RuntimeError, ConnectionError, OSError) as e:
            logger.error("Error getting connection info: %s", str(e))
            return {"alias": self.alias, "connected": False, "error": str(e)}

    def test_connection(self) -> bool:
        """
        Test the connection by attempting to list collections.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            self.ensure_connection()

            # Try to get a collection to test the connection
            test_collection_name = f"{self.database_name}_nodes"
            Collection(name=test_collection_name)

            logger.debug("Connection test successful")
            return True

        except (MilvusException, RuntimeError, ConnectionError, OSError) as e:
            logger.error("Connection test failed: %s", str(e))
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from Milvus (both sync and async clients).

        Returns:
            bool: True if disconnected successfully, False otherwise
        """
        try:
            success = True

            # Disconnect sync client
            if connections.has_connection(self.alias):
                connections.disconnect(self.alias)
                logger.info("Disconnected sync connection with alias: %s", self.alias)

            # Close async client if it exists
            if self._async_client is not None:
                try:
                    # Check if we can close the async client properly
                    try:
                        loop = asyncio.get_running_loop()
                        # If there's a running loop, create a task
                        loop.create_task(self._async_client.close())
                    except RuntimeError:
                        # No running loop, use asyncio.run in a thread
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            executor.submit(lambda: asyncio.run(self._async_client.close())).result(
                                timeout=5
                            )

                    self._async_client = None
                    logger.info("Closed async client for database: %s", self.database_name)
                except (TimeoutError, RuntimeError) as e:
                    logger.warning("Error closing async client: %s", str(e))
                    # Still clear the reference even if close failed
                    self._async_client = None
                    success = False

            # Clear sync client reference
            if self._sync_client is not None:
                self._sync_client = None
                logger.info("Cleared sync client reference")

            return success

        except (MilvusException, RuntimeError, ConnectionError, OSError) as e:
            logger.error("Error disconnecting from Milvus: %s", str(e))
            return False

    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a Milvus collection, ensuring connection is established.
        Thread-safe implementation.

        Args:
            collection_name: Name of the collection to retrieve

        Returns:
            Collection: The requested Milvus collection

        Raises:
            MilvusException: If collection cannot be retrieved
        """
        try:
            self.ensure_connection()
            collection = Collection(name=collection_name)
            collection.load()  # Load collection data
            logger.debug("Successfully loaded collection: %s", collection_name)
            return collection

        except Exception as e:
            logger.error("Failed to get collection %s: %s", collection_name, str(e))
            raise MilvusException(f"Failed to get collection {collection_name}: {str(e)}") from e

    async def async_search(self, params: SearchParams) -> list:
        """
        Perform asynchronous vector search.

        Args:
            params: SearchParams object containing all search parameters

        Returns:
            List of search results
        """
        try:
            async_client = self.get_async_client()
            if async_client is None:
                raise MilvusException("Failed to create async client")

            # Ensure collection is loaded before searching
            await async_client.load_collection(collection_name=params.collection_name)

            results = await async_client.search(
                collection_name=params.collection_name,
                data=params.data,
                anns_field=params.anns_field,
                search_params=params.search_params,
                limit=params.limit,
                output_fields=params.output_fields or [],
            )
            logger.debug("Async search completed for collection: %s", params.collection_name)
            return results
        except MilvusException as e:
            logger.warning(
                "Async search failed for collection %s: %s, falling back to sync",
                params.collection_name,
                str(e),
            )
            # Fallback to sync operation
            return await asyncio.to_thread(self._sync_search, params)

    def _sync_search(self, params: SearchParams) -> list:
        """Sync fallback for search operations."""
        try:
            collection = Collection(name=params.collection_name)
            collection.load()
            results = collection.search(
                data=params.data,
                anns_field=params.anns_field,
                param=params.search_params,
                limit=params.limit,
                output_fields=params.output_fields or [],
            )
            logger.debug(
                "Sync fallback search completed for collection: %s",
                params.collection_name,
            )
            return results
        except Exception as e:
            logger.error(
                "Sync fallback search failed for collection %s: %s",
                params.collection_name,
                str(e),
            )
            raise MilvusException(f"Search failed (sync fallback): {str(e)}") from e

    async def async_query(self, params: QueryParams) -> list:
        """
        Perform asynchronous query with sync fallback.

        Args:
            params: QueryParams object containing all query parameters

        Returns:
            List of query results
        """
        try:
            async_client = self.get_async_client()
            if async_client is None:
                raise MilvusException("Failed to create async client")

            # Ensure collection is loaded before querying
            await async_client.load_collection(collection_name=params.collection_name)

            results = await async_client.query(
                collection_name=params.collection_name,
                filter=params.expr,
                output_fields=params.output_fields or [],
                limit=params.limit,
            )
            logger.debug("Async query completed for collection: %s", params.collection_name)
            return results
        except MilvusException as e:
            logger.warning(
                "Async query failed for collection %s: %s, falling back to sync",
                params.collection_name,
                str(e),
            )
            # Fallback to sync operation
            return await asyncio.to_thread(self._sync_query, params)

    def _sync_query(self, params: QueryParams) -> list:
        """Sync fallback for query operations."""
        try:
            collection = Collection(name=params.collection_name)
            collection.load()
            results = collection.query(
                expr=params.expr,
                output_fields=params.output_fields or [],
                limit=params.limit,
            )
            logger.debug(
                "Sync fallback query completed for collection: %s",
                params.collection_name,
            )
            return results
        except Exception as e:
            logger.error(
                "Sync fallback query failed for collection %s: %s",
                params.collection_name,
                str(e),
            )
            raise MilvusException(f"Query failed (sync fallback): {str(e)}") from e

    async def async_load_collection(self, collection_name: str) -> bool:
        """
        Asynchronously load a collection.

        Args:
            collection_name: Name of the collection to load

        Returns:
            bool: True if loaded successfully
        """
        try:
            async_client = self.get_async_client()
            await async_client.load_collection(collection_name=collection_name)
            logger.debug("Async load completed for collection: %s", collection_name)
            return True
        except Exception as e:
            logger.error("Async load failed for collection %s: %s", collection_name, str(e))
            raise MilvusException(f"Async load failed: {str(e)}") from e

    async def async_get_collection_stats(self, collection_name: str) -> dict:
        """
        Get collection statistics asynchronously.

        Args:
            collection_name: Name of the collection

        Returns:
            dict: Collection statistics
        """
        try:
            # Note: Using sync client methods through asyncio.to_thread as fallback
            # since AsyncMilvusClient might not have all stat methods
            stats = await asyncio.to_thread(lambda: Collection(name=collection_name).num_entities)
            return {"num_entities": stats}
        except Exception as e:
            logger.error(
                "Failed to get async collection stats for %s: %s",
                collection_name,
                str(e),
            )
            raise MilvusException(f"Failed to get collection stats: {str(e)}") from e

    @classmethod
    def get_instance(cls, cfg: dict[str, Any]) -> "MilvusConnectionManager":
        """
        Get singleton instance for the given configuration.

        Args:
            cfg: Configuration dictionary containing Milvus DB settings

        Returns:
            MilvusConnectionManager: Singleton instance for the given config
        """
        return cls(cfg)

    @classmethod
    def clear_instances(cls):
        """
        Clear all singleton instances. Useful for testing or cleanup.
        """
        with cls._lock:
            # Disconnect all existing connections before clearing
            for instance in cls._instances.values():
                instance.disconnect()
            cls._instances.clear()
            logger.info("Cleared all MilvusConnectionManager singleton instances")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MilvusConnectionManager":
        """
        Create a MilvusConnectionManager from configuration.

        Args:
            cfg: Configuration object or dictionary

        Returns:
            MilvusConnectionManager: Configured connection manager instance
        """
        return cls(cfg)

    @classmethod
    def from_hydra_config(
        cls,
        config_path: str = "../configs",
        config_name: str = "config",
        overrides: list | None = None,
    ) -> "MilvusConnectionManager":
        """
        Create a MilvusConnectionManager from Hydra configuration.

        This method loads the Milvus database configuration using Hydra,
        providing complete backend separation from frontend configs.

        Args:
            config_path: Path to the configs directory
            config_name: Name of the main config file
            overrides: List of config overrides

        Returns:
            MilvusConnectionManager: Configured connection manager instance

        Example:
            # Load with default database config
            conn_manager = MilvusConnectionManager.from_hydra_config()

            # Load with specific overrides
            conn_manager = MilvusConnectionManager.from_hydra_config(
                overrides=["utils/database/milvus=default"]
            )
        """
        if overrides is None:
            overrides = ["utils/database/milvus=default"]

        try:
            with hydra.initialize(version_base=None, config_path=config_path):
                cfg_all = hydra.compose(config_name=config_name, overrides=overrides)
                cfg = cfg_all.utils.database.milvus  # Extract utils.database.milvus section
                logger.info("Loaded Milvus config from Hydra with overrides: %s", overrides)
                return cls(cfg)
        except Exception as e:
            logger.error("Failed to load Hydra configuration: %s", str(e))
            raise MilvusException(f"Configuration loading failed: {str(e)}") from e
