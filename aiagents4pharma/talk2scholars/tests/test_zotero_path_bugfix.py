#!/usr/bin/env python3

"""
Tests for Zotero tools to verify fixes for specific bugs:
1. Zotero Write failing when Zotero Read hasn't been called
2. Incorrect Collection Target (saving to wrong collection)
"""

import pytest
from unittest.mock import MagicMock, patch
from aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path import (
    fetch_papers_for_save,
    find_or_create_collection,
    get_all_collection_paths,
)


class TestZoteroPath:
    """Tests for the zotero_path utility functions."""

    def test_fetch_papers_for_save_no_papers(self):
        """Test that fetch_papers_for_save returns None when no papers are available."""
        # Empty state
        state = {}
        assert fetch_papers_for_save(state) is None

        # State with empty last_displayed_papers
        state = {"last_displayed_papers": ""}
        assert fetch_papers_for_save(state) is None

        # State with last_displayed_papers pointing to non-existent key
        state = {"last_displayed_papers": "nonexistent_key"}
        assert fetch_papers_for_save(state) is None

    def test_fetch_papers_for_save_with_papers(self):
        """Test that fetch_papers_for_save correctly retrieves papers from state."""
        # State with direct papers
        sample_papers = {"paper1": {"Title": "Test Paper"}}
        state = {"last_displayed_papers": sample_papers}
        assert fetch_papers_for_save(state) == sample_papers

        # State with papers referenced by key
        state = {"last_displayed_papers": "zotero_read", "zotero_read": sample_papers}
        assert fetch_papers_for_save(state) == sample_papers

    @patch("pyzotero.zotero.Zotero")
    def test_find_or_create_collection_exact_match(self, mock_zotero):
        """Test that find_or_create_collection correctly finds an exact match."""
        # Setup mock
        mock_zot = MagicMock()
        mock_zotero.return_value = mock_zot

        # Setup collections
        collections = [
            {"key": "abc123", "data": {"name": "Curiosity", "parentCollection": None}},
            {
                "key": "def456",
                "data": {"name": "Curiosity1", "parentCollection": "abc123"},
            },
            {"key": "ghi789", "data": {"name": "Random", "parentCollection": None}},
            {"key": "rad123", "data": {"name": "radiation", "parentCollection": None}},
        ]
        mock_zot.collections.return_value = collections

        # Test finding "Curiosity"
        result = find_or_create_collection(mock_zot, "/Curiosity")
        assert result == "abc123"

        # Test finding with different case
        result = find_or_create_collection(mock_zot, "/curiosity")
        assert result == "abc123"

        # Test finding "radiation" - direct match
        result = find_or_create_collection(mock_zot, "/radiation")
        assert result == "rad123"

        # Test finding without leading slash
        result = find_or_create_collection(mock_zot, "radiation")
        assert result == "rad123"

    @patch("pyzotero.zotero.Zotero")
    def test_find_or_create_collection_no_match(self, mock_zotero):
        """Test that find_or_create_collection returns None for non-existent collections."""
        # Setup mock
        mock_zot = MagicMock()
        mock_zotero.return_value = mock_zot

        # Setup collections
        collections = [
            {"key": "abc123", "data": {"name": "Curiosity", "parentCollection": None}},
            {
                "key": "def456",
                "data": {"name": "Curiosity1", "parentCollection": "abc123"},
            },
        ]
        mock_zot.collections.return_value = collections

        # Test finding non-existent "Curiosity2"
        result = find_or_create_collection(mock_zot, "/Curiosity2")
        assert result is None

        # Test finding non-existent nested path
        result = find_or_create_collection(mock_zot, "/Curiosity/Curiosity2")
        assert result is None

    @patch("pyzotero.zotero.Zotero")
    def test_find_or_create_collection_with_creation(self, mock_zotero):
        """Test that find_or_create_collection creates collections when requested."""
        # Setup mock
        mock_zot = MagicMock()
        mock_zotero.return_value = mock_zot

        # Setup collections
        collections = [
            {"key": "abc123", "data": {"name": "Curiosity", "parentCollection": None}}
        ]
        mock_zot.collections.return_value = collections

        # Setup create_collection response
        mock_zot.create_collection.return_value = {
            "successful": {"0": {"data": {"key": "new_key"}}}
        }

        # Test creating "Curiosity2" - note we're expecting lowercase in the call
        result = find_or_create_collection(mock_zot, "/Curiosity2", create_missing=True)
        assert result == "new_key"
        # Use case-insensitive check for the collection name
        mock_zot.create_collection.assert_called_once()
        call_args = mock_zot.create_collection.call_args[0][0]
        assert "name" in call_args
        assert call_args["name"].lower() == "curiosity2"

        # Test creating nested "Curiosity/Curiosity2"
        mock_zot.create_collection.reset_mock()
        result = find_or_create_collection(
            mock_zot, "/Curiosity/Curiosity2", create_missing=True
        )
        assert result == "new_key"
        # Check that the call includes parentCollection
        mock_zot.create_collection.assert_called_once()
        call_args = mock_zot.create_collection.call_args[0][0]
        assert "name" in call_args
        assert "parentCollection" in call_args
        assert call_args["name"].lower() == "curiosity2"
        assert call_args["parentCollection"] == "abc123"

    @patch("pyzotero.zotero.Zotero")
    def test_get_all_collection_paths(self, mock_zotero):
        """Test that get_all_collection_paths returns correct paths."""
        # Setup mock
        mock_zot = MagicMock()
        mock_zotero.return_value = mock_zot

        # Setup collections
        collections = [
            {"key": "abc123", "data": {"name": "Curiosity", "parentCollection": None}},
            {
                "key": "def456",
                "data": {"name": "Curiosity1", "parentCollection": "abc123"},
            },
            {"key": "ghi789", "data": {"name": "Random", "parentCollection": None}},
        ]
        mock_zot.collections.return_value = collections

        # Test getting all paths
        result = get_all_collection_paths(mock_zot)
        assert "/Curiosity" in result
        assert "/Random" in result
        assert "/Curiosity/Curiosity1" in result


class TestZoteroWrite:
    """Integration tests for zotero_write.py."""

    @pytest.fixture
    def mock_hydra(self):
        """Fixture to mock hydra configuration."""
        with (
            patch("hydra.initialize") as mock_init,
            patch("hydra.compose") as mock_compose,
        ):
            cfg = MagicMock()
            cfg.tools.zotero_write.user_id = "test_user"
            cfg.tools.zotero_write.library_type = "user"
            cfg.tools.zotero_write.api_key = "test_key"
            mock_compose.return_value = cfg
            yield

    @pytest.fixture
    def mock_zotero(self):
        """Fixture to mock Zotero client."""
        with patch("pyzotero.zotero.Zotero") as mock_zot_class:
            mock_zot = MagicMock()
            mock_zot_class.return_value = mock_zot
            yield mock_zot

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.fetch_papers_for_save"
    )
    def test_zotero_save_no_papers(self, mock_fetch, mock_hydra, mock_zotero):
        """Test that zotero_save_tool raises error when no papers are available."""
        from aiagents4pharma.talk2scholars.tools.zotero.zotero_write import (
            zotero_save_tool,
        )

        # Setup mock
        mock_fetch.return_value = None

        # Access the underlying function directly for testing
        underlying_func = zotero_save_tool.func

        # Test saving with no papers
        with pytest.raises(RuntimeError) as excinfo:
            # Call the underlying function directly with the arguments it expects
            underlying_func(
                tool_call_id="test_id", collection_path="/Curiosity", state={}
            )

        assert "No fetched papers were found" in str(excinfo.value)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.fetch_papers_for_save"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.find_or_create_collection"
    )
    def test_zotero_save_invalid_collection(self, mock_find, mock_hydra, mock_zotero):
        """Test that zotero_save_tool raises error for invalid collection path."""
        from aiagents4pharma.talk2scholars.tools.zotero.zotero_write import (
            zotero_save_tool,
        )

        # Create a test state with papers data
        sample_papers = {"paper1": {"Title": "Test Paper"}}
        test_state = {"papers": sample_papers, "last_displayed_papers": "papers"}

        # Set up mock for find_or_create_collection to return None (not found)
        with patch(
            "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.find_or_create_collection"
        ) as mock_find:
            mock_find.return_value = None

            # Mock get_all_collection_paths
            with patch(
                "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_all_collection_paths"
            ) as mock_get_paths:
                mock_get_paths.return_value = ["/Curiosity", "/Random"]

                # Access underlying function
                underlying_func = zotero_save_tool.func

                # Test saving to invalid collection
                with pytest.raises(RuntimeError) as excinfo:
                    underlying_func(
                        tool_call_id="test_id",
                        collection_path="/NonExistent",
                        state=test_state,
                    )

                # Check the correct error message is returned
                assert "does not exist in Zotero" in str(excinfo.value)

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.fetch_papers_for_save"
    )
    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.find_or_create_collection"
    )
    def test_zotero_save_success(self, mock_find, mock_fetch, mock_hydra, mock_zotero):
        """Test that zotero_save_tool successfully saves papers to a valid collection."""
        from aiagents4pharma.talk2scholars.tools.zotero.zotero_write import (
            zotero_save_tool,
        )

        # Setup mocks for successful save
        sample_papers = {"paper1": {"Title": "Test Paper", "Authors": ["Test Author"]}}
        mock_fetch.return_value = sample_papers
        mock_find.return_value = "abc123"  # Valid collection key

        # Setup collection name retrieval
        mock_zotero.collections.return_value = [
            {"key": "abc123", "data": {"name": "radiation"}}
        ]

        # Setup successful item creation
        mock_zotero.create_items.return_value = {
            "successful": {"0": {"key": "item123"}}
        }

        # Access underlying function
        underlying_func = zotero_save_tool.func

        # Create a test state that contains the fetched papers.
        test_state = {"last_displayed_papers": "papers", "papers": sample_papers}

        # Test successful save with valid state
        result = underlying_func(
            tool_call_id="test_id", collection_path="/radiation", state=test_state
        )

        # Verify result contains success message by accessing the update attribute
        assert result.update is not None
        assert "messages" in result.update
        assert len(result.update["messages"]) > 0
        assert "Save was successful" in result.update["messages"][0].content
        assert "radiation" in result.update["messages"][0].content


class TestZoteroRead:
    """Integration tests for zotero_read.py."""

    @pytest.fixture
    def mock_hydra(self):
        """Fixture to mock hydra configuration."""
        with (
            patch("hydra.initialize") as mock_init,
            patch("hydra.compose") as mock_compose,
        ):
            cfg = MagicMock()
            cfg.tools.zotero_read.user_id = "test_user"
            cfg.tools.zotero_read.library_type = "user"
            cfg.tools.zotero_read.api_key = "test_key"
            cfg.tools.zotero_read.zotero = MagicMock()
            cfg.tools.zotero_read.zotero.max_limit = 50
            cfg.tools.zotero_read.zotero.filter_item_types = [
                "journalArticle",
                "conferencePaper",
            ]
            mock_compose.return_value = cfg
            yield

    @pytest.fixture
    def mock_zotero(self):
        """Fixture to mock Zotero client."""
        with patch("pyzotero.zotero.Zotero") as mock_zot_class:
            mock_zot = MagicMock()
            mock_zot_class.return_value = mock_zot
            yield mock_zot

    @patch(
        "aiagents4pharma.talk2scholars.tools.zotero.utils.zotero_path.get_item_collections"
    )
    def test_zotero_read_item_collections_error(
        self, mock_get_collections, mock_hydra, mock_zotero
    ):
        """Test that zotero_read_tool handles errors in get_item_collections."""
        from aiagents4pharma.talk2scholars.tools.zotero.zotero_read import (
            zotero_search_tool,
        )

        # Setup mocks
        mock_get_collections.side_effect = Exception("Test error")

        # Setup items
        mock_zotero.items.return_value = [
            {
                "data": {
                    "key": "paper1",
                    "title": "Test Paper",
                    "itemType": "journalArticle",
                }
            }
        ]

        # Access the underlying function directly for testing
        underlying_func = zotero_search_tool.func

        # Test reading with get_item_collections error
        result = underlying_func(
            query="test", only_articles=True, tool_call_id="test_id", limit=2
        )

        # Verify the function didn't crash and returned some result
        assert result is not None
        assert isinstance(result.update, dict)
        assert "zotero_read" in result.update
