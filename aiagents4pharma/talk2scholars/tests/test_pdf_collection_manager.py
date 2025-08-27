"""collection_manager for managing Milvus collections for PDF chunks."""

from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
import pytest

from aiagents4pharma.talk2scholars.tools.pdf.utils import collection_manager


# -- Fixtures --


@pytest.fixture
def config_mock():
    """Dataclass config fixture to simulate Milvus config."""

    @dataclass
    class MilvusConfig:
        """Simulated Milvus inner config."""

        embedding_dim: int = 768

    @dataclass
    class Config:
        """Simulated outer config."""

        milvus: MilvusConfig = field(default_factory=MilvusConfig)

    return Config()


@pytest.fixture
def index_params():
    """Fixture to provide index parameters for tests."""
    return {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}


# -- Safe collection_cache access --


def set_collection_cache(key, value):
    """Set a mocked collection into the cache."""
    getattr(collection_manager, "_collection_cache")[key] = value


def clear_collection_cache(key):
    """Remove a mocked collection from the cache."""
    getattr(collection_manager, "_collection_cache").pop(key, None)


# -- Tests --


def test_cached_collection_returned(request):
    """Check if cached collection is returned."""
    config = request.getfixturevalue("config_mock")
    index = request.getfixturevalue("index_params")
    mock_collection = MagicMock()
    collection_name = "test_cached"

    set_collection_cache(collection_name, mock_collection)

    result = collection_manager.ensure_collection_exists(
        collection_name, config, index, has_gpu=False
    )

    assert result == mock_collection
    clear_collection_cache(collection_name)


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.Collection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.utility")
def test_create_new_collection(mock_utility, mock_collection_cls, request):
    """Check if new collection is created when it does not exist."""
    config = request.getfixturevalue("config_mock")
    index = request.getfixturevalue("index_params")
    mock_utility.list_collections.return_value = []

    mock_collection = MagicMock()
    mock_collection_cls.return_value = mock_collection
    mock_collection.indexes = [MagicMock(field_name="embedding")]
    mock_collection.num_entities = 5

    result = collection_manager.ensure_collection_exists(
        "new_collection", config, index, has_gpu=True
    )

    assert mock_collection.create_index.called
    assert mock_collection.load.called
    assert result == mock_collection


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.Collection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.utility")
def test_load_existing_collection(mock_utility, mock_collection_cls, request):
    """Test loading an existing collection."""
    config = request.getfixturevalue("config_mock")
    index = request.getfixturevalue("index_params")
    mock_utility.list_collections.return_value = ["existing_collection"]

    mock_collection = MagicMock()
    mock_collection_cls.return_value = mock_collection
    mock_collection.indexes = []
    mock_collection.num_entities = 0

    result = collection_manager.ensure_collection_exists(
        "existing_collection", config, index, has_gpu=False
    )

    mock_collection.load.assert_called_once()
    assert result == mock_collection


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.Collection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.utility")
def test_debug_collection_state_failure(mock_utility, mock_collection_cls, request):
    """debug_collection_state should log but not raise on failure."""
    config = request.getfixturevalue("config_mock")
    index = request.getfixturevalue("index_params")
    mock_utility.list_collections.return_value = ["bad_collection"]

    mock_collection = MagicMock()
    mock_collection_cls.return_value = mock_collection
    mock_collection.indexes = []
    mock_collection.num_entities = 10

    mock_collection.schema = property(
        lambda _: (_ for _ in ()).throw(Exception("bad schema"))
    )

    result = collection_manager.ensure_collection_exists(
        "bad_collection", config, index, has_gpu=True
    )

    assert result == mock_collection


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.Collection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.collection_manager.utility")
def test_ensure_collection_exception(mock_utility, mock_collection_cls, request):
    """ensure_collection_exists should raise on utility failure."""
    config = request.getfixturevalue("config_mock")
    index = request.getfixturevalue("index_params")
    mock_utility.list_collections.side_effect = RuntimeError("milvus failure")
    mock_collection_cls.return_value = MagicMock()

    with pytest.raises(RuntimeError, match="milvus failure"):
        collection_manager.ensure_collection_exists(
            "fail_collection", config, index, has_gpu=False
        )
