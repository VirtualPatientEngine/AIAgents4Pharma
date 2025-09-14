"""
Unit tests for the Vectorstore class with GPU support and embedding normalization.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store import Vectorstore

MODULE = "aiagents4pharma.talk2scholars.tools.pdf.utils.vector_store"


@pytest.fixture(name="mock_config")
def _mock_config():
    """
    Fixture providing a mock configuration object with default GPU detection off.
    """
    return SimpleNamespace(
        milvus=SimpleNamespace(
            host="localhost",
            port=19530,
            collection_name="test_collection",
            db_name="test_db",
            embedding_dim=384,
        ),
        gpu_detection=SimpleNamespace(force_cpu_mode=False),
    )


@pytest.fixture(name="mock_embedding")
def _mock_embedding():
    """
    Fixture providing a mock Embeddings model.
    """
    return MagicMock(spec=Embeddings)


@pytest.fixture(name="dummy_embedding")
def _dummy_embedding():
    """
    Fixture providing a dummy Embeddings model.
    """
    return MagicMock(spec=Embeddings)


@pytest.fixture(name="dummy_config")
def _dummy_config():
    """
    Fixture providing a dummy configuration object.
    """
    return SimpleNamespace(
        milvus=SimpleNamespace(
            host="localhost",
            port=19530,
            collection_name="test_collection",
            db_name="test_db",
            embedding_dim=768,
        ),
        gpu_detection=SimpleNamespace(force_cpu_mode=False),
    )


@pytest.fixture(name="dummy_vectorstore_components")
def _dummy_vectorstore_components():
    """
    Provides VectorstoreSingleton mock and vector_store with empty collection.
    """
    with (
        patch(f"{MODULE}.detect_nvidia_gpu", return_value=True),
        patch(
            f"{MODULE}.get_optimal_index_config",
            return_value=(
                {"index_type": "IVF_FLAT", "metric_type": "IP"},
                {"nprobe": 10},
            ),
        ),
        patch(f"{MODULE}.ensure_collection_exists", return_value=MagicMock()),
        patch(f"{MODULE}.VectorstoreSingleton") as singleton_cls,
    ):
        mock_singleton = MagicMock()
        mock_vector_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.num_entities = 0
        mock_collection.flush.return_value = None
        mock_vector_store.col = mock_collection
        mock_vector_store.collection = mock_collection
        mock_singleton.get_vector_store.return_value = mock_vector_store
        mock_singleton.get_connection.return_value = "connected"
        singleton_cls.return_value = mock_singleton
        yield mock_singleton, mock_vector_store


def test_vectorstore_initialization(mock_config, mock_embedding):
    """
    Test Vectorstore initialization with GPU and mocked dependencies.
    """
    with (
        patch(f"{MODULE}.detect_nvidia_gpu", return_value=True),
        patch(f"{MODULE}.log_index_configuration"),
        patch(
            f"{MODULE}.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(f"{MODULE}.wrap_embedding_model_if_needed", return_value=mock_embedding),
        patch(f"{MODULE}.ensure_collection_exists", return_value="mock_collection"),
        patch(f"{MODULE}.VectorstoreSingleton") as singleton_cls,
    ):
        mock_singleton = MagicMock()
        mock_vector_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.num_entities = 0
        mock_collection.flush.return_value = None
        mock_vector_store.col = mock_collection
        mock_vector_store.collection = mock_collection
        mock_singleton.get_vector_store.return_value = mock_vector_store
        mock_singleton.get_connection.return_value = None
        singleton_cls.return_value = mock_singleton

        vs = Vectorstore(embedding_model=mock_embedding, config=mock_config)

        assert vs.embedding_model is mock_embedding
        assert vs.collection == "mock_collection"
        assert vs.has_gpu
        assert vs.vector_store is mock_vector_store


def test_get_embedding_info(mock_config, mock_embedding):
    """
    Test retrieval of embedding configuration info.
    """
    with (
        patch(f"{MODULE}.detect_nvidia_gpu", return_value=True),
        patch(f"{MODULE}.log_index_configuration"),
        patch(
            f"{MODULE}.get_optimal_index_config",
            return_value=({"metric_type": "IP", "index_type": "IVF"}, {}),
        ),
        patch(f"{MODULE}.wrap_embedding_model_if_needed", return_value=mock_embedding),
        patch(f"{MODULE}.ensure_collection_exists", return_value="mock_collection"),
        patch(f"{MODULE}.VectorstoreSingleton") as singleton_cls,
    ):
        mock_singleton = MagicMock()
        mock_vector_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.num_entities = 0
        mock_collection.flush.return_value = None
        mock_vector_store.col = mock_collection
        mock_vector_store.collection = mock_collection
        mock_singleton.get_vector_store.return_value = mock_vector_store
        mock_singleton.get_connection.return_value = None
        singleton_cls.return_value = mock_singleton

        vs = Vectorstore(embedding_model=mock_embedding, config=mock_config)
        info = vs.get_embedding_info()

        assert info["has_gpu"]
        assert info["use_cosine"]
        assert "original_model_type" in info
        assert "wrapped_model_type" in info
        assert "normalization_enabled" in info


def test_load_existing_papers_with_exception(mock_embedding, mock_config):
    """
    Test that _load_existing_paper_ids propagates on flush failure.
    """
    with (
        patch(f"{MODULE}.wrap_embedding_model_if_needed", return_value=mock_embedding),
        patch(f"{MODULE}.VectorstoreSingleton") as singleton_cls,
        patch(f"{MODULE}.ensure_collection_exists"),
        patch(f"{MODULE}.detect_nvidia_gpu", return_value=True),
        patch(
            f"{MODULE}.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(f"{MODULE}.log_index_configuration"),
    ):
        mock_singleton = MagicMock()
        # Set up failing store directly for initialization
        bad_collection = MagicMock()
        bad_collection.num_entities = 0
        bad_collection.flush.side_effect = Exception("flush failed")
        bad_store = MagicMock()
        bad_store.col = bad_collection
        bad_store.collection = bad_collection
        mock_singleton.get_vector_store.return_value = bad_store
        mock_singleton.get_connection.return_value = None
        singleton_cls.return_value = mock_singleton

        # Test error propagation through initialization that calls _load_existing_paper_ids
        with pytest.raises(Exception) as excinfo:
            Vectorstore(embedding_model=mock_embedding, config=mock_config)
        assert "flush failed" in str(excinfo.value)


def test_ensure_collection_loaded_with_entities(mock_embedding, mock_config):
    """
    Test that _ensure_collection_loaded loads data when entities > 0.
    """
    with (
        patch(f"{MODULE}.wrap_embedding_model_if_needed", return_value=mock_embedding),
        patch(f"{MODULE}.VectorstoreSingleton") as singleton_cls,
        patch(f"{MODULE}.ensure_collection_exists"),
        patch(f"{MODULE}.detect_nvidia_gpu", return_value=True),
        patch(
            f"{MODULE}.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(f"{MODULE}.log_index_configuration"),
    ):
        mock_singleton = MagicMock()
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.num_entities = 5
        mock_collection.flush.return_value = None
        mock_store.col = mock_collection
        mock_store.collection = mock_collection
        mock_singleton.get_vector_store.return_value = mock_store
        mock_singleton.get_connection.return_value = None
        singleton_cls.return_value = mock_singleton

        # Test through initialization which calls _ensure_collection_loaded
        Vectorstore(embedding_model=mock_embedding, config=mock_config)
        assert mock_collection.load.called


def test_ensure_collection_loaded_handles_exception(mock_embedding, mock_config):
    """
    Test that _ensure_collection_loaded propagates on flush failure.
    """
    with (
        patch(f"{MODULE}.wrap_embedding_model_if_needed", return_value=mock_embedding),
        patch(f"{MODULE}.VectorstoreSingleton") as singleton_cls,
        patch(f"{MODULE}.ensure_collection_exists"),
        patch(f"{MODULE}.detect_nvidia_gpu", return_value=True),
        patch(
            f"{MODULE}.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(f"{MODULE}.log_index_configuration"),
    ):
        mock_singleton = MagicMock()
        # Set up failing store directly for initialization
        bad_collection = MagicMock()
        bad_collection.num_entities = 0
        bad_collection.flush.side_effect = Exception("flush error")
        bad_store = MagicMock()
        bad_store.col = bad_collection
        bad_store.collection = bad_collection
        mock_singleton.get_vector_store.return_value = bad_store
        mock_singleton.get_connection.return_value = None
        singleton_cls.return_value = mock_singleton

        # Test error propagation through initialization that calls _ensure_collection_loaded
        with pytest.raises(Exception) as excinfo:
            Vectorstore(embedding_model=mock_embedding, config=mock_config)
        assert "flush error" in str(excinfo.value)


def test_force_cpu_mode_logs_override(mock_config, mock_embedding):
    """
    Test that forcing CPU mode via config disables GPU detection.
    """
    mock_config.gpu_detection.force_cpu_mode = True
    with (
        patch(f"{MODULE}.wrap_embedding_model_if_needed", return_value=mock_embedding),
        patch(f"{MODULE}.VectorstoreSingleton") as singleton_cls,
        patch(f"{MODULE}.ensure_collection_exists", return_value="mock_collection"),
        patch(f"{MODULE}.detect_nvidia_gpu", return_value=True),
        patch(
            f"{MODULE}.get_optimal_index_config",
            return_value=({"metric_type": "IP"}, {}),
        ),
        patch(f"{MODULE}.log_index_configuration"),
    ):
        mock_singleton = MagicMock()
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_collection.num_entities = 0
        mock_collection.flush.return_value = None
        mock_store.col = mock_collection
        mock_store.collection = mock_collection
        mock_singleton.get_vector_store.return_value = mock_store
        mock_singleton.get_connection.return_value = None
        singleton_cls.return_value = mock_singleton

        vs = Vectorstore(embedding_model=mock_embedding, config=mock_config)

        assert not vs.has_gpu


def test_similarity_metric_override(dummy_embedding, dummy_config, dummy_vectorstore_components):
    """
    Test setting of use_cosine from config.similarity_metric.
    """
    dummy_config.similarity_metric = SimpleNamespace(use_cosine=False)
    # unpack and ignore vector_store
    _singleton, _mock_vector_store = dummy_vectorstore_components
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    assert not vs.use_cosine


def test_load_existing_paper_ids_fallback_to_collection(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """
    Test fallback if both `col` and `collection` missing.
    """
    _, mock_vector_store = dummy_vectorstore_components
    for attr in ("col", "collection"):
        if hasattr(mock_vector_store, attr):
            delattr(mock_vector_store, attr)

    vs = Vectorstore(dummy_embedding, config=dummy_config)
    # The loaded_papers is set during initialization via _load_existing_paper_ids
    assert isinstance(vs.loaded_papers, set)


def test_load_existing_papers_collection_empty_logs(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """
    Test logging when collection empty in _load_existing_paper_ids.
    """
    _, mock_vector_store = dummy_vectorstore_components
    mock_collection = MagicMock()
    mock_collection.num_entities = 0
    mock_collection.flush.return_value = None
    mock_vector_store.col = mock_collection

    vs = Vectorstore(dummy_embedding, config=dummy_config)
    # The loaded_papers is set during initialization via _load_existing_paper_ids
    assert len(vs.loaded_papers) == 0


def test_similarity_search_filter_paths(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """
    Test filter expression generation in similarity_search.
    """
    _, mock_vector_store = dummy_vectorstore_components
    mock_vector_store.similarity_search.return_value = [Document(page_content="test")]
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store

    filters = {
        "field1": "value",
        "field2": [1, 2],
        "field3": 99,
        "field4": 3.14,
    }
    result = vs.similarity_search(query="text", filter=filters)
    assert isinstance(result, list)


def test_mmr_search_filter_paths(dummy_embedding, dummy_config, dummy_vectorstore_components):
    """
    Test filter expression generation in max_marginal_relevance_search.
    """
    _, mock_vector_store = dummy_vectorstore_components
    mock_vector_store.max_marginal_relevance_search.return_value = [Document(page_content="test")]
    vs = Vectorstore(dummy_embedding, config=dummy_config)
    vs.vector_store = mock_vector_store

    filters = {"f": "text", "g": ["a", "b"], "h": 7, "j": 3.3}
    result = vs.max_marginal_relevance_search(query="q", filter=filters)
    assert isinstance(result, list)


def test_ensure_collection_loaded_no_col_and_no_collection(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """
    Test no-op when no collection attributes present.
    """
    _, mock_vector_store = dummy_vectorstore_components
    for attr in ("col", "collection"):
        if hasattr(mock_vector_store, attr):
            delattr(mock_vector_store, attr)

    # Test initialization succeeds without exception
    Vectorstore(dummy_embedding, config=dummy_config)
    # Collection loading is handled during initialization via _ensure_collection_loaded
    # no exception if we got this far


def test_ensure_collection_loaded_empty_logs(
    dummy_embedding, dummy_config, dummy_vectorstore_components
):
    """
    Test logging when collection empty in _ensure_collection_loaded.
    """
    _, mock_vector_store = dummy_vectorstore_components
    mock_collection = MagicMock()
    mock_collection.num_entities = 0
    mock_vector_store.col = mock_collection

    # Test initialization succeeds without exception
    Vectorstore(dummy_embedding, config=dummy_config)
    # Collection loading is handled during initialization via _ensure_collection_loaded
    # no exception if we got this far
