"""
Tests for singleton_manager: manages vector store connections and event loops.
"""

from unittest.mock import MagicMock, patch

import pytest
from pymilvus.exceptions import MilvusException

from aiagents4pharma.talk2scholars.tools.pdf.utils.get_vectorstore import (
    get_vectorstore,
)
from aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager import (
    VectorstoreSingleton,
)


def test_singleton_instance_identity():
    """Singleton should return the same instance."""
    a = VectorstoreSingleton()
    b = VectorstoreSingleton()
    assert a is b


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.detect_nvidia_gpu")
def test_detect_gpu_once(mock_detect, monkeypatch):
    """Ensure GPU detection is cached."""
    mock_detect.return_value = True
    singleton = VectorstoreSingleton()

    # Reset GPU detection cache safely
    monkeypatch.setattr(VectorstoreSingleton, "_gpu_detected", None, raising=False)

    result = singleton.detect_gpu_once()
    assert result is True

    # Second call should use cached value; detect_nvidia_gpu called only once
    result2 = singleton.detect_gpu_once()
    assert result2 is True
    mock_detect.assert_called_once()


def test_get_event_loop_reuses_existing():
    """get_event_loop should return the same loop if it exists."""
    singleton = VectorstoreSingleton()
    loop1 = singleton.get_event_loop()
    loop2 = singleton.get_event_loop()
    assert loop1 is loop2


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.connections")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.db")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.utility")
def test_get_connection_creates_connection(_, mock_db, mock_conns):
    """get_connection should create a new connection if none exists."""
    singleton = VectorstoreSingleton()
    mock_conns.has_connection.return_value = True
    mock_db.list_database.return_value = []

    conn_key = singleton.get_connection("localhost", 19530, "test_db")
    assert conn_key == "default"
    mock_conns.remove_connection.assert_called_once()
    mock_conns.connect.assert_called_once()
    mock_db.create_database.assert_called_once_with("test_db")
    mock_db.using_database.assert_called_once_with("test_db")


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.Milvus")
def test_get_vector_store_creates_if_missing(mock_milvus, monkeypatch):
    """get_vector_store should create a new vector store if missing."""
    singleton = VectorstoreSingleton()

    # Clear caches safely
    monkeypatch.setattr(VectorstoreSingleton, "_vector_stores", {}, raising=False)
    monkeypatch.setattr(VectorstoreSingleton, "_event_loops", {}, raising=False)

    mock_embed = MagicMock()
    connection_args = {"host": "localhost", "port": 19530}

    vs = singleton.get_vector_store("collection1", mock_embed, connection_args)

    assert vs is not None
    mock_milvus.assert_called_once()


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.get_vectorstore.Vectorstore")
def test_get_vectorstore_factory(mock_vectorstore_cls):
    """get_vectorstore should reuse or create Vectorstore."""
    mock_config = MagicMock()
    mock_config.milvus.collection_name = "demo"
    mock_config.milvus.embedding_dim = 768
    mock_embed = MagicMock()

    result1 = get_vectorstore(mock_embed, mock_config, force_new=True)
    assert result1 == mock_vectorstore_cls.return_value

    result2 = get_vectorstore(mock_embed, mock_config)
    assert result2 == result1


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.get_vectorstore.Vectorstore")
def test_get_vectorstore_force_new(mock_vectorstore_cls):
    """get_vectorstore should return a new instance if force_new=True."""
    mock_vs1 = MagicMock(name="Vectorstore1")
    mock_vs2 = MagicMock(name="Vectorstore2")
    mock_vectorstore_cls.side_effect = [mock_vs1, mock_vs2]

    dummy_config = MagicMock()
    dummy_config.milvus.collection_name = "my_test_collection"
    dummy_config.milvus.embedding_dim = 768

    vs1 = get_vectorstore(mock_vs1, dummy_config)
    vs2 = get_vectorstore(mock_vs2, dummy_config, force_new=True)

    assert vs1 is mock_vs1
    assert vs2 is mock_vs2
    assert vs1 != vs2


@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.connections.connect")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.connections.has_connection")
@patch("aiagents4pharma.talk2scholars.tools.pdf.utils.singleton_manager.db")
def test_get_connection_milvus_error(_, mock_has_connection, mock_connect, monkeypatch):
    """get_connection should raise MilvusException on connection failure."""
    manager = VectorstoreSingleton()

    # Reset connections cache safely
    monkeypatch.setattr(VectorstoreSingleton, "_connections", {}, raising=False)

    mock_has_connection.return_value = False
    mock_connect.side_effect = MilvusException("Connection failed")

    with pytest.raises(MilvusException, match="Connection failed"):
        manager.get_connection("localhost", 19530, "test_db")


def test_get_event_loop_creates_new_loop_on_closed(monkeypatch):
    """Ensure get_event_loop creates a new loop if current one is closed."""
    manager = VectorstoreSingleton()

    # Clear event loops safely
    monkeypatch.setattr(VectorstoreSingleton, "_event_loops", {}, raising=False)

    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = True

    with (
        patch("asyncio.get_event_loop", return_value=mock_loop),
        patch("asyncio.new_event_loop") as mock_new_loop,
        patch("asyncio.set_event_loop") as mock_set_loop,
    ):
        new_loop = MagicMock()
        mock_new_loop.return_value = new_loop

        result_loop = manager.get_event_loop()

        mock_new_loop.assert_called_once()
        mock_set_loop.assert_called_once_with(new_loop)
        assert result_loop == new_loop
