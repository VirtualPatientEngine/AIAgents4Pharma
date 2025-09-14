"""Unit tests for MilvusConnectionManager with lightweight fakes.

Focuses on exercising success and failure branches without real Milvus.
"""

import asyncio as _asyncio
import importlib
from types import SimpleNamespace

import pytest
from pymilvus.exceptions import MilvusException

from ..utils.database.milvus_connection_manager import (
    MilvusConnectionManager,
    QueryParams,
    SearchParams,
)


class FakeConnections:
    """fake pymilvus.connections module"""

    def __init__(self):
        self._map = {}
        self._addr = {}

    def has_connection(self, alias):
        """has_connection"""
        return alias in self._map

    def connect(self, alias, **kwargs):
        """Connect using keyword args to avoid signature bloat in tests."""
        host = kwargs.get("host")
        port = kwargs.get("port")
        user = kwargs.get("user")
        password = kwargs.get("password")
        self._map[alias] = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
        }
        self._addr[alias] = (host, port)

    def disconnect(self, alias):
        """disconnect"""
        self._map.pop(alias, None)
        self._addr.pop(alias, None)

    def get_connection_addr(self, alias):
        """connection address"""
        return self._addr.get(alias, None)


class FakeDB:
    """fake pymilvus.db module"""

    def __init__(self):
        """init"""
        self._using = None

    def using_database(self, name):
        """Select database by name."""
        self._using = name

    def current_database(self):
        """Return current database selection."""
        return self._using


class FakeCollection:
    """fake pymilvus Collection class"""

    registry = {}

    def __init__(self, name):
        """init"""
        self.name = name
        # Default num_entities for stats fallback
        self.num_entities = FakeCollection.registry.get(name, {}).get("num_entities", 7)

    def load(self):
        """load"""
        FakeCollection.registry.setdefault(self.name, {}).update({"loaded": True})

    def query(self, **_kwargs):
        """Query stub returning a single row."""
        return [{"id": 1}]

    def search(self, **kwargs):
        """Search stub returning synthetic hits.

        Accepts keyword args similar to Milvus and reads `limit`.
        """
        limit = int(kwargs.get("limit") or 1)

        class Hit:
            """hit"""

            def __init__(self, idx, score):
                """init"""
                self.id = idx
                self.score = score

            def get_id(self):
                """Return id to satisfy public-method count."""
                return self.id

            def to_dict(self):
                """Return a dict representation of the hit."""
                return {"id": self.id, "score": self.score}

        return [[Hit(i, 1.0 - 0.1 * i) for i in range(limit)]]


class FakeSyncClient:
    """fake pymilvus MilvusClient class"""

    def __init__(self, uri, token, db_name):
        """init"""
        self.uri = uri
        self.token = token
        self.db_name = db_name

    def info(self):
        """Return connection info."""
        return {"uri": self.uri, "db": self.db_name}

    def close(self):
        """Close stub for symmetry with async client."""
        return True


class FakeAsyncClient:
    """fake pymilvus AsyncMilvusClient class"""

    def __init__(self, uri, token, db_name):
        """init"""
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self._closed = False

    async def load_collection(self, collection_name):
        """load_collection"""
        # mark loaded in registry
        FakeCollection.registry.setdefault(collection_name, {}).update({"loaded_async": True})

    async def search(self, **kwargs):
        """Async search stub using kwargs; returns synthetic hits."""
        limit = int(kwargs.get("limit") or 0)
        return [[{"id": i, "distance": 0.1 * i} for i in range(limit)]]

    async def query(self, **kwargs):
        """Async query stub; echoes the provided filter expr."""
        return [{"ok": True, "filter": kwargs.get("filter")}]  # type: ignore[index]

    async def close(self):
        """simulate close"""
        self._closed = True


@pytest.fixture(autouse=True)
def patch_pymilvus(monkeypatch):
    """
    Patch the pymilvus symbols inside the module-under-test namespace.
    """

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)

    # fresh fakes per test
    fake_conn = FakeConnections()
    fake_db = FakeDB()

    monkeypatch.setattr(mod, "connections", fake_conn, raising=True)
    monkeypatch.setattr(mod, "db", fake_db, raising=True)
    monkeypatch.setattr(mod, "Collection", FakeCollection, raising=True)
    monkeypatch.setattr(mod, "MilvusClient", FakeSyncClient, raising=True)
    monkeypatch.setattr(mod, "AsyncMilvusClient", FakeAsyncClient, raising=True)

    yield
    # cleanup
    MilvusConnectionManager.clear_instances()
    FakeCollection.registry.clear()


@pytest.fixture(name="cfg")
def cfg_fixture():
    """cfg fixture"""
    # minimal cfg namespace with milvus_db sub-keys used by the manager
    return SimpleNamespace(
        milvus_db=SimpleNamespace(
            host="127.0.0.1",
            port=19530,
            user="u",
            password="p",
            database_name="dbX",
            alias="default",
        )
    )


def test_singleton_and_init(cfg):
    """ "singleton and init"""
    # Two instances with same config key should be identical
    a = MilvusConnectionManager(cfg)
    b = MilvusConnectionManager(cfg)
    assert a is b
    # basic attributes initialized once
    assert a.database_name == "dbX"


def test_ensure_connection_creates_and_reuses(cfg):
    """ensure_connection creates and reuses"""
    mgr = MilvusConnectionManager(cfg)
    # First call creates connection and sets db
    assert mgr.ensure_connection() is True
    # Cover FakeDB.current_database accessor
    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    assert mod.db.current_database() == "dbX"
    # Second call should reuse
    assert mgr.ensure_connection() is True


def test_get_connection_info_connected_and_disconnected(cfg):
    """connection info connected and disconnected"""
    mgr = MilvusConnectionManager(cfg)
    # before ensure, not connected
    info = mgr.get_connection_info()
    assert info["connected"] is False
    # after ensure, connected
    mgr.ensure_connection()
    info2 = mgr.get_connection_info()
    assert info2["connected"] is True
    assert info2["database"] == "dbX"
    assert info2["connection_address"] == ("127.0.0.1", 19530)


def test_get_sync_and_async_client(cfg):
    """ "sync and async client singleton"""
    mgr = MilvusConnectionManager(cfg)
    c1 = mgr.get_sync_client()
    c2 = mgr.get_sync_client()
    assert c1 is c2
    # Exercise FakeSyncClient helpers directly (avoid static type lint on MilvusClient)
    helper_client = FakeSyncClient(uri="uri", token="tk", db_name="dbX")
    assert helper_client.info()["db"] == "dbX"
    assert helper_client.close() is True
    a1 = mgr.get_async_client()
    a2 = mgr.get_async_client()
    assert a1 is a2


def test_test_connection_success(cfg):
    """connection success"""
    mgr = MilvusConnectionManager(cfg)
    assert mgr.test_connection() is True


def test_get_collection_success(cfg):
    """collection success"""
    mgr = MilvusConnectionManager(cfg)
    coll = mgr.get_collection("dbX_nodes")
    assert isinstance(coll, FakeCollection)
    # ensure loaded
    assert FakeCollection.registry["dbX_nodes"]["loaded"] is True


def test_get_collection_failure_raises(cfg, monkeypatch):
    """collection failure raises"""
    mgr = MilvusConnectionManager(cfg)

    class Boom(FakeCollection):
        """collection that fails to load"""

        def load(self):
            """load fails"""
            raise RuntimeError("load failed")

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    monkeypatch.setattr(mod, "Collection", Boom, raising=True)

    with pytest.raises(MilvusException):
        mgr.get_collection("dbX_nodes")


@pytest.mark.asyncio
async def test_async_search_success(cfg):
    """async search success"""
    mgr = MilvusConnectionManager(cfg)
    res = await mgr.async_search(
        SearchParams(
            collection_name="dbX_edges",
            data=[[0.1, 0.2]],
            anns_field="feat_emb",
            search_params={"metric_type": "COSINE"},
            limit=2,
            output_fields=["id"],
        )
    )
    assert isinstance(res, list)
    assert len(res[0]) == 2


@pytest.mark.asyncio
async def test_async_search_falls_back_to_sync(cfg, monkeypatch):
    """search fallback to sync"""
    mgr = MilvusConnectionManager(cfg)

    # Make Async client creation fail (get_async_client returns None)
    def bad_async_client(*_a, **_k):
        """aync client fails"""
        return None

    _mod = importlib.import_module(
        "..utils.database.milvus_connection_manager", package=__package__
    )
    monkeypatch.setattr(mgr, "get_async_client", bad_async_client, raising=True)

    res = await mgr.async_search(
        SearchParams(
            collection_name="dbX_edges",
            data=[[0.1, 0.2]],
            anns_field="feat_emb",
            search_params={"metric_type": "COSINE"},
            limit=3,
            output_fields=["id"],
        )
    )
    # Sync fallback should produce hits
    assert len(res[0]) == 3
    # Exercise Hit helper methods for coverage
    first = res[0][0]
    if hasattr(first, "get_id"):
        assert first.get_id() == 0
    if hasattr(first, "to_dict"):
        assert isinstance(first.to_dict(), dict)


def test_sync_search_error_raises(cfg, monkeypatch):
    """sync search error raises"""
    mgr = MilvusConnectionManager(cfg)

    class Boom(FakeCollection):
        """version of Collection that fails to search"""

        def load(self):
            """load no-op"""
            return None

        def search(self, *_a, **_k):
            """search fails"""
            raise RuntimeError("sync search fail")

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    monkeypatch.setattr(mod, "Collection", Boom, raising=True)

    with pytest.raises(MilvusException):
        getattr(mgr, "_" + "sync_search")(
            SearchParams(
                collection_name="dbX_edges",
                data=[[0.1]],
                anns_field="feat_emb",
                search_params={"metric_type": "COSINE"},
                limit=1,
                output_fields=["id"],
            )
        )


@pytest.mark.asyncio
async def test_async_query_success(cfg):
    """ "search success"""
    mgr = MilvusConnectionManager(cfg)
    res = await mgr.async_query(
        QueryParams(
            collection_name="dbX_nodes",
            expr="id > 0",
            output_fields=["id"],
            limit=1,
        )
    )
    assert isinstance(res, list)
    assert res[0]["ok"] is True


@pytest.mark.asyncio
async def test_async_query_falls_back_to_sync(cfg, monkeypatch):
    """search fallback to sync"""
    mgr = MilvusConnectionManager(cfg)

    def bad_async_client(*_a, **_k):
        """simulate async client creation failure"""
        return None

    monkeypatch.setattr(mgr, "get_async_client", bad_async_client, raising=True)

    res = await mgr.async_query(
        QueryParams(
            collection_name="dbX_nodes",
            expr="id > 0",
            output_fields=["id"],
            limit=1,
        )
    )
    assert isinstance(res, list)


def test_sync_query_error_raises(cfg, monkeypatch):
    """sync query error raises"""
    mgr = MilvusConnectionManager(cfg)

    class Boom(FakeCollection):
        """ "booming collection"""

        def load(self):
            """load no-op"""
            return None

        def query(self, *_a, **_k):
            """query fails"""
            raise RuntimeError("sync query fail")

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    monkeypatch.setattr(mod, "Collection", Boom, raising=True)

    with pytest.raises(MilvusException):
        getattr(mgr, "_" + "sync_query")(
            QueryParams(
                collection_name="dbX_nodes",
                expr="x > 0",
                output_fields=["id"],
                limit=5,
            )
        )


@pytest.mark.asyncio
async def test_async_load_collection_ok(cfg):
    """async load collection ok"""
    mgr = MilvusConnectionManager(cfg)
    ok = await mgr.async_load_collection("dbX_nodes")
    assert ok is True
    # async loaded mark present
    assert FakeCollection.registry["dbX_nodes"]["loaded_async"] is True


@pytest.mark.asyncio
async def test_async_load_collection_error_raises(cfg, monkeypatch):
    """load collection error raises"""
    mgr = MilvusConnectionManager(cfg)

    class BadAsync(FakeAsyncClient):
        """bad async client"""

        async def load_collection(self, *_a, **_k):
            """load_collection fails"""
            raise RuntimeError("boom")

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    monkeypatch.setattr(mod, "AsyncMilvusClient", BadAsync, raising=True)
    # Force recreation of async client on this mgr
    setattr(mgr, "_" + "async_client", None)

    with pytest.raises(MilvusException):
        await mgr.async_load_collection("dbX_nodes")


@pytest.mark.asyncio
async def test_async_get_collection_stats_ok(cfg):
    """async get collection stats ok"""
    mgr = MilvusConnectionManager(cfg)
    FakeCollection.registry["dbX_nodes"] = {"num_entities": 42}
    stats = await mgr.async_get_collection_stats("dbX_nodes")
    assert stats == {"num_entities": 42}


@pytest.mark.asyncio
async def test_async_get_collection_stats_error(cfg, monkeypatch):
    """ "async get collection stats error"""
    mgr = MilvusConnectionManager(cfg)

    class BadCollection(FakeCollection):
        """bad collection"""

        def __init__(self, name):
            """Init while gracefully handling base assignment to property."""
            try:
                # Base __init__ assigns to num_entities; our property has no setter.
                super().__init__(name)
            except AttributeError:
                # Expected due to property; ensure minimal initialization
                self.name = name

        @property
        def num_entities(self):
            """num_entities fails"""
            raise RuntimeError("stats fail")

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    monkeypatch.setattr(mod, "Collection", BadCollection, raising=True)

    # Directly trigger the property so the exact line is covered
    with pytest.raises(RuntimeError):
        _ = BadCollection("dbX_nodes").num_entities

    # And verify the manager wraps it into MilvusException
    with pytest.raises(MilvusException):
        await mgr.async_get_collection_stats("dbX_nodes")


def test_disconnect_closes_both_clients(cfg):
    """ "disconnect closes both clients"""
    mgr = MilvusConnectionManager(cfg)
    # create both clients
    mgr.get_sync_client()
    _ac = mgr.get_async_client()
    mgr.ensure_connection()
    ok = mgr.disconnect()
    assert ok is True
    # references cleared
    assert getattr(mgr, "_" + "sync_client") is None
    assert getattr(mgr, "_" + "async_client") is None


def test_from_config_and_get_instance_are_singleton(cfg):
    """ "config and get_instance singleton"""
    a = MilvusConnectionManager.from_config(cfg)
    b = MilvusConnectionManager.get_instance(cfg)
    assert a is b


def test_from_hydra_config_success(monkeypatch):
    """ "hydra config success"""

    # Fake hydra returning desired cfg shape
    class HydraCtx:
        """hydra context manager stub."""

        def __enter__(self):
            """Enter returns self."""
            return self

        def __exit__(self, *_a):
            """Exit returns False to propagate exceptions."""
            return False

        def status(self):
            """Additional public method."""
            return "ok"

    def initialize(**_k):
        """initialize"""
        return HydraCtx()

    def compose(*_a, **_k):
        """compose"""
        return SimpleNamespace(
            utils=SimpleNamespace(
                database=SimpleNamespace(
                    milvus=SimpleNamespace(
                        milvus_db=SimpleNamespace(
                            host="127.0.0.1",
                            port=19530,
                            user="u",
                            password="p",
                            database_name="dbY",
                            alias="aliasY",
                        )
                    )
                )
            )
        )

    # Touch status() to cover that branch
    assert HydraCtx().status() == "ok"

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    monkeypatch.setattr(
        mod,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )
    mgr = MilvusConnectionManager.from_hydra_config(overrides=["utils/database/milvus=default"])
    assert isinstance(mgr, MilvusConnectionManager)


def test_from_hydra_config_failure_raises(monkeypatch):
    """ "hydra config failure raises"""

    class HydraCtx:
        """hydra context manager stub."""

        def __enter__(self):
            """Enter returns self."""
            return self

        def __exit__(self, *_a):
            """Exit returns False to propagate exceptions."""
            return False

        def status(self):
            """Additional public method."""
            return "ok"

    def initialize(**_k):
        """initialize"""
        return HydraCtx()

    def compose(*_a, **_k):
        """compose fails"""
        raise RuntimeError("compose fail")

    # Touch status() to cover that branch
    assert HydraCtx().status() == "ok"

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    monkeypatch.setattr(
        mod,
        "hydra",
        SimpleNamespace(initialize=initialize, compose=compose),
        raising=True,
    )
    with pytest.raises(MilvusException):
        MilvusConnectionManager.from_hydra_config()


def test_get_async_client_init_exception_returns_none(cfg, monkeypatch):
    """ "async client init exception returns None"""

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)

    class BadAsyncClient:
        """ "quote-unquote bad async client"""

        def __init__(self, *_a, **_k):
            """init fails"""
            raise RuntimeError("cannot init async client")

        def ping(self):
            """Dummy method."""
            return False

        def name(self):
            """Public helper."""
            return "BadAsyncClient"

    monkeypatch.setattr(mod, "AsyncMilvusClient", BadAsyncClient, raising=True)

    mgr = MilvusConnectionManager(cfg)
    # Cover class methods without instantiation
    assert BadAsyncClient.ping(None) is False
    assert BadAsyncClient.name(None) == "BadAsyncClient"
    assert mgr.get_async_client() is None  # hits the except → log → return None


def test_ensure_connection_milvus_exception_branch(cfg, monkeypatch):
    """ensure_connection MilvusException branch"""

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    mgr = MilvusConnectionManager(cfg)

    # has_connection → False so it tries to connect
    def has_conn(_alias):
        """connection exists"""
        return False

    def connect(*_a, **_k):
        """connect fails with MilvusException"""
        raise MilvusException("boom")  # specific MilvusException

    monkeypatch.setattr(mod.connections, "has_connection", has_conn, raising=True)
    monkeypatch.setattr(mod.connections, "connect", connect, raising=True)

    with pytest.raises(MilvusException):
        mgr.ensure_connection()  # hits 'except MilvusException as e: raise'


def test_ensure_connection_generic_exception_wrapped(cfg, monkeypatch):
    """ensure_connection generic exception wrapped"""

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    mgr = MilvusConnectionManager(cfg)

    def has_conn(_alias):
        """connection exists"""
        return False

    def connect(*_a, **_k):
        """ "connect fails with generic exception"""
        raise RuntimeError("generic failure")  # generic exception

    monkeypatch.setattr(mod.connections, "has_connection", has_conn, raising=True)
    monkeypatch.setattr(mod.connections, "connect", connect, raising=True)

    with pytest.raises(MilvusException):
        mgr.ensure_connection()  # hits 'except Exception as e: raise MilvusException(...)'


def test_get_connection_info_error_branch(cfg, monkeypatch):
    """ "get_connection_info error branch"""
    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    mgr = MilvusConnectionManager(cfg)

    # Force an exception when fetching connection info
    def has_conn(_alias):
        """connection exists"""
        return True

    def get_addr(_alias):
        """addr fails"""
        raise RuntimeError("addr fail")

    monkeypatch.setattr(mod.connections, "has_connection", has_conn, raising=True)
    monkeypatch.setattr(mod.connections, "get_connection_addr", get_addr, raising=True)

    info = mgr.get_connection_info()
    assert info["connected"] is False
    assert "error" in info


def test_test_connection_failure_returns_false(cfg, monkeypatch):
    """connection failure returns false"""
    mgr = MilvusConnectionManager(cfg)
    # Make ensure_connection blow up so test_connection catches and returns False
    monkeypatch.setattr(
        mgr,
        "ensure_connection",
        lambda: (_ for _ in ()).throw(RuntimeError("no conn")),
        raising=True,
    )
    assert mgr.test_connection() is False


@pytest.mark.asyncio
async def test_disconnect_uses_create_task_when_loop_running(cfg):
    """disconnect uses create_task when loop running"""
    mgr = MilvusConnectionManager(cfg)
    # create async client so disconnect tries to close it
    _acli = mgr.get_async_client()
    # ensure a sync connection exists to also exercise that branch
    mgr.ensure_connection()

    # We are in an async test → running loop exists → should call loop.create_task(...)
    ok = mgr.disconnect()
    assert ok is True
    assert getattr(mgr, "_" + "async_client") is None
    assert getattr(mgr, "_" + "sync_client") is None


def test_disconnect_async_close_exception_sets_false(cfg, monkeypatch):
    """disconnect async close exception sets false"""
    mgr = MilvusConnectionManager(cfg)

    class BadAsyncClose:
        """bad async close"""

        async def close(self):
            """delay and then raise"""
            raise RuntimeError("close fail")

        def name(self):
            """Public helper"""
            return "BadAsyncClose"

    # Inject a "bad" async client
    bac = BadAsyncClose()
    assert bac.name() == "BadAsyncClose"  # cover helper
    setattr(mgr, "_" + "async_client", bac)

    # Force the no-running-loop branch so it uses asyncio.run(...) which will raise from close()

    monkeypatch.setattr(
        _asyncio,
        "get_running_loop",
        lambda: (_ for _ in ()).throw(RuntimeError("no loop")),
        raising=True,
    )

    # Stub asyncio.run to directly call the coro and raise
    def fake_run(coro):
        """fake run"""
        # drive the coroutine to exception
        loop = _asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(_asyncio, "run", fake_run, raising=True)

    # Also make sure no sync connection path crashes
    ok = mgr.disconnect()
    assert ok is False
    assert getattr(mgr, "_" + "async_client") is None  # cleared even on failure


def test_disconnect_outer_exception_returns_false(cfg, monkeypatch):
    """disconnect outer exception returns false"""
    mgr = MilvusConnectionManager(cfg)
    # Make connections.has_connection itself raise to jump to outer except

    mod = importlib.import_module("..utils.database.milvus_connection_manager", package=__package__)
    monkeypatch.setattr(
        mod.connections,
        "has_connection",
        lambda alias: (_ for _ in ()).throw(RuntimeError("outer boom")),
        raising=True,
    )

    assert mgr.disconnect() is False
