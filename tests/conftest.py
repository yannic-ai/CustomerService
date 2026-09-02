import os
from collections.abc import Iterator

import pytest

from app.config import get_settings
from app.db.models import Base
from app.db.seed import seed_if_empty
from app.db.session import get_engine, init_db, session_scope
from app.rag.embeddings import get_embeddings
from app.rag.vectorstore import build_faiss_index, reset_indexes
from app.session_cache import reset_redis_client
from app.tenancy import set_current_user


@pytest.fixture(scope="session", autouse=True)
def isolate_test_runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """默认离线：ngram + 临时 FAISS；未设 RUN_LLM_EVAL=1 时清空 API Key。"""
    settings = get_settings()
    original = {
        "redis_url": settings.redis_url,
        "embedding_backend": settings.embedding_backend,
        "embedding_dim": settings.embedding_dim,
        "faiss_index_path": settings.faiss_index_path,
        "deepseek_api_key": settings.deepseek_api_key,
    }
    settings.redis_url = ""
    settings.embedding_backend = "ngram"
    settings.embedding_dim = 384
    settings.faiss_index_path = str(tmp_path_factory.mktemp("faiss"))
    settings.prometheus_multiproc_dir = ""
    os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
    if os.environ.get("RUN_LLM_EVAL") != "1":
        settings.deepseek_api_key = ""
    reset_redis_client()
    get_embeddings.cache_clear()
    yield
    settings.redis_url = original["redis_url"]
    settings.embedding_backend = original["embedding_backend"]
    settings.embedding_dim = original["embedding_dim"]
    settings.faiss_index_path = original["faiss_index_path"]
    settings.deepseek_api_key = original["deepseek_api_key"]
    get_embeddings.cache_clear()
    reset_redis_client()


@pytest.fixture(scope="session", autouse=True)
def bootstrap_app(isolate_test_runtime: None) -> None:
    engine = get_engine()
    Base.metadata.drop_all(engine)
    init_db()
    with session_scope() as session:
        seed_if_empty(session)
    reset_indexes()
    build_faiss_index(persist=True)


@pytest.fixture(autouse=True)
def reset_current_user() -> Iterator[None]:
    set_current_user("anonymous")
    yield
    set_current_user("anonymous")


@pytest.fixture(autouse=True)
def reset_chat_slots() -> Iterator[None]:
    from app.chat_slots import reset_chat_slot_store

    reset_chat_slot_store()
    yield
    reset_chat_slot_store()
