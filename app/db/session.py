"""数据库引擎与 Session 管理。"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.models import Base

_engine = None
_SessionLocal = None


def get_engine():
    """懒加载数据库引擎；SQLite 需关闭同线程检查。"""
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        connect_args = {}
        engine_kwargs: dict = {"echo": False, "connect_args": connect_args}
        if settings.database_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        else:
            engine_kwargs["pool_pre_ping"] = True
            engine_kwargs["pool_size"] = settings.db_pool_size
            engine_kwargs["max_overflow"] = settings.db_max_overflow
        _engine = create_engine(settings.database_url, **engine_kwargs)
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def init_db() -> None:
    """按模型创建缺失的数据表。"""
    engine = get_engine()
    Base.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    """FastAPI 依赖用的 Session 生成器，请求结束自动关闭。"""
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
    finally:
        session.close()


def session_scope() -> Session:
    """业务代码手动获取 Session，调用方负责关闭。"""
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()
