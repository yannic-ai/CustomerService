"""进程探活与依赖就绪检查。"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text


def check_liveness() -> dict[str, str]:
    """进程活着即可。"""
    return {"status": "ok"}


def check_readiness() -> dict[str, Any]:
    """探活 Redis（若已配置）、数据库与 FAISS 索引文件。"""
    from app.config import get_settings
    from app.db.session import session_scope
    from app.session_cache import RedisUnavailableError, get_redis_client

    settings = get_settings()
    checks: dict[str, str] = {}

    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error:{exc}"

    if settings.redis_url.strip():
        try:
            client = get_redis_client(required=True)
            if client is None:
                checks["redis"] = "error:client_missing"
            else:
                checks["redis"] = "ok"
        except RedisUnavailableError as exc:
            checks["redis"] = f"error:{exc}"
        except Exception as exc:
            checks["redis"] = f"error:{exc}"
    else:
        checks["redis"] = "skipped"

    from app.rag.vectorstore import faiss_index_status

    faiss_state = faiss_index_status(settings.faiss_index_path)
    checks["faiss"] = "ok" if faiss_state == "ok" else f"error:index_{faiss_state}"

    ready = all(value in {"ok", "skipped"} for value in checks.values())
    return {"status": "ok" if ready else "not_ready", "checks": checks}
