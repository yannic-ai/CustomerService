"""课程主数据查询：用 ``(tenant_id, code)`` 对齐订单课名与知识库文件。"""

from __future__ import annotations

import logging

from sqlalchemy import select

from app.db.models import Course
from app.db.session import session_scope

logger = logging.getLogger("cs.db.courses")


def load_course_catalog(tenant_id: str | None = None) -> dict[tuple[str, str], str]:
    """``{(tenant_id, code): name}``。库不可用时返回空字典，不阻断 ingest。"""
    try:
        with session_scope() as session:
            stmt = select(Course.tenant_id, Course.code, Course.name)
            if tenant_id:
                stmt = stmt.where(Course.tenant_id == tenant_id)
            return {
                (row.tenant_id, row.code): row.name
                for row in session.execute(stmt).all()
            }
    except Exception:
        logger.debug("course catalog unavailable", exc_info=True)
        return {}


def resolve_course_code(
    tenant_id: str,
    *,
    code: str | None = None,
    name: str | None = None,
) -> str | None:
    """有订单时用 ``code``，不要靠展示名反查。无名无码则返回 None。"""
    stripped_code = (code or "").strip()
    if stripped_code:
        return stripped_code
    stripped_name = (name or "").strip()
    if not stripped_name:
        return None
    catalog = load_course_catalog(tenant_id)
    matches = [
        course_code
        for (tenant, course_code), course_name in catalog.items()
        if tenant == tenant_id and course_name == stripped_name
    ]
    return matches[0] if matches else None
