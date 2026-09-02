"""管理面知识库 ingest API。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Header, HTTPException, UploadFile

from app.config import KNOWLEDGE_DIR, get_settings
from app.rag.vectorstore import ingest_indexes
from app.schemas import KnowledgeIngestResponse
from app.tenancy import normalize_tenant_id

logger = logging.getLogger("cs.rag.ingest_api")

ADMIN_TOKEN_HEADER = "X-Admin-Token"
router = APIRouter(prefix="/admin/knowledge", tags=["admin"])


def _require_admin_token(
    x_admin_token: Annotated[str | None, Header(alias=ADMIN_TOKEN_HEADER)] = None,
) -> None:
    expected = get_settings().admin_ingest_token.strip()
    if not expected:
        raise HTTPException(status_code=503, detail="未配置 ADMIN_INGEST_TOKEN，ingest API 已禁用")
    if (x_admin_token or "").strip() != expected:
        raise HTTPException(status_code=401, detail="无效的管理员令牌")


def _safe_markdown_name(filename: str) -> str:
    name = Path(filename or "").name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=422, detail="文件名无效")
    if not name.lower().endswith(".md"):
        raise HTTPException(status_code=422, detail="仅支持 .md 文件")
    return name


def _target_dir(tenant_id: str) -> Path:
    settings = get_settings()
    tenant = normalize_tenant_id(tenant_id or settings.default_tenant)
    if tenant == settings.default_tenant:
        return KNOWLEDGE_DIR
    target = KNOWLEDGE_DIR / tenant
    target.mkdir(parents=True, exist_ok=True)
    return target


def _run_ingest(*, rebuild: bool) -> None:
    try:
        ingest_indexes(persist=True, rebuild=rebuild)
        logger.info("admin ingest completed rebuild=%s", rebuild)
    except Exception:
        logger.exception("admin ingest failed rebuild=%s", rebuild)
        raise


@router.post("/ingest", response_model=KnowledgeIngestResponse)
async def ingest_knowledge(
    background_tasks: BackgroundTasks,
    tenant_id: Annotated[str, Form()] = "",
    rebuild: Annotated[bool, Form()] = False,
    file: UploadFile | None = File(default=None),
    _: None = Depends(_require_admin_token),
) -> KnowledgeIngestResponse:
    """上传 Markdown 并异步触发增量 ingest；落盘后 bump generation 供多 worker 热加载。"""
    saved_path: str | None = None
    if file is not None and file.filename:
        target_dir = _target_dir(tenant_id)
        filename = _safe_markdown_name(file.filename)
        destination = target_dir / filename
        body = await file.read()
        if not body.strip():
            raise HTTPException(status_code=422, detail="文件内容为空")
        destination.write_bytes(body)
        saved_path = str(destination.relative_to(KNOWLEDGE_DIR))

    background_tasks.add_task(_run_ingest, rebuild=rebuild)
    tenant = normalize_tenant_id(tenant_id or get_settings().default_tenant)
    return KnowledgeIngestResponse(
        status="accepted",
        tenant_id=tenant,
        rebuild=rebuild,
        saved_path=saved_path,
        message="ingest 已在后台启动，完成后各 worker 将自动热加载索引",
    )
