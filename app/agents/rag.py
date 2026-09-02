"""RAG Agent：检索课程大纲并输出模块 + 学习路径。"""

from __future__ import annotations

import json
import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.config import get_settings
from app.context import expand_query, format_history_for_prompt
from app.evals.harvest import record_retrieval_miss
from app.rag.catalog import enrich_query_with_course_aliases
from app.observability import inc_rag_empty, inc_tool_call, log_event, set_span_attributes
from app.observability.langfuse import PROMPT_RAG_MARKDOWN, PROMPT_RAG_STRUCTURED, get_text_prompt
from app.observability.otel import start_span
from app.rag.llm_rewrite import decompose_retrieval_queries
from app.rag.rewrite import RewriteResult, maybe_rewrite_query
from app.rag.vectorstore import (
    expand_hit_documents_scored,
    format_rag_context_chunk,
    get_indexes,
    iter_tenant_documents,
    search_scopes,
    search_tenant_documents,
    trim_rag_context_by_budget,
)
from app.schemas import CourseConsultResult, CourseModule, EvidenceSection, LearningStage
from app.tenancy import get_current_tenant

logger = logging.getLogger("cs.rag")

NOT_FOUND_COURSE = "未找到课程"
NOT_FOUND_SUMMARY = "知识库中没有匹配内容，请换个课程名称再试。"
_EVIDENCE_EXCERPT_CHARS = 120

RAG_STRUCTURED_SYSTEM = (
    "你是课程咨询助手。只根据检索到的大纲作答，不要编造。"
    "输出课程模块列表和学习路径（入门/巩固/实战）。"
    "可结合对话历史理解指代，但答案必须以检索结果为准。"
)

RAG_MARKDOWN_SYSTEM = (
    "你是课程咨询助手。只根据检索到的大纲作答，不要编造。"
    "用 Markdown 回复：先写课程标题与摘要，再列课程模块，再写学习路径（入门/巩固/实战）。"
    "可结合对话历史理解指代，但答案必须以检索结果为准。"
)


@dataclass(frozen=True)
class SearchQueryContext:
    """最近一次检索 query 管道结果，供 harvest / 日志使用。"""

    original_query: str
    search_query: str
    rewrite_applied: bool
    rewrite_reason: str
    retrieval_queries: tuple[str, ...] = ()


_search_query_ctx: ContextVar[SearchQueryContext | None] = ContextVar("search_query_ctx", default=None)


def get_last_search_query_context() -> SearchQueryContext | None:
    return _search_query_ctx.get()


def build_evidence_from_docs(docs: list[Document]) -> list[EvidenceSection]:
    """从检索文档构建章节级 evidence，按 (source, section_path) 去重。"""
    seen: set[tuple[str, str]] = set()
    evidence: list[EvidenceSection] = []
    for doc in docs:
        source = str(doc.metadata.get("source") or "").strip()
        section_path = str(doc.metadata.get("section_path") or "").strip()
        module_name = str(doc.metadata.get("module_name") or "").strip()
        key = (source, section_path or module_name)
        if key in seen:
            continue
        seen.add(key)
        excerpt = (doc.page_content or "").replace("\n", " ").strip()
        if len(excerpt) > _EVIDENCE_EXCERPT_CHARS:
            excerpt = excerpt[:_EVIDENCE_EXCERPT_CHARS] + "…"
        evidence.append(
            EvidenceSection(
                source=source,
                section_path=section_path,
                module_name=module_name,
                excerpt=excerpt,
            )
        )
    return evidence


def _with_evidence(result: CourseConsultResult, docs: list[Document]) -> CourseConsultResult:
    result.evidence = build_evidence_from_docs(docs)
    return result


def _prepare_search_query(
    query: str,
    history: list[BaseMessage] | None,
    *,
    tenant_id: str,
    last_course_query: str | None = None,
    last_course_name: str | None = None,
) -> str:
    expanded = expand_query(
        query,
        history,
        last_course_query=last_course_query,
        last_course_name=last_course_name,
    )
    settings = get_settings()
    rewrite: RewriteResult
    if settings.rag_rewrite_enabled:
        rewrite = maybe_rewrite_query(
            expanded,
            original_query=query,
            last_course_name=last_course_name,
            last_course_query=last_course_query,
        )
    else:
        rewrite = RewriteResult(query=expanded, rewritten=False, reason="none")
    final = enrich_query_with_course_aliases(rewrite.query, tenant_id=tenant_id)
    retrieval_queries = tuple(decompose_retrieval_queries(final))
    _search_query_ctx.set(
        SearchQueryContext(
            original_query=query,
            search_query=final,
            rewrite_applied=rewrite.rewritten,
            rewrite_reason=rewrite.reason,
            retrieval_queries=retrieval_queries,
        )
    )
    return final


def get_store(tenant_id: str | None = None):
    """当前租户的 FAISS 子索引；没有则返回 None。"""
    tenant = tenant_id or get_current_tenant()
    return get_indexes().get(tenant)


def _not_found_result() -> CourseConsultResult:
    return CourseConsultResult(course_name=NOT_FOUND_COURSE, summary=NOT_FOUND_SUMMARY)


def retrieve_course_docs(
    query: str,
    k: int = 8,
    tenant_id: str | None = None,
    *,
    course_code: str | None = None,
) -> list[Document]:
    """在租户（及平台）子索引内检索；低分当空召回；命中后只补同节与相邻切片。

    绑定 ``course_code`` 时按大纲文件名过滤，展示名对不齐也不会空召回。
    每次调用记 ``cs_tool_calls_total{tool=course_retrieve,outcome=ok|not_found}``。
    """
    tenant = tenant_id or get_current_tenant()
    with start_span("cs.rag.retrieve", **{"cs.node": "rag.retrieve", "tenant.id": tenant}):
        search_ctx = get_last_search_query_context()
        retrieval_queries: list[str] | None = None
        if search_ctx and search_ctx.retrieval_queries and search_ctx.search_query == query:
            retrieval_queries = list(search_ctx.retrieval_queries)
        else:
            decomposed = decompose_retrieval_queries(query)
            if len(decomposed) > 1:
                retrieval_queries = decomposed
        search_result = search_tenant_documents(
            query,
            tenant,
            k=k,
            course_code=course_code,
            retrieval_queries=retrieval_queries,
        )
        ranked = search_result.hits
        settings = get_settings()
        min_score = settings.rag_min_score
        best = ranked[0] if ranked else None
        hits = [hit.document for hit in ranked if hit.passes_min_score(min_score)][:k]
        low_confidence = any(hit.low_confidence for hit in ranked[:k])
        if not hits and course_code and ranked:
            # 绑课编码时允许低分/低置信候选，但禁止无排序全量 dump
            hits = [hit.document for hit in ranked if hit.low_confidence or hit.passes_min_score(0)][:k]
            low_confidence = low_confidence or search_result.low_confidence_fallback
        rejected_low_score = bool(ranked) and not hits
        corpus = iter_tenant_documents(tenant, course_code=course_code)
        scored = (
            expand_hit_documents_scored(
                hits,
                corpus,
                neighbor=settings.rag_neighbor_chunks,
            )
            if hits
            else []
        )
        expanded, trim_stats = trim_rag_context_by_budget(
            scored,
            settings.rag_context_token_budget,
        )
        log_event(
            logger,
            "rag_retrieve",
            hit_count=len(hits),
            expanded_count=len(scored),
            trimmed_count=len(expanded),
            k=k,
            empty=not expanded,
            query_chars=len(query or ""),
            search_query_chars=len((search_ctx.search_query if search_ctx else query) or ""),
            rewrite_applied=bool(search_ctx and search_ctx.rewrite_applied),
            rewrite_reason=(search_ctx.rewrite_reason if search_ctx else "none"),
            scopes=search_scopes(tenant, course_code=course_code),
            course_code=course_code or "",
            best_score=None if best is None else round(best.score, 4),
            best_vector_sim=None if best is None else round(best.vector_sim, 4),
            best_bm25_score=None if best is None else round(best.bm25_score, 4),
            min_score=min_score,
            hybrid=settings.rag_hybrid_enabled,
            neighbor_chunks=settings.rag_neighbor_chunks,
            rejected_low_score=rejected_low_score,
            context_tokens=trim_stats["context_tokens"],
            context_budget=settings.rag_context_token_budget,
            context_trimmed=trim_stats["context_trimmed"],
            context_budget_exceeded=trim_stats["context_budget_exceeded"],
            fetch_k=search_result.fetch_k,
            query_profile=search_result.query_profile,
            fused_count=search_result.fused_count,
            rerank_enabled=search_result.rerank_enabled,
            rerank_top_k=search_result.rerank_top_k,
            low_confidence=low_confidence,
            low_confidence_fallback=search_result.low_confidence_fallback,
            retrieval_queries=list(search_result.retrieval_queries),
        )
        set_span_attributes(
            **{
                "cs.rag.hit_count": str(len(hits)),
                "cs.rag.empty": "true" if not expanded else "false",
                "cs.rag.rejected_low_score": "true" if rejected_low_score else "false",
                "cs.rag.context_tokens": str(trim_stats["context_tokens"]),
                "cs.rag.context_trimmed": str(trim_stats["context_trimmed"]),
                "cs.rag.fetch_k": str(search_result.fetch_k),
                "cs.rag.query_profile": search_result.query_profile,
                "cs.rag.rerank_enabled": "true" if search_result.rerank_enabled else "false",
                "cs.rag.low_confidence": "true" if low_confidence else "false",
            }
        )
        if not hits:
            inc_rag_empty(tenant, reason="low_score" if rejected_low_score else "empty")
            # 与查单 cs_tool_calls_total 对等：空召回 / 低分拒答都记 not_found
            inc_tool_call("course_retrieve", "not_found", tenant)
            miss_reason = "low_score" if rejected_low_score else "rag_empty"
            record_retrieval_miss(
                tenant_id=tenant,
                query=(search_ctx.original_query if search_ctx else query) or query,
                search_query=(search_ctx.search_query if search_ctx else query) or query,
                reason=miss_reason,
                rewrite_reason=(search_ctx.rewrite_reason if search_ctx else ""),
            )
            return []
        inc_tool_call("course_retrieve", "ok", tenant)
        return expanded


_COURSE_NAME_RE = re.compile(r"课程名称[：:]\s*(.+)")


def _course_code_query_overlap(query: str, docs: list[Document], course_code: str) -> int:
    """问句与某课召回文本的重叠度，用于多数派并列时的 tie-break。"""
    from app.rag.lexical import tokenize
    from app.rag.vectorstore import document_course_code

    tokens = [token for token in tokenize(query) if len(token) >= 2]
    if not tokens:
        return 0
    score = 0
    for doc in docs:
        if document_course_code(doc) != course_code:
            continue
        text = (doc.page_content or "").lower()
        score += sum(1 for token in tokens if token in text)
    return score


def _course_title_from_docs(query: str, docs: list[Document], *, default: str = "相关课程") -> str:
    """从检索切片抽课名；问句里出现过的课名优先，否则按召回文档多数派 course_code。"""
    from collections import Counter

    from app.rag.vectorstore import document_course_code

    catalog_names: list[str] = []
    for doc in docs:
        stamped = str(doc.metadata.get("course_name") or "").strip()
        if stamped and stamped not in catalog_names:
            catalog_names.append(stamped)
    query_text = query or ""
    for name in catalog_names:
        if name and name in query_text:
            return name

    code_counts = Counter(
        document_course_code(doc) for doc in docs if document_course_code(doc)
    )
    if code_counts:
        ranked_codes = sorted(
            code_counts.items(),
            key=lambda item: (-item[1], -_course_code_query_overlap(query, docs, item[0])),
        )
        dominant_code = ranked_codes[0][0]
        for doc in docs:
            if document_course_code(doc) != dominant_code:
                continue
            stamped = str(doc.metadata.get("course_name") or "").strip()
            if stamped:
                return stamped
        from app.db.courses import load_course_catalog
        from app.tenancy import get_current_tenant

        catalog = load_course_catalog()
        catalog_name = catalog.get((get_current_tenant(), dominant_code))
        if catalog_name:
            return catalog_name

    names: list[str] = []
    for doc in docs:
        for match in _COURSE_NAME_RE.finditer(doc.page_content or ""):
            name = match.group(1).strip()
            if name and name not in names:
                names.append(name)
    for name in names:
        if name and name in query_text:
            return name
    if names:
        return names[0]
    if catalog_names:
        return catalog_names[0]
    joined = "\n".join(doc.page_content for doc in docs)
    if "Python入门" in joined:
        return "Python入门课"
    return default


def _on_llm_failure(query: str, docs: list[Document], exc: BaseException) -> CourseConsultResult:
    """LLM 调用失败（含熔断）时回退规则结构；配额耗尽继续上抛。"""
    from app.llm_gateway import is_llm_degradable
    from app.tenancy import get_current_tenant, get_current_user
    from app.tool_cache import (
        annotate_stale_course,
        get_cached_course,
        normalize_course_cache_key,
    )
    from app.tool_escalation import escalation_for_transient, raise_if_escalated
    from app.tool_cache import TOOL_COURSE

    if not is_llm_degradable(exc):
        raise exc
    tenant = get_current_tenant()
    uid = get_current_user()
    resource_key = normalize_course_cache_key(query)
    cached = get_cached_course(resource_key, tenant_id=tenant, user_id=uid)
    if cached:
        import time

        age = max(0.0, time.time() - cached.cached_at)
        logger.warning("RAG LLM 不可用，回退课程缓存: %s", exc)
        return annotate_stale_course(cached.payload, age_seconds=age)
    raise_if_escalated(escalation_for_transient(TOOL_COURSE, resource_key))
    logger.warning("RAG LLM 不可用，回退规则结构: %s", exc)
    return _fallback_structure(query, docs)


def _fallback_structure(query: str, docs: list[Document]) -> CourseConsultResult:
    """无 LLM 时用正则从检索文本拼出模块与学习路径。"""
    joined = "\n\n".join(doc.page_content for doc in docs)
    title = _course_title_from_docs(query, docs)

    modules: list[CourseModule] = []
    for match in re.finditer(r"模块\s*(\d+)[：:.\s]*([^\n]+)", joined):
        modules.append(CourseModule(name=f"模块{match.group(1)} {match.group(2).strip()}"))

    path = [
        LearningStage(stage="入门", modules=[m.name for m in modules[:2]], goal="建立语法与环境基础"),
        LearningStage(stage="巩固", modules=[m.name for m in modules[2:4]], goal="掌握常用数据结构与函数"),
        LearningStage(stage="实战", modules=[m.name for m in modules[4:]], goal="独立完成小项目"),
    ]
    return CourseConsultResult(
        course_name=title,
        summary=joined[:180].replace("\n", " "),
        modules=modules or [CourseModule(name="课程模块详见知识库")],
        learning_path=[stage for stage in path if stage.modules],
        sources=sorted({str(doc.metadata.get("source")) for doc in docs}),
        evidence=build_evidence_from_docs(docs),
    )


def _retrieval_stub(docs: list[Document]) -> CourseConsultResult:
    """流式路径：只保留出处和课名，不用正则拼完整大纲。"""
    joined = "\n".join(doc.page_content for doc in docs)
    title = ""
    title_match = re.search(r"课程名称[：:]\s*(.+)", joined)
    if title_match:
        title = title_match.group(1).strip()
    return CourseConsultResult(
        course_name=title,
        summary="",
        sources=sorted({str(doc.metadata.get("source")) for doc in docs}),
        evidence=build_evidence_from_docs(docs),
    )


def format_course_answer(result: CourseConsultResult) -> str:
    """把结构化课程结果渲染为 Markdown 回复。"""
    if not result.modules:
        if result.evidence:
            lines = [result.summary, "", "### 参考依据"]
            for item in result.evidence:
                label = item.module_name or item.section_path or item.source
                lines.append(f"- {item.source} · {label}：{item.excerpt}")
            return "\n".join(lines)
        return result.summary
    lines = [f"## {result.course_name}", result.summary, "", "### 课程模块"]
    for index, module in enumerate(result.modules, start=1):
        topics = "、".join(module.topics) if module.topics else "见大纲"
        extra = f"（约 {module.duration_hours:g} 小时）" if module.duration_hours else ""
        lines.append(f"{index}. **{module.name}**{extra}：{topics}")
        if module.outcome:
            lines.append(f"   - 学习目标：{module.outcome}")
    if result.learning_path:
        lines.append("\n### 学习路径")
        for stage in result.learning_path:
            modules = " → ".join(stage.modules) if stage.modules else ""
            lines.append(f"- **{stage.stage}**：{modules}。目标：{stage.goal}")
    if result.prerequisites:
        lines.append("\n### 前置要求")
        lines.extend(f"- {item}" for item in result.prerequisites)
    if result.evidence:
        lines.append("\n### 参考依据")
        for item in result.evidence:
            label = item.module_name or item.section_path or item.source
            lines.append(f"- {item.source} · {label}：{item.excerpt}")
    return "\n".join(lines)


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def _emit_token(content: str) -> None:
    """向 LangGraph custom 流推送 token；图外调用时静默跳过。"""
    if not content:
        return
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    from app.security.filters import mask_pii

    masked, _ = mask_pii(content)
    if masked:
        writer({"event": "token", "content": masked})


async def consult_course_json(query: str, tenant_id: str | None = None) -> str:
    """供 ReACT 工具调用的 JSON 字符串接口（异步）。"""
    result = await consult_course(query, tenant_id=tenant_id)
    return json.dumps(result.model_dump(), ensure_ascii=False)


async def consult_course(
    query: str,
    tenant_id: str | None = None,
    history: list[BaseMessage] | None = None,
    last_course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    course_code: str | None = None,
) -> CourseConsultResult:
    """检索课程知识库，并用 DeepSeek 异步生成结构化咨询结果。

    不再经同步课程线程池（run_course）：检索在事件循环内完成、生成走 await。
    旧调用方（CLI / 工具路径）请用 ``await consult_course(...)`` 或 ``asyncio.run``。
    """
    return await _consult_course(
        query,
        tenant_id,
        history,
        last_course_query,
        last_course_name,
        session_summary,
        course_code,
    )


async def aconsult_course(
    query: str,
    tenant_id: str | None = None,
    history: list[BaseMessage] | None = None,
    last_course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    course_code: str | None = None,
) -> CourseConsultResult:
    """向后兼容别名：等价于 :func:`consult_course`。"""
    return await consult_course(
        query,
        tenant_id=tenant_id,
        history=history,
        last_course_query=last_course_query,
        last_course_name=last_course_name,
        session_summary=session_summary,
        course_code=course_code,
    )


async def _consult_course(
    query: str,
    tenant_id: str | None,
    history: list[BaseMessage] | None,
    last_course_query: str | None,
    last_course_name: str | None,
    session_summary: str | None,
    course_code: str | None = None,
) -> CourseConsultResult:
    tenant = tenant_id or get_current_tenant()
    search_query = _prepare_search_query(
        query,
        history,
        tenant_id=tenant,
        last_course_query=last_course_query,
        last_course_name=last_course_name,
    )
    docs = retrieve_course_docs(search_query, tenant_id=tenant, course_code=course_code)
    if not docs:
        return _not_found_result()

    context = "\n\n---\n\n".join(format_rag_context_chunk(doc) for doc in docs)
    settings = get_settings()
    if not settings.llm_enabled:
        return _with_evidence(_fallback_structure(query, docs), docs)

    from app.llm import get_chat_model

    history_text = format_history_for_prompt(history, session_summary=session_summary)
    user_block = f"用户问题：{query}\n\n检索结果：\n{context}"
    if history_text:
        user_block = f"对话历史：\n{history_text}\n\n{user_block}"

    llm = get_chat_model(temperature=0, usage_tag="rag", cache_enabled=True).with_structured_output(
        CourseConsultResult
    )
    try:
        result = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": get_text_prompt(PROMPT_RAG_STRUCTURED, RAG_STRUCTURED_SYSTEM),
                },
                {"role": "user", "content": user_block},
            ]
        )
    except Exception as exc:
        return _with_evidence(_on_llm_failure(query, docs, exc), docs)
    if not isinstance(result, CourseConsultResult):
        return _with_evidence(_retrieval_stub(docs), docs)
    result.sources = sorted({str(doc.metadata.get("source")) for doc in docs})
    return _with_evidence(result, docs)


async def stream_course_answer(
    query: str,
    tenant_id: str | None = None,
    history: list[BaseMessage] | None = None,
    last_course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    course_code: str | None = None,
) -> tuple[str, CourseConsultResult]:
    """图节点用（异步）：检索后 async for llm.astream 生成 Markdown。

    structured 只带出处，避免正则假大纲；token 经 _emit_token 推入 LangGraph 流。
    """
    return await astream_course_answer(
        query,
        tenant_id=tenant_id,
        history=history,
        last_course_query=last_course_query,
        last_course_name=last_course_name,
        session_summary=session_summary,
        course_code=course_code,
    )


async def astream_course_answer(
    query: str,
    tenant_id: str | None = None,
    history: list[BaseMessage] | None = None,
    last_course_query: str | None = None,
    last_course_name: str | None = None,
    session_summary: str | None = None,
    course_code: str | None = None,
) -> tuple[str, CourseConsultResult]:
    """图节点用（异步）：检索后 async for llm.astream 生成 Markdown。

    structured 只带出处，避免正则假大纲；token 经 _emit_token 推入 LangGraph 流。

    注意：本模块已全异步化，对外标准名是 :func:`stream_course_answer`；
    `astream_course_answer` 作为别名保留以兼容旧导入。
    """
    tenant = tenant_id or get_current_tenant()
    search_query = _prepare_search_query(
        query,
        history,
        tenant_id=tenant,
        last_course_query=last_course_query,
        last_course_name=last_course_name,
    )
    docs = retrieve_course_docs(search_query, tenant_id=tenant, course_code=course_code)
    if not docs:
        result = _not_found_result()
        return result.summary, result

    settings = get_settings()
    if not settings.llm_enabled:
        structured = _with_evidence(_fallback_structure(query, docs), docs)
        return format_course_answer(structured), structured

    from app.llm import get_chat_model

    structured = _with_evidence(_retrieval_stub(docs), docs)
    context = "\n\n---\n\n".join(format_rag_context_chunk(doc) for doc in docs)
    history_text = format_history_for_prompt(history, session_summary=session_summary)
    user_block = f"用户问题：{query}\n\n检索结果：\n{context}"
    if history_text:
        user_block = f"对话历史：\n{history_text}\n\n{user_block}"

    llm = get_chat_model(temperature=0, usage_tag="rag", cache_enabled=False)
    parts: list[str] = []
    try:
        async for chunk in llm.astream(
            [
                {"role": "system", "content": get_text_prompt(PROMPT_RAG_MARKDOWN, RAG_MARKDOWN_SYSTEM)},
                {"role": "user", "content": user_block},
            ]
        ):
            delta = _chunk_text(chunk)
            if not delta:
                continue
            parts.append(delta)
            _emit_token(delta)
    except Exception as exc:
        structured = _with_evidence(_on_llm_failure(query, docs, exc), docs)
        return format_course_answer(structured), structured
    answer = "".join(parts).strip()
    if not answer:
        answer = "已检索到相关资料，但未能生成说明，请稍后重试。"
    return answer, structured
