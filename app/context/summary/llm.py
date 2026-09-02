"""LLM 结构化摘要：触发判断与异步调用。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage

from app.context.slots import SessionSlots
from app.context.summary.facts import (
    ORDER_NO_RE,
    StructuredSummary,
    extract_course_names,
    extract_summary_facts,
    finalize_structured,
    finalize_summary,
    fold_session_summary,
)
from app.context.window import message_text
from app.observability.langfuse import PROMPT_SUMMARY, get_text_prompt

SUMMARY_SYSTEM_PROMPT = (
    "你是客服会话摘要助手。根据已有结构化摘要和本批溢出消息，输出更新后的摘要。"
    "只记录约束和结论（例如「用户要退款」「已确认不适合零基础」），"
    "不要重复槽位里已有的订单号、课程名，不要抄对话原文。"
    "互斥结论只留后写的（适合零基础 / 已确认不适合零基础；退款审核中 / 退款已完成）。"
    "槽位或摘要已有订单号则不要保留「待提供订单号」，已有课名则不要保留「待确认课程」。"
    "格式严格如下，没有的栏目整行省略：\n"
    "订单：\n课程：\n意图：\n已确认：\n未决：\n"
    "不要编号，不要前后缀。"
)

_BUSINESS_SIGNAL_RE = re.compile(
    r"订单|退款|退钱|物流|快递|发货|取消|进度|"
    r"课程|大纲|模块|学费|价格|多少钱|零基础|适合|付费|推荐|"
    r"太难|很难|偏难|转人工|转接|开课|学什么|讲什么|"
    r"不想学|改主意|别推|不要推|别推荐|不要推荐"
)


def _overflow_has_business_signal(messages: list[BaseMessage]) -> bool:
    for message in messages:
        text = message_text(message).strip()
        if not text:
            continue
        if ORDER_NO_RE.search(text):
            return True
        if _BUSINESS_SIGNAL_RE.search(text):
            return True
        if extract_course_names(text):
            return True
    return False


def should_invoke_llm_summary(
    messages: list[BaseMessage],
    *,
    existing: StructuredSummary,
    delta: StructuredSummary,
) -> bool:
    if not messages:
        return False
    if delta.novel_against(existing):
        return True
    return _overflow_has_business_signal(messages)


@dataclass(frozen=True)
class _LlmSummaryPrepared:
    kind: Literal["done", "invoke"]
    result: str = ""
    existing: StructuredSummary | None = None
    delta: StructuredSummary | None = None
    user_block: str = ""


def _prepare_llm_summary(
    messages: list[BaseMessage],
    existing_summary: str,
    *,
    max_chars: int | None = None,
    slots: SessionSlots | None = None,
) -> _LlmSummaryPrepared:
    existing = StructuredSummary.parse(existing_summary)
    delta = extract_summary_facts(messages, slots=slots)
    if not messages:
        return _LlmSummaryPrepared(
            kind="done",
            result=finalize_summary(existing_summary, slots=slots, max_chars=max_chars),
        )
    if not should_invoke_llm_summary(messages, existing=existing, delta=delta):
        return _LlmSummaryPrepared(
            kind="done",
            result=fold_session_summary(
                existing_summary, messages, max_chars=max_chars, slots=slots
            ),
        )

    lines: list[str] = []
    for m in messages:
        role = "用户" if isinstance(m, HumanMessage) else "助手"
        text = message_text(m).strip().replace("\n", " ")
        if text:
            lines.append(f"{role}：{text[:200]}")
    if not lines:
        return _LlmSummaryPrepared(
            kind="done",
            result=finalize_structured(existing, slots=slots, max_chars=max_chars),
        )

    slot_hint = ""
    if slots:
        slot_hint = (
            f"槽位已有：订单={slots.last_order_no or '无'}；"
            f"课程={slots.last_course_name or '无'}；"
            f"意图={slots.last_intent or '无'}。\n"
        )
    user_block = (
        f"{slot_hint}已有摘要：\n{existing.render() or '（空）'}\n\n"
        f"溢出消息：\n" + "\n".join(lines)
    )
    return _LlmSummaryPrepared(
        kind="invoke",
        existing=existing,
        delta=delta,
        user_block=user_block,
    )


async def _invoke_llm_summary(prepared: _LlmSummaryPrepared) -> str:
    from app.llm import get_chat_model

    llm = get_chat_model(temperature=0, usage_tag="summary", cache_enabled=False)
    response = await llm.ainvoke(
        [
            {"role": "system", "content": get_text_prompt(PROMPT_SUMMARY, SUMMARY_SYSTEM_PROMPT)},
            {"role": "user", "content": prepared.user_block},
        ]
    )
    return message_text(response).strip()


async def _llm_batch_summarize(
    messages: list[BaseMessage],
    existing_summary: str,
    *,
    max_chars: int | None = None,
    slots: SessionSlots | None = None,
) -> str:
    prepared = _prepare_llm_summary(
        messages, existing_summary, max_chars=max_chars, slots=slots
    )
    if prepared.kind == "done":
        return prepared.result

    assert prepared.existing is not None and prepared.delta is not None
    try:
        summary_text = await _invoke_llm_summary(prepared)
    except Exception:
        return fold_session_summary(
            existing_summary, messages, max_chars=max_chars, slots=slots
        )
    return _merge_llm_summary_text(
        summary_text,
        existing=prepared.existing,
        delta=prepared.delta,
        existing_summary=existing_summary,
        messages=messages,
        max_chars=max_chars,
        slots=slots,
    )


def _merge_llm_summary_text(
    summary_text: str,
    *,
    existing: StructuredSummary,
    delta: StructuredSummary,
    existing_summary: str,
    messages: list[BaseMessage],
    max_chars: int | None,
    slots: SessionSlots | None,
) -> str:
    if not summary_text:
        return fold_session_summary(
            existing_summary, messages, max_chars=max_chars, slots=slots
        )
    llm_delta = StructuredSummary.parse(summary_text)
    if not llm_delta.has_content():
        llm_delta = StructuredSummary(confirmed=[summary_text.replace("\n", " ").strip()[:120]])
    merged = existing.merge(llm_delta).merge(delta)
    return finalize_structured(merged, slots=slots, max_chars=max_chars)

