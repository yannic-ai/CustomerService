"""分级压缩：评分 → 按轮分级 → 保底再裁。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol

from langchain_core.messages import BaseMessage

from app.context.slots import SessionSlots
from app.context.summary.facts import finalize_summary, fold_session_summary
from app.context.summary.llm import _llm_batch_summarize
from app.context.summary.rag_compact import apply_rag_compact, compact_rag_message
from app.context.summary.scoring import partition_turns, score_message
from app.context.window import message_token_count, prompt_summary_tokens


@dataclass
class CompressResult:
    kept: list[BaseMessage] = field(default_factory=list)
    summarized: list[BaseMessage] = field(default_factory=list)
    discarded: list[BaseMessage] = field(default_factory=list)
    summary: str = ""
    updated: list[BaseMessage] = field(default_factory=list)


@dataclass
class _CompressionPlan:
    kept: list[BaseMessage]
    summarized: list[BaseMessage]
    discarded: list[BaseMessage]
    fact_overflow: list[BaseMessage]
    updated: list[BaseMessage]


class _SummaryFn(Protocol):
    def __call__(
        self,
        messages: list[BaseMessage],
        existing_summary: str,
        *,
        max_chars: int | None = None,
        slots: SessionSlots | None = None,
    ) -> Awaitable[str]: ...


def _split_kept_by_budget(
    kept: list[BaseMessage],
    *,
    token_budget: int,
    summary: str,
    always_keep_recent: int,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    if not kept:
        return [], []
    budget = max(1, token_budget - prompt_summary_tokens(summary))
    systems, turns = partition_turns(kept)
    system_msgs = [m for _, m in systems]
    pinned_cost = sum(message_token_count(m) for m in system_msgs)
    rest_budget = max(1, budget - pinned_cost)
    body_count = max(1, len(kept) - len(system_msgs))
    min_keep = min(max(1, always_keep_recent), body_count)

    selected_rev: list[list[tuple[int, BaseMessage]]] = []
    total = 0
    overflow_head: list[list[tuple[int, BaseMessage]]] = []
    for offset, turn in enumerate(reversed(turns)):
        msgs = [m for _, m in turn]
        cost = sum(message_token_count(m) for m in msgs)
        n_selected = sum(len(item) for item in selected_rev)
        if selected_rev and total + cost > rest_budget and n_selected >= min_keep:
            overflow_head = turns[: len(turns) - offset]
            break
        selected_rev.append(turn)
        total += cost
    else:
        overflow_head = []

    keep_ids = {id(m) for t in reversed(selected_rev) for _, m in t}
    keep_ids.update(id(m) for m in system_msgs)
    kept_out = [m for m in kept if id(m) in keep_ids]
    overflow = [m for t in overflow_head for _, m in t]
    return kept_out, overflow


def _plan_compression(
    messages: list[BaseMessage],
    *,
    current_query: str,
    token_budget: int,
    slots: SessionSlots | None,
    always_keep_recent: int,
    discard_score: float,
    summarize_score: float,
    max_messages: int,
    existing_summary: str,
    max_summary_chars: int | None,
) -> CompressResult | _CompressionPlan:
    if not messages:
        return CompressResult(
            summary=finalize_summary(
                existing_summary, slots=slots, max_chars=max_summary_chars
            )
        )

    total_tokens = sum(message_token_count(m) for m in messages) + prompt_summary_tokens(
        existing_summary
    )
    if total_tokens <= token_budget and len(messages) <= max_messages:
        kept, updated = apply_rag_compact(messages)
        return CompressResult(
            kept=kept,
            summary=finalize_summary(
                existing_summary, slots=slots, max_chars=max_summary_chars
            ),
            updated=updated,
        )

    scores = [
        score_message(m, messages, current_query=current_query, slots=slots)
        for m in messages
    ]

    n = len(messages)
    recent_start = max(0, n - always_keep_recent)
    systems, turns = partition_turns(messages)

    keep_orig: set[int] = {id(m) for _, m in systems}
    compact_by_orig: dict[int, BaseMessage] = {}
    summarized: list[BaseMessage] = []
    discarded: list[BaseMessage] = []
    fact_overflow: list[BaseMessage] = []
    updated: list[BaseMessage] = []

    def _compact_turn(msgs: list[BaseMessage]) -> list[BaseMessage]:
        compacted: list[BaseMessage] = []
        for orig in msgs:
            new = compact_rag_message(orig)
            compacted.append(new)
            if new is not orig:
                compact_by_orig[id(orig)] = new
        return compacted

    for turn in turns:
        msgs = [m for _, m in turn]
        last_idx = turn[-1][0]
        is_recent = last_idx >= recent_start
        if is_recent:
            _compact_turn(msgs)
            keep_orig.update(id(m) for m in msgs)
            continue
        score = max((scores[i] for i, _ in turn), default=0.0)
        if score < discard_score:
            discarded.extend(msgs)
            fact_overflow.extend(msgs)
        elif score < summarize_score:
            compacted = _compact_turn(msgs)
            summarized.extend(compacted)
            fact_overflow.extend(compacted)
        else:
            _compact_turn(msgs)
            keep_orig.update(id(m) for m in msgs)

    kept = [compact_by_orig.get(id(m), m) for m in messages if id(m) in keep_orig]
    updated = [
        compact_by_orig[id(m)]
        for m in messages
        if id(m) in keep_orig and id(m) in compact_by_orig
    ]

    def _drop_updated(overflow: list[BaseMessage]) -> None:
        overflow_ids = {getattr(m, "id", None) for m in overflow}
        overflow_ids.discard(None)
        if not overflow_ids:
            return
        updated[:] = [m for m in updated if getattr(m, "id", None) not in overflow_ids]

    kept, fallback_overflow = _split_kept_by_budget(
        kept,
        token_budget=token_budget,
        summary=existing_summary,
        always_keep_recent=always_keep_recent,
    )
    summarized.extend(fallback_overflow)
    fact_overflow.extend(fallback_overflow)
    _drop_updated(fallback_overflow)

    return _CompressionPlan(
        kept=kept,
        summarized=summarized,
        discarded=discarded,
        fact_overflow=fact_overflow,
        updated=updated,
    )


def _finalize_compression(
    plan: _CompressionPlan,
    *,
    existing_summary: str,
    summary: str,
    token_budget: int,
    always_keep_recent: int,
    max_summary_chars: int | None,
    slots: SessionSlots | None,
) -> CompressResult:
    kept = plan.kept
    summarized = list(plan.summarized)
    updated = list(plan.updated)

    def _drop_updated(overflow: list[BaseMessage]) -> None:
        overflow_ids = {getattr(m, "id", None) for m in overflow}
        overflow_ids.discard(None)
        if not overflow_ids:
            return
        updated[:] = [m for m in updated if getattr(m, "id", None) not in overflow_ids]

    kept, extra = _split_kept_by_budget(
        kept,
        token_budget=token_budget,
        summary=summary,
        always_keep_recent=always_keep_recent,
    )
    if extra:
        summarized.extend(extra)
        _drop_updated(extra)
        summary = fold_session_summary(
            summary, extra, max_chars=max_summary_chars, slots=slots
        )

    summary = finalize_summary(summary, slots=slots, max_chars=max_summary_chars)
    return CompressResult(
        kept=kept,
        summarized=summarized,
        discarded=plan.discarded,
        summary=summary,
        updated=updated,
    )


async def _run_compression(
    messages: list[BaseMessage],
    *,
    current_query: str = "",
    token_budget: int,
    slots: SessionSlots | None = None,
    always_keep_recent: int = 10,
    discard_score: float = 0.3,
    summarize_score: float = 0.6,
    max_messages: int = 50,
    existing_summary: str = "",
    use_llm_summary: bool = True,
    max_summary_chars: int | None = None,
    llm_summarize: _SummaryFn,
) -> CompressResult:
    planned = _plan_compression(
        messages,
        current_query=current_query,
        token_budget=token_budget,
        slots=slots,
        always_keep_recent=always_keep_recent,
        discard_score=discard_score,
        summarize_score=summarize_score,
        max_messages=max_messages,
        existing_summary=existing_summary,
        max_summary_chars=max_summary_chars,
    )
    if isinstance(planned, CompressResult):
        return planned

    summary = existing_summary
    if planned.fact_overflow:
        if use_llm_summary:
            summary = await llm_summarize(
                planned.fact_overflow,
                existing_summary,
                max_chars=max_summary_chars,
                slots=slots,
            )
        else:
            summary = fold_session_summary(
                existing_summary,
                planned.fact_overflow,
                max_chars=max_summary_chars,
                slots=slots,
            )

    return _finalize_compression(
        planned,
        existing_summary=existing_summary,
        summary=summary,
        token_budget=token_budget,
        always_keep_recent=always_keep_recent,
        max_summary_chars=max_summary_chars,
        slots=slots,
    )


async def compress_by_importance(
    messages: list[BaseMessage],
    *,
    current_query: str = "",
    token_budget: int,
    slots: SessionSlots | None = None,
    always_keep_recent: int = 10,
    discard_score: float = 0.3,
    summarize_score: float = 0.6,
    max_messages: int = 50,
    existing_summary: str = "",
    use_llm_summary: bool = True,
    max_summary_chars: int | None = None,
) -> CompressResult:
    return await _run_compression(
        messages,
        current_query=current_query,
        token_budget=token_budget,
        slots=slots,
        always_keep_recent=always_keep_recent,
        discard_score=discard_score,
        summarize_score=summarize_score,
        max_messages=max_messages,
        existing_summary=existing_summary,
        use_llm_summary=use_llm_summary,
        max_summary_chars=max_summary_chars,
        llm_summarize=_llm_batch_summarize,
    )
