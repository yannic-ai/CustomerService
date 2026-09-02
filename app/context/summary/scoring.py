"""消息重要性评分与轮次分组。"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.context.slots import SessionSlots
from app.context.window import message_text

SCORE_WEIGHT_RELEVANCE = 0.4
SCORE_WEIGHT_RECENCY = 0.3
SCORE_WEIGHT_FREQUENCY = 0.2
SCORE_WEIGHT_MARKED = 0.1

_CJK_PATTERN = re.compile(r"[一-鿿]+")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]{2,}")


def _text_tokens(text: str) -> set[str]:
    if not text:
        return set()
    tokens: set[str] = set()
    for run in _CJK_PATTERN.findall(text):
        if len(run) == 1:
            tokens.add(run)
        else:
            for i in range(len(run) - 1):
                tokens.add(run[i : i + 2])
    tokens.update(w.lower() for w in _WORD_PATTERN.findall(text))
    return tokens


def _relevance_score(message_text_value: str, current_query: str) -> float:
    if not current_query or not message_text_value:
        return 0.0
    msg_tokens = _text_tokens(message_text_value)
    query_tokens = _text_tokens(current_query)
    union = msg_tokens | query_tokens
    if not union:
        return 0.0
    return len(msg_tokens & query_tokens) / len(union)


def _recency_score(msg_index: int, total: int) -> float:
    if total <= 0:
        return 0.0
    distance = max(0, total - 1 - msg_index)
    return max(0.0, 1.0 - (distance / 5) * 0.1)


def _frequency_score(
    msg_text: str,
    later_messages: list[BaseMessage],
    slots: SessionSlots | None,
) -> float:
    if not msg_text or not later_messages:
        return 0.0
    keywords: list[str] = []
    if slots:
        for val in (
            slots.last_order_no,
            slots.last_course_name,
            slots.last_course_query,
        ):
            val = (val or "").strip()
            if val and len(val) >= 2 and val in msg_text:
                keywords.append(val)
    if not keywords:
        return 0.0
    refs = 0
    for later in later_messages:
        later_text = message_text(later)
        if any(kw in later_text for kw in keywords):
            refs += 1
    return min(1.0, refs * 0.2)


def score_message(
    message: BaseMessage,
    messages: list[BaseMessage],
    *,
    current_query: str = "",
    slots: SessionSlots | None = None,
    marked_important: bool = False,
) -> float:
    text = message_text(message)
    if not text.strip():
        return 0.0
    try:
        msg_index = messages.index(message)
    except ValueError:
        msg_index = len(messages) - 1

    relevance = _relevance_score(text, current_query)
    recency = _recency_score(msg_index, len(messages))
    later = messages[msg_index + 1 :] if msg_index + 1 < len(messages) else []
    frequency = _frequency_score(msg_text=text, later_messages=later, slots=slots)
    marked = 1.0 if marked_important else 0.0

    return (
        relevance * SCORE_WEIGHT_RELEVANCE
        + recency * SCORE_WEIGHT_RECENCY
        + frequency * SCORE_WEIGHT_FREQUENCY
        + marked * SCORE_WEIGHT_MARKED
    )


def partition_turns(
    messages: list[BaseMessage],
) -> tuple[list[tuple[int, BaseMessage]], list[list[tuple[int, BaseMessage]]]]:
    systems: list[tuple[int, BaseMessage]] = []
    turns: list[list[tuple[int, BaseMessage]]] = []
    current: list[tuple[int, BaseMessage]] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            if current:
                turns.append(current)
                current = []
            systems.append((i, msg))
            continue
        if (
            isinstance(msg, HumanMessage)
            and current
            and any(isinstance(item, AIMessage) for _, item in current)
        ):
            turns.append(current)
            current = [(i, msg)]
        elif (
            isinstance(msg, AIMessage)
            and current
            and all(not isinstance(item, HumanMessage) for _, item in current)
        ):
            turns.append(current)
            current = [(i, msg)]
        else:
            current.append((i, msg))
    if current:
        turns.append(current)
    return systems, turns
