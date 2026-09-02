"""上下文窗口：token 估算、消息截取、写入 prompt 的历史片段。

职责边界：
- 只决定「最近哪些消息进本轮 prompt」，不负责把溢出压成摘要。
- 摘要行也会占用同一 token 预算：先扣 ``session_summary``，再用剩余额度截消息。
- 短追问扩检索（``expand_query``）也放这里，因为它只拼文本、不改槽位。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage

# 未读配置时的兜底；生产以 HISTORY_TOKEN_BUDGET / HISTORY_MIN_MESSAGES 为准
DEFAULT_HISTORY_TOKEN_BUDGET = 8000
DEFAULT_HISTORY_MIN_MESSAGES = 10
# 每条消息额外计入的角色/分隔开销（tiktoken 不计 chat template 时的近似）
MESSAGE_TOKEN_OVERHEAD = 4
# prompt 里摘要行前缀；计入 token 预算，避免「摘要 + 窗口」一起超模
SUMMARY_PROMPT_PREFIX = "更早对话摘要："
# 短于该字数的问句视为追问，用课名或近期用户句扩展检索
SHORT_QUERY_CHARS = 12
# DeepSeek 等非 OpenAI 模型在 tiktoken 无法识别时的显式回退编码
_FALLBACK_ENCODING_NAME = "cl100k_base"


def message_text(message: BaseMessage) -> str:
    """提取消息文本内容。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _resolve_tokenizer_config() -> tuple[str, str, float]:
    """读取 tokenizer 配置：(显式编码名, 模型名, 校准系数)。"""
    try:
        from app.config import get_settings

        settings = get_settings()
        return (
            (settings.history_tokenizer_name or "").strip(),
            (settings.deepseek_model or "").strip(),
            float(settings.history_token_estimate_ratio or 1.0),
        )
    except Exception:
        return "", "", 1.0


@lru_cache(maxsize=8)
def _load_encoding(tokenizer_name: str, model_name: str) -> tuple[object | None, str]:
    """按显式编码名 → 模型名 → cl100k 回退加载 tiktoken；返回 (encoding, strategy)。"""
    try:
        import tiktoken
    except Exception:
        return None, "byte_approx"

    if tokenizer_name:
        try:
            return tiktoken.get_encoding(tokenizer_name), f"named:{tokenizer_name}"
        except Exception:
            pass

    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name), f"model:{model_name}"
        except Exception:
            # DeepSeek 等非 OpenAI 模型：显式回退，而不是静默写死在调用点
            try:
                return (
                    tiktoken.get_encoding(_FALLBACK_ENCODING_NAME),
                    f"cl100k_fallback:{model_name}",
                )
            except Exception:
                return None, "byte_approx"

    try:
        return tiktoken.get_encoding(_FALLBACK_ENCODING_NAME), "cl100k_fallback"
    except Exception:
        return None, "byte_approx"


def token_estimate_strategy() -> str:
    """当前 token 估算策略标签（供测试与诊断）。"""
    tokenizer_name, model_name, _ = _resolve_tokenizer_config()
    _, strategy = _load_encoding(tokenizer_name, model_name)
    return strategy


def count_tokens(text: str) -> int:
    """估算文本 token 数。

    优先：``HISTORY_TOKENIZER_NAME`` → ``tiktoken.encoding_for_model(deepseek_model)``
    → 显式 ``cl100k_base`` 回退；再乘 ``HISTORY_TOKEN_ESTIMATE_RATIO``。
    tiktoken 不可用时按 UTF-8 字节近似。
    """
    if not text:
        return 0
    tokenizer_name, model_name, ratio = _resolve_tokenizer_config()
    encoding, _ = _load_encoding(tokenizer_name, model_name)
    if encoding is not None:
        raw = len(encoding.encode(text))  # type: ignore[attr-defined]
    else:
        # ~4 bytes / token，至少计 1
        raw = max(1, (len(text.encode("utf-8")) + 3) // 4)
    scaled = int(round(raw * max(0.01, float(ratio))))
    return max(1, scaled)


def message_token_count(message: BaseMessage) -> int:
    """单条消息的 token 估算（含角色开销）。"""
    return count_tokens(message_text(message)) + MESSAGE_TOKEN_OVERHEAD


def prompt_summary_line(summary: str | None) -> str:
    """写入 prompt 的摘要行；空摘要返回空串。"""
    text = (summary or "").strip()
    if not text:
        return ""
    return f"{SUMMARY_PROMPT_PREFIX}{text}"


def prompt_summary_tokens(summary: str | None) -> int:
    """摘要行占用的 token；空摘要为 0。"""
    line = prompt_summary_line(summary)
    return count_tokens(line) if line else 0


def history_limits(
    token_budget: int | None = None,
    min_keep: int | None = None,
) -> tuple[int, int]:
    """读取历史裁剪预算；未传则用配置默认值。"""
    if token_budget is not None and min_keep is not None:
        return max(1, token_budget), max(1, min_keep)
    try:
        from app.config import get_settings

        settings = get_settings()
        budget = token_budget if token_budget is not None else settings.history_token_budget
        keep = min_keep if min_keep is not None else settings.history_min_messages
    except Exception:
        budget = token_budget if token_budget is not None else DEFAULT_HISTORY_TOKEN_BUDGET
        keep = min_keep if min_keep is not None else DEFAULT_HISTORY_MIN_MESSAGES
    return max(1, int(budget)), max(1, int(keep))


def select_messages_by_tokens(
    messages: list[BaseMessage] | None,
    *,
    token_budget: int | None = None,
    min_keep: int | None = None,
) -> list[BaseMessage]:
    """从最新消息往前累加，直到达到 token 预算；至少保留 min_keep 条。"""
    if not messages:
        return []
    budget, keep = history_limits(token_budget, min_keep)
    selected: list[BaseMessage] = []
    total = 0
    for message in reversed(messages):
        cost = message_token_count(message)
        if selected and total + cost > budget and len(selected) >= keep:
            break
        selected.append(message)
        total += cost
    selected.reverse()
    return selected


def recent_messages(
    messages: list[BaseMessage] | None,
    limit: int | None = None,
    *,
    token_budget: int | None = None,
    min_keep: int | None = None,
) -> list[BaseMessage]:
    """按 token 预算截取最近消息；limit 仅作条数上限（兼容旧调用）。"""
    selected = select_messages_by_tokens(
        messages, token_budget=token_budget, min_keep=min_keep
    )
    if limit is not None and limit >= 0:
        return selected[-limit:]
    return selected


def recent_human_texts(messages: list[BaseMessage] | None, *, exclude_last: bool = False) -> list[str]:
    """按时间倒序返回用户消息文本（可选排除最后一条，通常是本轮 query）。"""
    items = [message_text(m) for m in (messages or []) if isinstance(m, HumanMessage)]
    if exclude_last and items:
        items = items[:-1]
    return list(reversed(items))


def order_no_from_history(messages: list[BaseMessage] | None) -> str | None:
    """从最近用户消息中提取订单号。"""
    from app.services.order import extract_order_no

    for text in recent_human_texts(messages, exclude_last=False):
        order_no = extract_order_no(text)
        if order_no:
            return order_no
    return None


def overflow_messages(
    messages: list[BaseMessage] | None,
    *,
    token_budget: int | None = None,
    min_keep: int | None = None,
) -> list[BaseMessage]:
    """窗口外将被裁掉的旧消息（时间更早的前缀）。"""
    if not messages:
        return []
    selected = select_messages_by_tokens(
        messages, token_budget=token_budget, min_keep=min_keep
    )
    if len(selected) >= len(messages):
        return []
    return list(messages[: len(messages) - len(selected)])


def format_history_for_prompt(
    messages: list[BaseMessage] | None,
    limit: int | None = None,
    *,
    token_budget: int | None = None,
    session_summary: str | None = None,
) -> str:
    """把最近对话格式化为 prompt 片段（默认按 token 预算截取）。

    ``session_summary`` 占用同一预算：先扣摘要 token，再用剩余额度截取消息。
    """
    lines: list[str] = []
    summary_line = prompt_summary_line(session_summary)
    if summary_line:
        lines.append(summary_line)
        budget, _ = history_limits(token_budget, None)
        token_budget = max(1, budget - prompt_summary_tokens(session_summary))
    for message in recent_messages(messages, limit=limit, token_budget=token_budget):
        if isinstance(message, HumanMessage):
            role = "用户"
        elif isinstance(message, AIMessage):
            role = "助手"
        else:
            continue
        text = message_text(message).strip()
        if text:
            lines.append(f"{role}：{text}")
    return "\n".join(lines)


def expand_query(
    query: str,
    messages: list[BaseMessage] | None,
    *,
    last_course_query: str | None = None,
    last_course_name: str | None = None,
) -> str:
    """短追问时优先用会话槽位中的课程，再用近期用户句扩展检索 query。"""
    text = (query or "").strip()
    if len(text) >= SHORT_QUERY_CHARS:
        return text
    course_hint = (last_course_name or last_course_query or "").strip()
    if course_hint:
        return f"{course_hint} {text}".strip() if text else course_hint
    priors = recent_human_texts(messages, exclude_last=True)
    if not priors:
        return text
    context = " ".join(reversed(priors[:2]))
    return f"{context} {text}".strip() if text else context


def trim_message_updates(
    messages: list[BaseMessage] | None,
    keep: int | None = None,
    *,
    token_budget: int | None = None,
    min_keep: int | None = None,
) -> list[BaseMessage]:
    """生成 RemoveMessage 列表，按 token 预算裁掉窗口外旧消息。

    ``keep`` 若传入，表示最少保留条数（覆盖 min_keep），不再表示「固定保留 N 条」。
    """
    if not messages:
        return []
    selected = select_messages_by_tokens(
        messages,
        token_budget=token_budget,
        min_keep=keep if keep is not None else min_keep,
    )
    if len(selected) >= len(messages):
        return []
    overflow = messages[: len(messages) - len(selected)]
    return [RemoveMessage(id=m.id) for m in overflow if getattr(m, "id", None)]
