import asyncio

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from app.context import (
    CompressResult,
    DEFAULT_HISTORY_MIN_MESSAGES,
    DEFAULT_HISTORY_TOKEN_BUDGET,
    SessionSlots,
    StructuredSummary,
    _clip_summary,
    _llm_batch_summarize,
    compact_rag_history,
    compress_by_importance,
    count_tokens,
    finalize_respond_turn,
    fold_session_summary,
    format_history_for_prompt,
    overflow_messages,
    recent_messages,
    score_message,
    select_messages_by_tokens,
    token_estimate_strategy,
    trim_message_updates,
)
from app.context.turn import build_message_updates
from app.context.summary import _should_invoke_llm_summary
from app.context.window import _load_encoding
from app.config import get_settings


def _compress(messages, **kwargs):
    return asyncio.run(compress_by_importance(messages, **kwargs))


def _llm_summarize(messages, existing_summary, **kwargs):
    return asyncio.run(_llm_batch_summarize(messages, existing_summary, **kwargs))


def test_count_tokens_positive() -> None:
    assert count_tokens("hello") > 0
    assert count_tokens("查询订单退款进度") > count_tokens("订单")


def test_select_keeps_suffix_within_budget() -> None:
    blob = "课程模块说明" * 80
    messages = []
    for index in range(6):
        messages.append(HumanMessage(content=f"问{index} {blob}", id=f"u{index}"))
        messages.append(AIMessage(content=f"答{index} {blob}", id=f"a{index}"))
    kept = select_messages_by_tokens(messages, token_budget=200, min_keep=2)
    assert kept
    assert kept[-1].id == "a5"
    assert len(kept) < len(messages)
    assert [m.id for m in kept] == [m.id for m in messages[-len(kept) :]]


def test_select_always_keeps_minimum_even_if_over_budget() -> None:
    huge = "很长的订单进度报告" * 200
    messages = [
        HumanMessage(content=huge, id="u0"),
        AIMessage(content=huge, id="a0"),
        HumanMessage(content=huge, id="u1"),
        AIMessage(content=huge, id="a1"),
    ]
    kept = select_messages_by_tokens(messages, token_budget=10, min_keep=2)
    assert [m.id for m in kept] == ["u1", "a1"]


def test_trim_emits_remove_for_overflow() -> None:
    blob = "大纲" * 100
    messages = [HumanMessage(content=f"{index}-{blob}", id=str(index)) for index in range(8)]
    removals = trim_message_updates(messages, token_budget=80, min_keep=2)
    assert removals
    assert all(isinstance(item, RemoveMessage) for item in removals)
    removed_ids = {item.id for item in removals}
    kept = select_messages_by_tokens(messages, token_budget=80, min_keep=2)
    assert removed_ids.isdisjoint({m.id for m in kept})
    assert len(removed_ids) + len(kept) == len(messages)


def test_recent_messages_token_budget_used_in_prompt() -> None:
    blob = "学习路径" * 60
    messages = [
        HumanMessage(content=f"早期问题 {blob}", id="old"),
        AIMessage(content=f"早期回答 {blob}", id="old-a"),
        HumanMessage(content="现在问退款", id="new"),
        AIMessage(content="请提供订单号", id="new-a"),
    ]
    recent = recent_messages(messages, token_budget=40, min_keep=2)
    assert recent[-1].id == "new-a"
    text = format_history_for_prompt(messages, token_budget=40)
    assert "现在问退款" in text


def test_fold_summary_and_prompt() -> None:
    overflow = [
        HumanMessage(content="很早以前问过订单#20251114001", id="old"),
        AIMessage(content="已查询退款", id="old-a"),
    ]
    summary = fold_session_summary("", overflow, max_chars=200)
    assert "20251114001" in summary
    assert "用户要退款" in summary
    assert "用户：" not in summary
    text = format_history_for_prompt(
        [HumanMessage(content="那退款呢")],
        session_summary=summary,
    )
    assert "更早对话摘要" in text
    assert overflow_messages(
        overflow + [HumanMessage(content="x", id="n")],
        token_budget=5,
        min_keep=1,
    )


def test_fold_structured_records_constraints() -> None:
    """结构化摘要记录约束/结论，而不是对话原文拼接。"""
    overflow = [
        HumanMessage(content="Python入门课适合零基础吗？太难的话就算了", id="u0"),
        AIMessage(content="## Python入门课\n不适合零基础，建议先补语法。", id="a0"),
    ]
    summary = fold_session_summary("", overflow, max_chars=400)
    parsed = StructuredSummary.parse(summary)
    assert "已确认不适合零基础" in parsed.confirmed
    assert "适合零基础" not in parsed.confirmed
    assert "用户：" not in summary
    assert "助手：" not in summary


def test_fold_skips_identifiers_already_in_slots() -> None:
    """槽位已有订单号/课名时，摘要只补约束。"""
    slots = SessionSlots(last_order_no="20251114001", last_course_name="Python入门课")
    overflow = [
        HumanMessage(content="订单#20251114001 我要退款，Python入门课不适合零基础", id="u0"),
        AIMessage(content="好的，已记下", id="a0"),
    ]
    summary = fold_session_summary("", overflow, slots=slots, max_chars=400)
    assert "用户要退款" in summary
    assert "已确认不适合零基础" in summary
    assert "20251114001" not in summary
    assert "Python入门课" not in summary


def test_structured_summary_merges_and_drops_oldest() -> None:
    current = StructuredSummary(confirmed=["旧结论A", "旧结论B"])
    delta = StructuredSummary(confirmed=["新结论C"])
    merged = current.merge(delta)
    assert merged.confirmed[-1] == "新结论C"
    text = merged.render(max_chars=80)
    assert "新结论C" in text


def test_exclusive_confirmed_later_write_wins() -> None:
    """互斥已确认以后写为准，其它标签保留。"""
    current = StructuredSummary(confirmed=["适合零基础", "用户觉得难"])
    delta = StructuredSummary(confirmed=["已确认不适合零基础"])
    merged = current.merge(delta)
    assert "已确认不适合零基础" in merged.confirmed
    assert "适合零基础" not in merged.confirmed
    assert "用户觉得难" in merged.confirmed


def test_exclusive_refund_status_later_write_wins() -> None:
    current = StructuredSummary(confirmed=["退款审核中"])
    delta = StructuredSummary(confirmed=["退款已完成"])
    merged = current.merge(delta)
    assert merged.confirmed == ["退款已完成"]


def test_pending_order_cleared_when_slot_has_order_no() -> None:
    """槽位已有订单号时关闭「待提供订单号」。"""
    summary = fold_session_summary(
        "已确认：用户要退款\n未决：待提供订单号",
        [HumanMessage(content="帮我查一下进度", id="u0")],
        slots=SessionSlots(last_order_no="20251114001"),
        max_chars=400,
    )
    parsed = StructuredSummary.parse(summary)
    assert "待提供订单号" not in parsed.pending
    assert "用户要退款" in parsed.confirmed


def test_pending_order_cleared_when_overflow_has_order_no() -> None:
    """溢出消息里出现订单号时同样关闭未决。"""
    summary = fold_session_summary(
        "未决：待提供订单号",
        [HumanMessage(content="订单号是 20251114001", id="u0")],
        max_chars=400,
    )
    parsed = StructuredSummary.parse(summary)
    assert "待提供订单号" not in parsed.pending
    assert any("20251114001" in item for item in parsed.orders)


def test_pending_course_cleared_when_slot_has_course_name() -> None:
    summary = fold_session_summary(
        "未决：待确认课程",
        [HumanMessage(content="那就这门吧", id="u0")],
        slots=SessionSlots(last_course_name="Python入门课"),
        max_chars=400,
    )
    parsed = StructuredSummary.parse(summary)
    assert "待确认课程" not in parsed.pending


def test_compress_reconciles_pending_under_threshold() -> None:
    """未触发分级压缩时，槽位补单号仍关闭未决。"""
    messages = [
        HumanMessage(content="查一下进度", id="u0"),
        AIMessage(content="好的", id="a0"),
    ]
    result = _compress(
        messages,
        current_query="查一下进度",
        token_budget=10000,
        max_messages=50,
        existing_summary="已确认：用户要退款\n未决：待提供订单号",
        slots=SessionSlots(last_order_no="20251114001"),
    )
    parsed = StructuredSummary.parse(result.summary)
    assert "待提供订单号" not in parsed.pending
    assert "用户要退款" in parsed.confirmed


def test_format_history_counts_summary_against_budget() -> None:
    """session_summary 占用同一 token 预算，超出 min_keep 的旧消息应被挤出 prompt。"""
    summary = "更早确认过订单 SO20240001 的退款进度。" * 8
    blob = "课程大纲说明" * 40
    messages = []
    for i in range(8):
        messages.append(HumanMessage(content=f"早期问题{i} {blob}", id=f"old-{i}"))
        messages.append(AIMessage(content=f"早期回答{i} {blob}", id=f"old-a-{i}"))
    messages.extend(
        [
            HumanMessage(content="现在问退款", id="new"),
            AIMessage(content="请提供订单号", id="new-a"),
        ]
    )
    text = format_history_for_prompt(
        messages,
        token_budget=80,
        session_summary=summary,
    )
    assert "更早对话摘要" in text
    assert "现在问退款" in text
    assert "早期问题0" not in text


def test_clip_summary_keeps_tail() -> None:
    """规则与 LLM 共用留尾截断。"""
    text = "旧事实AAAA" * 20 + "新事实订单SO20249999已退款"
    clipped = _clip_summary(text, max_chars=80)
    assert len(clipped) <= 80
    assert "SO20249999" in clipped
    assert not clipped.startswith("旧事实")


# ---------------------------- score_message 测试 ---------------------------- #


def test_score_returns_zero_for_empty_message() -> None:
    msg = AIMessage(content="", id="empty")
    assert score_message(msg, [msg]) == 0.0


def test_score_in_range_zero_to_one() -> None:
    messages = [
        HumanMessage(content="我想查订单 SO20240001 的物流", id="u0"),
        AIMessage(content="已为您查询到订单 SO20240001 的物流状态", id="a0"),
        HumanMessage(content="退款进度呢", id="u1"),
        AIMessage(content="请提供退款申请单号", id="a1"),
    ]
    for m in messages:
        score = score_message(m, messages, current_query="退款进度")
        assert 0.0 <= score <= 1.0


def test_score_recent_message_higher_than_old() -> None:
    """最新消息的 recency 分应高于早期消息（同等 relevance 下）。"""
    messages = [
        AIMessage(content="订单查询", id="old"),
        AIMessage(content="订单查询", id="new"),
    ]
    old_score = score_message(messages[0], messages, current_query="订单查询")
    new_score = score_message(messages[1], messages, current_query="订单查询")
    assert new_score > old_score


def test_score_relevance_to_current_query() -> None:
    """与 query 关键词重叠的消息分应高于不相关消息。"""
    messages = [
        AIMessage(content="课程 Python 入门", id="m0"),
        AIMessage(content="订单退款进度查询", id="m1"),
    ]
    course_score = score_message(messages[0], messages, current_query="课程 Python")
    order_score = score_message(messages[1], messages, current_query="课程 Python")
    assert course_score > order_score


def test_score_frequency_with_slots() -> None:
    """含槽位关键词且后续被引用的消息，frequency 维有分。"""
    slots = SessionSlots(last_order_no="SO20240001")
    messages = [
        AIMessage(content="订单 SO20240001 已发货", id="m0"),
        HumanMessage(content="那 SO20240001 什么时候到", id="m1"),
    ]
    # m0 含订单号，m1 引用了同一订单号 → frequency > 0
    score = score_message(messages[0], messages, current_query="到货", slots=slots)
    # recency 较低(0.1*0.3=0.03),relevance 部分重叠,frequency 贡献 > 0
    assert score > 0.0


def test_score_marked_important_boost() -> None:
    """显式标记 important 时分数应提升 0.1。"""
    msg = AIMessage(content="无关内容 xyz", id="m0")
    messages = [msg]
    base = score_message(msg, messages, current_query="查询", marked_important=False)
    marked = score_message(msg, messages, current_query="查询", marked_important=True)
    assert abs(marked - base - 0.1) < 1e-9


def test_compress_result_default_empty() -> None:
    result = CompressResult()
    assert result.kept == []
    assert result.summarized == []
    assert result.discarded == []
    assert result.summary == ""
    assert result.updated == []


# ---------------------- compress_by_importance 测试 ----------------------- #


def test_compress_no_action_when_under_threshold() -> None:
    """未超 token 预算和条数上限时不压缩。"""
    messages = [
        HumanMessage(content="查订单", id="u0"),
        AIMessage(content="请提供订单号", id="a0"),
    ]
    result = _compress(
        messages,
        current_query="查订单",
        token_budget=10000,
        max_messages=50,
    )
    assert result.kept == messages
    assert result.discarded == []
    assert result.summarized == []


def test_compress_old_turns_including_user_can_leave_window() -> None:
    """用户句不再豁免：低分旧轮次整轮丢正文（事实另抽进摘要）。"""
    blob = "无关内容" * 5
    messages = []
    for i in range(20):
        messages.append(HumanMessage(content=f"用户{i} {blob}", id=f"u{i}"))
        messages.append(AIMessage(content=f"助手{i} {blob}", id=f"a{i}"))
    result = _compress(
        messages,
        current_query="完全不相关的查询 xyz",
        token_budget=8000,
        always_keep_recent=3,
        discard_score=0.99,
        summarize_score=1.0,
        max_messages=10,
    )
    kept_ids = {m.id for m in result.kept}
    assert "u0" not in kept_ids
    assert "a0" not in kept_ids
    assert any(isinstance(m, HumanMessage) for m in result.discarded)


def test_compress_keeps_user_with_assistant_in_same_turn() -> None:
    """一轮内 user/assistant 同去同留。"""
    messages = []
    for i in range(8):
        messages.append(HumanMessage(content=f"闲聊{i} 今天天气如何", id=f"u{i}"))
        messages.append(AIMessage(content=f"闲聊回复{i} 今天晴朗", id=f"a{i}"))
    messages[4] = HumanMessage(content="查询订单 SO20240001 退款进度", id="u2")
    messages[5] = AIMessage(content="订单 SO20240001 退款已受理", id="a2")
    result = _compress(
        messages,
        current_query="SO20240001 退款",
        token_budget=8000,
        always_keep_recent=2,
        discard_score=0.15,
        summarize_score=0.35,
        max_messages=4,
    )
    kept_ids = {m.id for m in result.kept}
    assert "u2" in kept_ids
    assert "a2" in kept_ids


def test_compress_recent_snaps_to_turn_boundary() -> None:
    """最近 N 条若切在一轮中间，整轮保留。"""
    messages = []
    for i in range(6):
        messages.append(HumanMessage(content=f"问{i} 天气", id=f"u{i}"))
        messages.append(AIMessage(content=f"答{i} 晴", id=f"a{i}"))
    result = _compress(
        messages,
        current_query="无关查询 xyz",
        token_budget=8000,
        always_keep_recent=3,
        discard_score=0.99,
        max_messages=4,
    )
    kept_ids = {m.id for m in result.kept}
    assert {"u4", "a4", "u5", "a5"} <= kept_ids
    assert "u0" not in kept_ids


def _long_rag_answer(course: str = "Python入门课") -> str:
    modules = "\n".join(f"{i}. **模块{i}**：变量与函数专题内容补充说明" for i in range(1, 25))
    return (
        f"## {course}\n适合零基础学员入门。\n\n"
        f"### 课程模块\n{modules}\n\n"
        "### 学习路径\n- **入门**：模块1"
    )


def test_compact_rag_history_strips_modules() -> None:
    """课纲压缩只留课名+结论，短回复原样返回。"""
    rag = _long_rag_answer()
    stub = compact_rag_history(rag)
    assert "Python入门课" in stub
    assert "适合零基础" in stub
    assert "### 课程模块" not in stub
    assert compact_rag_history("短回复即可") == "短回复即可"


def test_compress_compacts_rag_under_window_threshold() -> None:
    """未超 token/条数阈值也压缩课纲，不等窗口爆掉。"""
    rag = _long_rag_answer()
    messages = [
        HumanMessage(content="Python入门课大纲", id="u0"),
        AIMessage(content=rag, id="a0"),
    ]
    result = _compress(
        messages,
        current_query="Python入门课大纲",
        token_budget=10000,
        max_messages=50,
    )
    kept_a0 = next(m for m in result.kept if m.id == "a0")
    assert "Python入门课" in str(kept_a0.content)
    assert "### 课程模块" not in str(kept_a0.content)
    assert any(m.id == "a0" for m in result.updated)
    assert result.discarded == []
    assert result.summarized == []


def test_build_message_updates_removes_discarded() -> None:
    """RemoveMessage 覆盖 discarded + summarized；updated 与本轮 AI 一并写入。"""
    result = CompressResult(
        kept=[AIMessage(content="本轮", id=None)],
        summarized=[AIMessage(content="旧摘要", id="a0")],
        discarded=[HumanMessage(content="旧丢弃", id="u0")],
        updated=[AIMessage(content="已压课纲", id="a1")],
        summary="滚动摘要",
    )
    updates = build_message_updates(result=result, stored_answer="stored")
    assert len(updates) == 4
    assert isinstance(updates[0], RemoveMessage) and updates[0].id == "u0"
    assert isinstance(updates[1], RemoveMessage) and updates[1].id == "a0"
    assert updates[2].id == "a1"
    assert str(updates[3].content) == "本轮"


async def _run_finalize_respond_turn(state: dict):
    return await finalize_respond_turn(state)


def test_finalize_respond_turn_stores_compacted_rag_keeps_full_answer() -> None:
    """服务层：写入 checkpoint 的是压缩课纲；answer 仍是全文。"""
    rag = _long_rag_answer()
    update = asyncio.run(
        _run_finalize_respond_turn(
            {
                "query": "Python入门课大纲",
                "answer": rag,
                "messages": [HumanMessage(content="Python入门课大纲", id="u0")],
                "tenant_id": "demo",
                "user_id": "u1",
                "session_id": "s1",
                "session_summary": "",
            }
        )
    )
    assert "### 课程模块" in update.answer
    stored = update.message_updates[-1]
    assert "Python入门课" in str(stored.content)
    assert "### 课程模块" not in str(stored.content)


def test_respond_stores_compacted_rag_keeps_full_answer(monkeypatch) -> None:
    """respond_node 薄封装：仍返回全文 answer，checkpoint 写入压缩课纲。"""
    from app.graph.nodes import respond_node

    monkeypatch.setattr("app.context.turn.upsert_session_archive", lambda *args, **kwargs: None)
    rag = _long_rag_answer()
    out = asyncio.run(
        respond_node(
            {
                "query": "Python入门课大纲",
                "answer": rag,
                "messages": [HumanMessage(content="Python入门课大纲", id="u0")],
                "tenant_id": "demo",
                "user_id": "u1",
                "session_id": "s1",
                "session_summary": "",
            }
        )
    )
    assert "### 课程模块" in out["answer"]
    stored = out["messages"][-1]
    assert "Python入门课" in str(stored.content)
    assert "### 课程模块" not in str(stored.content)


def test_respond_empty_answer_uses_llm_unavailable_template(monkeypatch) -> None:
    from app.graph.nodes import respond_node
    from app.llm_gateway import LLM_UNAVAILABLE_ANSWER

    monkeypatch.setattr("app.context.turn.upsert_session_archive", lambda *args, **kwargs: None)
    out = asyncio.run(
        respond_node(
            {
                "query": "Python入门课大纲",
                "answer": "",
                "messages": [HumanMessage(content="Python入门课大纲", id="u0")],
                "tenant_id": "demo",
                "user_id": "u1",
                "session_id": "s1",
                "session_summary": "",
            }
        )
    )
    assert out["answer"] == LLM_UNAVAILABLE_ANSWER


def test_compress_compacts_long_rag_in_kept() -> None:
    """高分保留的 RAG 长回复只留课名+结论，并写入 updated。"""
    rag = _long_rag_answer()
    assert "### 课程模块" in rag
    messages = []
    for i in range(6):
        messages.append(HumanMessage(content=f"闲聊{i} 天气", id=f"u{i}"))
        messages.append(AIMessage(content=f"闲聊回复{i} 晴", id=f"a{i}"))
    messages[0] = HumanMessage(content="Python入门课包含哪些模块", id="u0")
    messages[1] = AIMessage(content=rag, id="a0")
    result = _compress(
        messages,
        current_query="Python入门课模块",
        token_budget=8000,
        always_keep_recent=2,
        discard_score=0.0,
        summarize_score=0.0,
        max_messages=4,
    )
    kept_a0 = next(m for m in result.kept if m.id == "a0")
    assert "Python入门课" in str(kept_a0.content)
    assert "### 课程模块" not in str(kept_a0.content)
    assert any(m.id == "a0" for m in result.updated)


def test_compress_compacts_recent_rag_as_well() -> None:
    """最近轮次的 RAG 长回复同样压缩，不只压旧轮。"""
    rag = _long_rag_answer("数据分析课")
    messages = [
        HumanMessage(content="天气如何", id="u0"),
        AIMessage(content="晴", id="a0"),
        HumanMessage(content="数据分析课大纲", id="u1"),
        AIMessage(content=rag, id="a1"),
    ]
    result = _compress(
        messages,
        current_query="数据分析课",
        token_budget=8000,
        always_keep_recent=2,
        max_messages=2,
    )
    kept_a1 = next(m for m in result.kept if m.id == "a1")
    assert "数据分析课" in str(kept_a1.content)
    assert "### 课程模块" not in str(kept_a1.content)


def test_compress_system_message_always_kept() -> None:
    """仅 SystemMessage 因角色无条件保留；首条 assistant 不再豁免。"""
    blob = "内容" * 5
    messages: list = [SystemMessage(content="你是客服助手", id="sys")]
    messages.extend(AIMessage(content=f"{blob}{i}", id=f"a{i}") for i in range(20))
    result = _compress(
        messages,
        current_query="无关查询",
        token_budget=8000,
        always_keep_recent=3,
        discard_score=0.99,
        max_messages=10,
    )
    assert messages[0] in result.kept
    assert messages[1] not in result.kept


def test_compress_recent_n_always_kept() -> None:
    """最近 N 条无条件保留。"""
    blob = "内容" * 5
    messages = [AIMessage(content=f"{blob}{i}", id=f"a{i}") for i in range(30)]
    result = _compress(
        messages,
        current_query="无关",
        token_budget=8000,
        always_keep_recent=5,
        discard_score=0.99,
        max_messages=10,
    )
    for msg in messages[-5:]:
        assert msg in result.kept


def test_compress_low_score_assistant_discarded() -> None:
    """低分 assistant 消息进入 discarded。"""
    blob = "无关内容" * 5
    messages = []
    for i in range(15):
        messages.append(AIMessage(content=f"{blob}{i}", id=f"a{i}"))
    result = _compress(
        messages,
        current_query="完全不同的话题",
        token_budget=8000,
        always_keep_recent=3,
        discard_score=0.3,
        summarize_score=0.6,
        max_messages=10,
    )
    assert len(result.discarded) > 0
    assert all(isinstance(m, AIMessage) for m in result.discarded)


def test_compress_discarded_business_facts_still_summarized() -> None:
    """话题切走后低分旧轮次丢正文，但订单/退款约束仍写入 session_summary。"""
    chitchat = "今天天气真不错我们聊点别的" * 4
    messages = [
        HumanMessage(content="订单#20251114001 我要退款", id="u0"),
        AIMessage(content="好的，已记下退款诉求", id="a0"),
    ]
    for i in range(1, 12):
        messages.append(HumanMessage(content=f"{chitchat}{i}", id=f"u{i}"))
        messages.append(AIMessage(content=f"闲聊回复{i}", id=f"a{i}"))
    result = _compress(
        messages,
        current_query="Python入门课难不难",
        token_budget=8000,
        always_keep_recent=4,
        discard_score=0.3,
        summarize_score=0.6,
        max_messages=10,
    )
    discarded_ids = {m.id for m in result.discarded}
    kept_ids = {m.id for m in result.kept}
    assert "u0" in discarded_ids
    assert "a0" in discarded_ids
    assert "u0" not in kept_ids
    assert "a0" not in kept_ids
    assert "20251114001" in result.summary
    assert "用户要退款" in result.summary


def test_compress_discarded_chitchat_does_not_pollute_summary() -> None:
    """纯闲聊低分丢弃抽不到结构化事实，不凭空写入摘要。"""
    blob = "今天天气真好我们去散步" * 4
    messages = []
    for i in range(12):
        messages.append(HumanMessage(content=f"{blob}{i}", id=f"u{i}"))
        messages.append(AIMessage(content=f"闲聊回复{i} 好的", id=f"a{i}"))
    result = _compress(
        messages,
        current_query="完全无关的查询 xyzabc",
        token_budget=8000,
        always_keep_recent=4,
        discard_score=0.3,
        summarize_score=0.6,
        max_messages=10,
        use_llm_summary=False,
    )
    assert result.discarded
    assert result.summary == ""


def test_compress_fallback_keeps_at_most_recent_n() -> None:
    """保底：全低分时仍保留最近 N 条（min_keep 强制，可能超 token 预算）。"""
    huge = "很长的无关内容" * 50
    messages = [AIMessage(content=f"{huge}{i}", id=f"a{i}") for i in range(20)]
    result = _compress(
        messages,
        current_query="无关",
        token_budget=300,
        always_keep_recent=3,
        discard_score=0.99,
        summarize_score=1.0,
        max_messages=5,
    )
    assert len(result.kept) <= 3
    assert len(result.discarded) >= 16


def test_compress_fallback_overflow_is_summarized() -> None:
    """保底溢出进入 summarized 并写入 summary，而不是 discarded。"""
    blob = "订单号 SO20240001 课程Python入门 " * 20
    messages = []
    for i in range(12):
        messages.append(HumanMessage(content=f"用户{i} {blob}", id=f"u{i}"))
        messages.append(AIMessage(content=f"助手{i} {blob}", id=f"a{i}"))
    result = _compress(
        messages,
        current_query="订单",
        token_budget=400,
        always_keep_recent=4,
        discard_score=0.0,
        summarize_score=0.0,
        max_messages=5,
        max_summary_chars=800,
    )
    assert result.summarized
    assert result.summary
    assert result.discarded == []
    summarized_ids = {m.id for m in result.summarized}
    kept_ids = {m.id for m in result.kept}
    assert summarized_ids.isdisjoint(kept_ids)
    assert "SO20240001" in result.summary or "Python" in result.summary


def test_compress_fallback_does_not_split_turn() -> None:
    """保底裁剪按整轮进行，不会只留 assistant 丢掉对应 user。"""
    blob = "订单号 SO20240001 课程Python入门 " * 15
    messages = []
    for i in range(10):
        messages.append(HumanMessage(content=f"用户{i} {blob}", id=f"u{i}"))
        messages.append(AIMessage(content=f"助手{i} {blob}", id=f"a{i}"))
    result = _compress(
        messages,
        current_query="订单",
        token_budget=350,
        always_keep_recent=4,
        discard_score=0.0,
        summarize_score=0.0,
        max_messages=4,
    )
    kept_ids = {m.id for m in result.kept}
    for mid in list(kept_ids):
        if mid.startswith("a"):
            assert f"u{mid[1:]}" in kept_ids
        if mid.startswith("u"):
            assert f"a{mid[1:]}" in kept_ids


def test_compress_summarized_produces_summary() -> None:
    """中分 assistant 消息被摘要化，summary 非空。"""
    messages = []
    for i in range(15):
        messages.append(AIMessage(content=f"订单 SO2024000{i} 已处理", id=f"a{i}"))
    messages.append(HumanMessage(content="查订单", id="u_last"))
    result = _compress(
        messages,
        current_query="查订单",
        token_budget=200,
        always_keep_recent=2,
        discard_score=0.0,
        summarize_score=1.0,
        max_messages=5,
    )
    assert result.summary  # 非空


def test_compress_empty_messages() -> None:
    result = _compress([], token_budget=1000)
    assert result.kept == []
    assert result.summary == ""


def test_compress_preserves_message_order_in_kept() -> None:
    """kept 列表保持原始时间顺序。"""
    blob = "内容" * 15
    messages = [
        AIMessage(content=f"早期 {blob}", id="a0"),
        HumanMessage(content="用户问题 订单", id="u0"),
        AIMessage(content=f"近期回答 {blob}", id="a1"),
    ]
    result = _compress(
        messages,
        current_query="订单",
        token_budget=100,
        always_keep_recent=2,
        max_messages=2,
    )
    kept_ids = [m.id for m in result.kept]
    # 保持相对顺序
    for i in range(len(kept_ids) - 1):
        original_idx = [m.id for m in messages].index(kept_ids[i])
        next_idx = [m.id for m in messages].index(kept_ids[i + 1])
        assert original_idx < next_idx


# ------------------------ _llm_batch_summarize 测试 ----------------------- #

from unittest.mock import AsyncMock, MagicMock, patch


def _mock_llm(response_text: str) -> MagicMock:
    """构造一个 mock LLM，ainvoke 返回指定文本。"""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=AIMessage(content=response_text))
    return mock


def test_llm_summary_returns_generated_text() -> None:
    """LLM 成功时返回 LLM 生成的摘要。"""
    messages = [
        HumanMessage(content="查订单 SO20240001 退款", id="u0"),
        AIMessage(content="订单 SO20240001 退款已处理", id="a0"),
    ]
    mock = _mock_llm("- 用户询问了订单 SO20240001 的退款进度\n- 助手回复退款已处理")
    with patch("app.llm.get_chat_model", return_value=mock):
        result = _llm_summarize(messages, existing_summary="")
    assert "SO20240001" in result
    assert "退款" in result
    mock.ainvoke.assert_called_once()


def test_llm_summary_falls_back_on_exception() -> None:
    """LLM 异常时回退到 fold_session_summary（规则截断）。"""
    messages = [
        HumanMessage(content="查订单 SO20240001 退款", id="u0"),
        AIMessage(content="订单 SO20240001 退款已处理", id="a0"),
    ]
    mock = MagicMock()
    mock.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 不可用"))
    with patch("app.llm.get_chat_model", return_value=mock):
        result = _llm_summarize(messages, existing_summary="", max_chars=200)
    # 规则合并仍会保留订单号
    assert "20240001" in result


def test_llm_summary_appends_to_existing() -> None:
    """已有摘要时 LLM 新事实合并进去。"""
    messages = [AIMessage(content="订单 SO20240002 已发货", id="a0")]
    mock = _mock_llm("助手回复订单 SO20240002 已发货")
    with patch("app.llm.get_chat_model", return_value=mock):
        result = _llm_summarize(messages, existing_summary="更早：用户查了 SO20240001")
    assert "SO20240001" in result
    assert "SO20240002" in result


def test_llm_summary_empty_messages_returns_existing() -> None:
    """空消息列表时仍对已有摘要做未决关闭。"""
    result = _llm_summarize([], existing_summary="已确认：用户要退款")
    assert "用户要退款" in result
    cleared = _llm_summarize(
        [],
        existing_summary="已确认：用户要退款\n未决：待提供订单号",
        slots=SessionSlots(last_order_no="20251114001"),
    )
    parsed = StructuredSummary.parse(cleared)
    assert "待提供订单号" not in parsed.pending
    assert "用户要退款" in parsed.confirmed


def test_llm_summary_truncates_to_max_chars() -> None:
    """超过 max_chars 时截断，并保留尾部新事实。"""
    messages = [AIMessage(content="订单 20251114001 用户要退款", id="a0")]
    mock = _mock_llm("旧摘要AAAA" * 40 + "订单SO20249999已退款")
    with patch("app.llm.get_chat_model", return_value=mock):
        result = _llm_summarize(messages, existing_summary="", max_chars=80)
    assert len(result) <= 80
    assert "退款" in result


def test_llm_summary_skips_when_no_new_facts() -> None:
    """溢出没有新约束/结论时不调用 LLM。"""
    messages = [
        HumanMessage(content="今天天气真好", id="u0"),
        AIMessage(content="今天晴朗", id="a0"),
    ]
    mock = _mock_llm("不该出现")
    with patch("app.llm.get_chat_model", return_value=mock) as factory:
        result = _llm_summarize(messages, existing_summary="已确认：用户要退款")
    factory.assert_not_called()
    assert "用户要退款" in result


def test_llm_summary_disables_cache() -> None:
    """摘要 LLM 关闭短缓存，避免不同批次碰撞。"""
    messages = [
        HumanMessage(content="查订单 20240001 退款", id="u0"),
        AIMessage(content="退款已受理", id="a0"),
    ]
    mock = _mock_llm("已确认：用户要退款")
    with patch("app.llm.get_chat_model", return_value=mock) as factory:
        _llm_summarize(messages, existing_summary="")
    assert factory.call_args.kwargs.get("cache_enabled") is False


def test_compress_with_llm_summary_uses_llm_path() -> None:
    """_compress(use_llm_summary=True) 走 LLM 摘要路径。"""
    messages = []
    for i in range(15):
        messages.append(AIMessage(content=f"订单 SO2024000{i} 已处理", id=f"a{i}"))
    messages.append(HumanMessage(content="查订单", id="u_last"))
    mock = _mock_llm("- 助手处理了订单 SO2024000x")
    with patch("app.llm.get_chat_model", return_value=mock):
        result = _compress(
            messages,
            current_query="查订单",
            token_budget=200,
            always_keep_recent=2,
            discard_score=0.0,
            summarize_score=1.0,
            max_messages=5,
            use_llm_summary=True,
        )
    assert result.summary
    mock.ainvoke.assert_called_once()


def test_compress_with_llm_summary_falls_back_on_failure() -> None:
    """LLM 失败时 compress_by_importance 仍能产出摘要（规则回退）。"""
    messages = []
    for i in range(15):
        messages.append(AIMessage(content=f"订单 SO2024000{i} 已处理", id=f"a{i}"))
    messages.append(HumanMessage(content="查订单", id="u_last"))
    mock = MagicMock()
    mock.ainvoke = AsyncMock(side_effect=RuntimeError("网络错误"))
    with patch("app.llm.get_chat_model", return_value=mock):
        result = _compress(
            messages,
            current_query="查订单",
            token_budget=200,
            always_keep_recent=2,
            discard_score=0.0,
            summarize_score=1.0,
            max_messages=5,
            use_llm_summary=True,
        )
    assert result.summary
    assert "2024000" in result.summary


def test_default_history_limits_align_with_compress_keep() -> None:
    """窗口默认保底与压缩层 always_keep_recent 对齐为 10；预算默认 8000。"""
    settings = get_settings()
    assert settings.history_min_messages == 10
    assert settings.history_token_budget == 8000
    assert settings.context_always_keep_recent == 10
    assert settings.context_summary_use_llm is True
    assert DEFAULT_HISTORY_MIN_MESSAGES == 10
    assert DEFAULT_HISTORY_TOKEN_BUDGET == 8000


def test_select_default_min_keep_retains_ten_when_over_budget() -> None:
    """超预算时默认至少保留 10 条（不再是 2）。"""
    messages = []
    for index in range(20):
        messages.append(HumanMessage(content=f"短问{index}", id=f"u{index}"))
        messages.append(AIMessage(content=f"短答{index}", id=f"a{index}"))
    kept = select_messages_by_tokens(messages, token_budget=30)
    assert len(kept) >= 10
    assert kept[-1].id == "a19"


def test_llm_summary_invokes_on_soft_constraint_without_regex_hit() -> None:
    """正则未抽到新事实但有业务信号（如软约束）时仍调用 LLM。"""
    messages = [
        HumanMessage(content="别推荐付费课，我只想看免费资料", id="u0"),
        AIMessage(content="好的，我记下了你的偏好", id="a0"),
    ]
    existing = StructuredSummary.parse("已确认：用户要退款")
    delta = StructuredSummary()  # 模拟规则无新事实
    assert not delta.novel_against(existing)
    assert _should_invoke_llm_summary(messages, existing=existing, delta=delta)

    mock = _mock_llm("已确认：用户要退款；别推荐付费课")
    with patch("app.llm.get_chat_model", return_value=mock) as factory:
        result = _llm_summarize(messages, existing_summary="已确认：用户要退款")
    factory.assert_called_once()
    assert "退款" in result


def test_llm_summary_still_skips_short_chitchat() -> None:
    """极短闲聊无业务信号时仍跳过 LLM。"""
    messages = [
        HumanMessage(content="嗯", id="u0"),
        AIMessage(content="好的", id="a0"),
    ]
    existing = StructuredSummary.parse("已确认：用户要退款")
    delta = StructuredSummary()
    assert not _should_invoke_llm_summary(messages, existing=existing, delta=delta)


def test_token_estimate_deepseek_uses_explicit_fallback() -> None:
    """DeepSeek 模型名无法被 tiktoken 识别时，走显式 cl100k_fallback 策略。"""
    _load_encoding.cache_clear()
    real = get_settings()
    patched = real.model_copy(
        update={
            "deepseek_model": "deepseek-chat",
            "history_tokenizer_name": "",
            "history_token_estimate_ratio": 1.0,
        }
    )
    with patch("app.config.get_settings", return_value=patched):
        strategy = token_estimate_strategy()
    assert strategy.startswith("cl100k_fallback")
    _load_encoding.cache_clear()


def test_token_estimate_ratio_scales_count() -> None:
    """HISTORY_TOKEN_ESTIMATE_RATIO 会缩放估算结果。"""
    _load_encoding.cache_clear()
    real = get_settings()
    base = real.model_copy(
        update={
            "history_tokenizer_name": "cl100k_base",
            "history_token_estimate_ratio": 1.0,
        }
    )
    scaled = real.model_copy(
        update={
            "history_tokenizer_name": "cl100k_base",
            "history_token_estimate_ratio": 2.0,
        }
    )
    text = "查询订单退款进度与课程大纲"
    with patch("app.config.get_settings", return_value=base):
        n1 = count_tokens(text)
    _load_encoding.cache_clear()
    with patch("app.config.get_settings", return_value=scaled):
        n2 = count_tokens(text)
    assert n2 == n1 * 2
    _load_encoding.cache_clear()
