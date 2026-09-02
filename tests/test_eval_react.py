import asyncio
import os

import pytest

from app.agents.react import tool_order_query
from app.concurrency import achat
from app.evalsets import load_jsonl


def _chat(message, **kwargs):
    return asyncio.run(achat(message, **kwargs))


def _tool_names_from_messages(messages: list) -> list[str]:
    names: list[str] = []
    for message in messages:
        calls = getattr(message, "tool_calls", None) or []
        for call in calls:
            if isinstance(call, dict):
                name = call.get("name") or ""
            else:
                name = getattr(call, "name", "") or ""
            if name:
                names.append(str(name))
    return names


def test_react_offline_mixed_answer() -> None:
    failures: list[str] = []
    for row in load_jsonl("react.jsonl"):
        if row.get("expect_tools_any") or row.get("expect_tool_message_contains"):
            continue
        response = _chat(
            row["query"],
            user_id=row.get("user_id") or "anonymous",
            tenant_id=row.get("tenant_id") or "demo",
        )
        if row.get("expect_intent") and response.intent != row["expect_intent"]:
            failures.append(f"{row['id']}: intent {response.intent!r} != {row['expect_intent']!r}")
        answer = response.answer or ""
        order_ok = any(token in answer for token in row.get("expect_answer_contains_any") or [""])
        if row.get("expect_answer_contains_any") and not order_ok:
            failures.append(f"{row['id']}: answer missing order tokens: {answer[:120]!r}")
        for token in row.get("expect_course_tokens") or []:
            if token not in answer:
                failures.append(f"{row['id']}: answer missing course token {token!r}")
    assert not failures, "\n".join(failures)


def test_react_tool_missing_order_no() -> None:
    for row in load_jsonl("react.jsonl"):
        needle = row.get("expect_tool_message_contains")
        if not needle:
            continue
        payload = tool_order_query.invoke({"order_no": "", "query": row["query"]})
        assert needle in payload


@pytest.mark.llm
def test_react_llm_tool_selection() -> None:
    if os.environ.get("RUN_LLM_EVAL") != "1":
        pytest.skip("nightly：设置 RUN_LLM_EVAL=1 后才跑 ReACT 工具轨迹")
    from app.config import get_settings
    from app.agents.react import build_react_agent

    if not get_settings().llm_enabled:
        pytest.skip("未配置 DEEPSEEK_API_KEY")

    agent = build_react_agent(user_id="u10001", tenant_id="demo")
    assert agent is not None
    failures: list[str] = []
    for row in load_jsonl("react.jsonl"):
        expected = row.get("expect_tools_any")
        if not expected:
            continue
        result = agent.invoke(
            {"messages": [{"role": "user", "content": row["query"]}]},
        )
        messages = result.get("messages") or []
        names = _tool_names_from_messages(messages)
        if not any(name in names for name in expected):
            failures.append(f"{row['id']}: tools {names} missing any of {expected}")
        # 缺单号场景由 tool 单测覆盖；此处额外确保不会无意义连打
        if names.count("tool_order_query") > 3:
            failures.append(f"{row['id']}: tool_order_query called too many times ({names})")
    assert not failures, "\n".join(failures)
