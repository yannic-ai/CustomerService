import asyncio

from app.agents.intent import recognize_intent
from app.config import get_settings
from app.context import SessionSlots
from app.evalsets import load_jsonl


def _slots(row: dict) -> SessionSlots:
    raw = row.get("slots") or {}
    return SessionSlots(
        last_intent=raw.get("last_intent"),
        last_order_no=raw.get("last_order_no"),
        last_course_query=raw.get("last_course_query"),
        last_course_name=raw.get("last_course_name"),
        handoff_pending=bool(raw.get("handoff_pending")),
    )


def _check_row(row: dict, decision, *, llm: bool) -> str | None:
    expected_intent = row.get("expect_intent_llm") if llm else None
    expected_intent = expected_intent or row["expect_intent"]
    if decision.intent != expected_intent:
        return f"{row['id']}: intent {decision.intent!r} != {expected_intent!r}"
    if "expect_target" in row and decision.target != row["expect_target"]:
        return f"{row['id']}: target {decision.target!r} != {row['expect_target']!r}"
    if "expect_order_no" in row:
        expected_no = row["expect_order_no"]
        if (decision.order_no or None) != expected_no:
            return f"{row['id']}: order_no {decision.order_no!r} != {expected_no!r}"
    return None


def test_routing_keyword_golden(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "deepseek_api_key", "")
    failures: list[str] = []
    for row in load_jsonl("routing.jsonl"):
        decision = asyncio.run(recognize_intent(row["query"], slots=_slots(row)))
        error = _check_row(row, decision, llm=False)
        if error:
            failures.append(error)
    assert not failures, "\n".join(failures)
