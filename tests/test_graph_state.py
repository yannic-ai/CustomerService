"""CSState 按 identity / route / memory / result 分组；checkpoint 仍扁平。"""

from __future__ import annotations

from typing import get_type_hints

from app.graph.state import (
    CSState,
    IdentityState,
    IdentityView,
    MEMORY_SLOT_KEYS,
    MemoryState,
    MemoryView,
    RESULT_TURN_RESET,
    ROUTE_TURN_RESET,
    ResultState,
    ResultView,
    RouteState,
    RouteView,
    reset_turn_fields,
)


def test_csstate_merges_group_fields() -> None:
    keys = set(get_type_hints(CSState))
    assert set(get_type_hints(IdentityState)) <= keys
    assert set(get_type_hints(RouteState)) <= keys
    assert set(get_type_hints(MemoryState)) <= keys
    assert set(get_type_hints(ResultState)) <= keys
    assert "intent" in keys and "last_order_no" in keys and "answer" in keys


def test_reset_turn_clears_route_and_result_not_memory() -> None:
    cleared = reset_turn_fields()
    assert set(ROUTE_TURN_RESET) <= set(cleared)
    assert set(RESULT_TURN_RESET) <= set(cleared)
    for key in MEMORY_SLOT_KEYS:
        assert key not in cleared
    assert "session_summary" not in cleared
    assert "query" not in cleared
    assert "messages" not in cleared


def test_views_round_trip_flat_checkpoint_keys() -> None:
    state = {
        "query": "查一下退款",
        "user_id": "u10001",
        "tenant_id": "demo",
        "session_id": "s1",
        "intent": "order",
        "target": "tool",
        "confidence": 0.9,
        "order_no": "20251114001",
        "last_order_no": "20251114001",
        "session_summary": "用户在问退款",
        "answer": "退款处理中",
        "blocked": False,
    }
    ident = IdentityView.from_state(state)
    route = RouteView.from_state(state)
    memory = MemoryView.from_state(state)
    result = ResultView.from_state(state)
    assert ident.user_id == "u10001"
    assert route.intent == "order" and route.order_no == "20251114001"
    assert memory.last_order_no == "20251114001"
    assert result.answer == "退款处理中"
    restored = memory.as_state()
    assert restored["last_order_no"] == "20251114001"
    assert restored["session_summary"] == "用户在问退款"
