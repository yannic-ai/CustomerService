"""本轮出口：回复文本与结构化报告。"""

from __future__ import annotations

from typing import Any, TypedDict


class ResultState(TypedDict, total=False):
    """只活在本轮的答案。新增出口字段写这里。"""

    answer: str
    structured: dict[str, Any] | None
    visualization: str | None
    oss_url: str | None


RESULT_TURN_RESET: dict[str, Any] = {
    "answer": "",
    "structured": None,
    "visualization": None,
    "oss_url": None,
}
