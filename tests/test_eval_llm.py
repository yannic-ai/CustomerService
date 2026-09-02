import os

import pytest

from app.agents.intent import recognize_intent
from app.config import get_settings
from app.evalsets import load_jsonl
from tests.test_eval_routing import _check_row, _slots


@pytest.mark.llm
def test_routing_llm_golden() -> None:
    if os.environ.get("RUN_LLM_EVAL") != "1":
        pytest.skip("nightly：设置 RUN_LLM_EVAL=1 后才跑真实 LLM 路由评测")
    if not get_settings().llm_enabled:
        pytest.skip("未配置 DEEPSEEK_API_KEY")
    failures: list[str] = []
    for row in load_jsonl("routing.jsonl"):
        decision = recognize_intent(row["query"], slots=_slots(row))
        error = _check_row(row, decision, llm=True)
        if error:
            failures.append(error)
    assert not failures, "\n".join(failures)
