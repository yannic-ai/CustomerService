import os

import pytest

from app.evals.ragas_runner import run_ragas_eval


@pytest.mark.llm
def test_ragas_faithfulness_nightly() -> None:
    if os.environ.get("RUN_LLM_EVAL") != "1":
        pytest.skip("nightly：设置 RUN_LLM_EVAL=1 后才跑 RAGAS 评测")
    pytest.importorskip("ragas")
    failures = run_ragas_eval()
    assert not failures, "\n".join(failures)
