"""失败样本回流 evals/pending 单测。"""

from __future__ import annotations

import json

from app.evals.harvest import (
    TurnSnapshot,
    append_pending,
    cache_turn_snapshot,
    merge_pending,
    record_feedback_down,
    record_retrieval_miss,
    reset_harvest_state_for_tests,
)


def test_append_pending_dedupes(tmp_path, monkeypatch) -> None:
    reset_harvest_state_for_tests()
    settings = __import__("app.config", fromlist=["get_settings"]).get_settings()
    monkeypatch.setattr(settings, "rag_eval_harvest_enabled", True)
    monkeypatch.setattr(settings, "rag_eval_harvest_dir", str(tmp_path))

    case = {
        "source": "rag_empty",
        "query": "烘焙蛋糕课讲什么",
        "tenant_id": "demo",
        "expect_empty": True,
    }
    assert append_pending(case, kind="rag") is True
    assert append_pending(case, kind="rag") is False

    path = tmp_path / "rag.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["query"] == "烘焙蛋糕课讲什么"


def test_record_feedback_down_uses_snapshot(tmp_path, monkeypatch) -> None:
    reset_harvest_state_for_tests()
    settings = __import__("app.config", fromlist=["get_settings"]).get_settings()
    monkeypatch.setattr(settings, "rag_eval_harvest_enabled", True)
    monkeypatch.setattr(settings, "rag_eval_harvest_dir", str(tmp_path))

    cache_turn_snapshot(
        TurnSnapshot(
            trace_id="trace-1",
            tenant_id="demo",
            original_query="Python入门课怎么样",
            search_query="Python入门课 怎么样 课程介绍",
            intent="course_consult",
            course_name="Python入门课",
            sources=["python-intro.md"],
            rewrite_reason="abstract",
            answer_preview="示例回答",
        )
    )
    assert record_feedback_down("trace-1", comment="不准确") is True

    path = tmp_path / "faithfulness.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["source"] == "user_feedback_down"
    assert row["meta"]["comment"] == "不准确"


def test_harvest_disabled_writes_nothing(tmp_path, monkeypatch) -> None:
    reset_harvest_state_for_tests()
    settings = __import__("app.config", fromlist=["get_settings"]).get_settings()
    monkeypatch.setattr(settings, "rag_eval_harvest_enabled", False)
    monkeypatch.setattr(settings, "rag_eval_harvest_dir", str(tmp_path))

    assert (
        record_retrieval_miss(
            tenant_id="demo",
            query="未知课程",
            reason="rag_empty",
        )
        is False
    )
    assert not (tmp_path / "rag.jsonl").exists()


def test_merge_pending_dry_run(tmp_path, monkeypatch) -> None:
    reset_harvest_state_for_tests()
    settings = __import__("app.config", fromlist=["get_settings"]).get_settings()
    monkeypatch.setattr(settings, "rag_eval_harvest_dir", str(tmp_path))
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "rag.jsonl").write_text(
        '{"id":"pending-x","source":"rag_empty","query":"q","tenant_id":"demo"}\n',
        encoding="utf-8",
    )
    preview = merge_pending(apply=False)
    assert preview
    assert (tmp_path / "rag.jsonl").read_text(encoding="utf-8").strip()
