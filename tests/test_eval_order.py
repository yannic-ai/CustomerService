import asyncio

from app.concurrency import achat
from app.evalsets import load_jsonl


def _chat(message, **kwargs):
    return asyncio.run(achat(message, **kwargs))


def test_order_facts_golden() -> None:
    failures: list[str] = []
    for row in load_jsonl("order.jsonl"):
        response = _chat(
            row["query"],
            user_id=row.get("user_id") or "anonymous",
            tenant_id=row.get("tenant_id") or "demo",
        )
        answer = response.answer or ""
        structured = response.structured or {}

        if row.get("expect_blocked"):
            if not response.blocked or "拦截" not in answer:
                failures.append(
                    f"{row['id']}: expected blocked, blocked={response.blocked} answer={answer[:80]!r}"
                )
        if row.get("expect_intent") and response.intent != row["expect_intent"]:
            failures.append(
                f"{row['id']}: intent {response.intent!r} != {row['expect_intent']!r}"
            )

        if row.get("expect_missing_order"):
            if "未识别到订单号" not in answer and "订单编号" not in answer:
                failures.append(f"{row['id']}: expected missing-order message, got {answer[:80]!r}")
        if row.get("expect_not_found"):
            if "未找到订单" not in answer:
                failures.append(f"{row['id']}: expected 未找到订单, got {answer[:80]!r}")

        expected_no = row.get("expect_order_no")
        if expected_no:
            got = structured.get("order_no") if isinstance(structured, dict) else None
            if got != expected_no and expected_no not in answer:
                failures.append(f"{row['id']}: order_no {got!r} != {expected_no!r}")

        status_needle = row.get("expect_status_contains")
        if status_needle and isinstance(structured, dict):
            status = str(structured.get("current_status") or "")
            if status_needle not in status and status_needle not in answer:
                failures.append(f"{row['id']}: status missing {status_needle!r} (status={status!r})")

        for token in row.get("expect_answer_contains") or []:
            if token not in answer:
                failures.append(f"{row['id']}: answer missing {token!r}")

        for token in row.get("forbid_answer_contains") or []:
            if token in answer:
                failures.append(f"{row['id']}: answer leaked {token!r}")

        for bad in row.get("forbid_order_nos") or []:
            if bad in answer:
                failures.append(f"{row['id']}: answer invented order {bad}")

        if row.get("forbid_structured_order"):
            if isinstance(structured, dict) and structured.get("order_no"):
                failures.append(
                    f"{row['id']}: structured should be empty, got order_no={structured.get('order_no')!r}"
                )
            if response.visualization:
                failures.append(f"{row['id']}: unexpected visualization without order_no")

    assert not failures, "\n".join(failures)
