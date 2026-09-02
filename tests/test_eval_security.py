from app.evalsets import load_jsonl
from app.security.filters import inspect_input


def test_security_golden() -> None:
    failures: list[str] = []
    for row in load_jsonl("security.jsonl"):
        query = row.get("query") or ""
        repeat = row.get("query_repeat") or {}
        if repeat:
            query = str(repeat.get("char") or "x") * int(repeat["count"])
        verdict = inspect_input(query)
        expected = bool(row["expect_allowed"])
        if verdict.allowed is not expected:
            failures.append(
                f"{row['id']}: allowed={verdict.allowed} expected={expected} reason={verdict.reason!r}"
            )
        needle = row.get("expect_reason_contains")
        if needle and needle not in (verdict.reason or ""):
            failures.append(f"{row['id']}: reason missing {needle!r} ({verdict.reason!r})")
        for token in [row.get("expect_sanitized_contains")] if row.get("expect_sanitized_contains") else []:
            if token not in (verdict.sanitized_text or ""):
                failures.append(f"{row['id']}: sanitized missing {token!r}")
        for token in [row.get("forbid_sanitized_contains")] if row.get("forbid_sanitized_contains") else []:
            if token in (verdict.sanitized_text or ""):
                failures.append(f"{row['id']}: sanitized leaked {token!r}")
    assert not failures, "\n".join(failures)
