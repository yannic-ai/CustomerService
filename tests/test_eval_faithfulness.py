import asyncio

from app.agents.rag import consult_course, retrieve_course_docs
from app.context import expand_query
from app.evals.harvest import maybe_harvest_eval_failure
from app.evalsets import load_jsonl
from app.tenancy import set_current_tenant


def _eval_context(row: dict) -> dict:
    expand = row.get("expand") or {}
    return {
        "last_course_query": expand.get("last_course_query"),
        "last_course_name": expand.get("last_course_name"),
    }


def _course_code(row: dict) -> str | None:
    code = row.get("course_code")
    return str(code).strip() if code else None


def test_faithfulness_golden() -> None:
    failures: list[str] = []
    for row in load_jsonl("faithfulness.jsonl"):
        tenant = row.get("tenant_id") or "demo"
        set_current_tenant(tenant)
        query = row["query"]
        ctx = _eval_context(row)
        course_code = _course_code(row)
        search_query = expand_query(
            query,
            None,
            last_course_query=ctx.get("last_course_query"),
            last_course_name=ctx.get("last_course_name"),
        )
        docs = retrieve_course_docs(search_query, tenant_id=tenant, course_code=course_code)
        result = asyncio.run(
            consult_course(
                query,
                tenant_id=tenant,
                last_course_query=ctx.get("last_course_query"),
                last_course_name=ctx.get("last_course_name"),
                course_code=course_code,
            )
        )

        if row.get("expect_not_found"):
            if result.course_name != "未找到课程":
                msg = f"{row['id']}: expected 未找到课程, got {result.course_name!r}"
                failures.append(msg)
                maybe_harvest_eval_failure(test_name="faithfulness", case_id=row["id"], row=row, message=msg)
            if result.modules:
                msg = f"{row['id']}: expected no modules for empty retrieval, got {len(result.modules)}"
                failures.append(msg)
                maybe_harvest_eval_failure(test_name="faithfulness", case_id=row["id"], row=row, message=msg)
            if docs:
                sources = sorted({str(doc.metadata.get("source")) for doc in docs})
                msg = f"{row['id']}: expected empty retrieval, got {sources}"
                failures.append(msg)
                maybe_harvest_eval_failure(test_name="faithfulness", case_id=row["id"], row=row, message=msg)
            continue

        needle = row.get("expect_course_contains")
        if needle and needle not in (result.course_name or "") and needle not in (result.summary or ""):
            msg = (
                f"{row['id']}: course text missing {needle!r} "
                f"(name={result.course_name!r})"
            )
            failures.append(msg)
            maybe_harvest_eval_failure(test_name="faithfulness", case_id=row["id"], row=row, message=msg)

        if docs and not result.evidence:
            msg = f"{row['id']}: expected evidence sections when retrieval hits"
            failures.append(msg)
            maybe_harvest_eval_failure(test_name="faithfulness", case_id=row["id"], row=row, message=msg)

        grounded = " ".join(
            [
                result.course_name or "",
                result.summary or "",
                *(module.name for module in result.modules),
            ]
        )
        for token in row.get("forbid_tokens") or []:
            if token in grounded:
                msg = f"{row['id']}: grounded text leaked {token!r}"
                failures.append(msg)
                maybe_harvest_eval_failure(test_name="faithfulness", case_id=row["id"], row=row, message=msg)

        if row.get("require_modules_in_context"):
            context = "\n".join(doc.page_content for doc in docs)
            if not docs:
                msg = f"{row['id']}: no retrieval context for faithfulness check"
                failures.append(msg)
                maybe_harvest_eval_failure(test_name="faithfulness", case_id=row["id"], row=row, message=msg)
                continue
            for module in result.modules:
                name = module.name.replace("模块", " ").strip()
                tokens = [t for t in name.replace("：", " ").split() if len(t) >= 2]
                if not tokens:
                    continue
                if not any(token in context for token in tokens):
                    msg = f"{row['id']}: module {module.name!r} not grounded in retrieval"
                    failures.append(msg)
                    maybe_harvest_eval_failure(test_name="faithfulness", case_id=row["id"], row=row, message=msg)
    assert not failures, "\n".join(failures)


def test_empty_retrieval_returns_not_found(monkeypatch) -> None:
    set_current_tenant("demo")
    monkeypatch.setattr("app.agents.rag.retrieve_course_docs", lambda *a, **k: [])
    result = asyncio.run(consult_course("任意课程", tenant_id="demo"))
    assert result.course_name == "未找到课程"
    assert not result.modules
