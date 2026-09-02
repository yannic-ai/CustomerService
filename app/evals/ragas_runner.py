"""RAGAS nightly 评测：生成质量（faithfulness / answer_relevancy）。"""

from __future__ import annotations

import asyncio
from typing import Any

from app.agents.rag import consult_course, format_course_answer, retrieve_course_docs
from app.context import expand_query
from app.evals.harvest import maybe_harvest_eval_failure
from app.evalsets import load_jsonl
from app.tenancy import set_current_tenant

FAITHFULNESS_THRESHOLD = 0.7
ANSWER_RELEVANCY_THRESHOLD = 0.7


def _eval_context(row: dict[str, Any]) -> dict[str, str | None]:
    expand = row.get("expand") or {}
    return {
        "last_course_query": expand.get("last_course_query"),
        "last_course_name": expand.get("last_course_name"),
    }


def _course_code(row: dict[str, Any]) -> str | None:
    code = row.get("course_code")
    return str(code).strip() if code else None


async def _build_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in load_jsonl("faithfulness.jsonl"):
        if row.get("expect_not_found"):
            continue
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
        result = await consult_course(
            query,
            tenant_id=tenant,
            last_course_query=ctx.get("last_course_query"),
            last_course_name=ctx.get("last_course_name"),
            course_code=course_code,
        )
        answer = format_course_answer(result) if result.modules else (result.summary or result.course_name or "")
        samples.append(
            {
                "id": row["id"],
                "question": query,
                "answer": answer,
                "contexts": [doc.page_content for doc in docs],
            }
        )
    return samples


def run_ragas_eval() -> list[str]:
    """跑 RAGAS 指标；返回失败说明列表。未安装 ragas 时抛 ImportError。"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    from app.llm import get_chat_model

    samples = asyncio.run(_build_samples())
    if not samples:
        return []

    rows_by_id = {
        row["id"]: row
        for row in load_jsonl("faithfulness.jsonl")
        if not row.get("expect_not_found")
    }
    ids = [sample.pop("id") for sample in samples]
    dataset = Dataset.from_list(samples)
    llm = LangchainLLMWrapper(get_chat_model(temperature=0, usage_tag="rag"))
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
    )
    frame = result.to_pandas()
    failures: list[str] = []
    for row_id, faith, relevancy in zip(
        ids,
        frame["faithfulness"].tolist(),
        frame["answer_relevancy"].tolist(),
        strict=True,
    ):
        if float(faith) < FAITHFULNESS_THRESHOLD:
            msg = f"{row_id}: faithfulness={float(faith):.2f}"
            failures.append(msg)
            maybe_harvest_eval_failure(
                test_name="faithfulness",
                case_id=row_id,
                row=rows_by_id.get(row_id, {"id": row_id}),
                message=msg,
            )
        if float(relevancy) < ANSWER_RELEVANCY_THRESHOLD:
            msg = f"{row_id}: answer_relevancy={float(relevancy):.2f}"
            failures.append(msg)
            maybe_harvest_eval_failure(
                test_name="faithfulness",
                case_id=row_id,
                row=rows_by_id.get(row_id, {"id": row_id}),
                message=msg,
            )
    return failures
