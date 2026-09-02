from app.agents.rag import retrieve_course_docs
from app.config import get_settings
from app.context import expand_query
from app.evals.harvest import maybe_harvest_eval_failure
from app.evals.retrieval_metrics import hit_at_k, mrr, recall_at_k
from app.evalsets import load_jsonl
from app.rag.vectorstore import search_tenant_documents
from app.tenancy import set_current_tenant

HIT_RATE_THRESHOLD = 0.85
MRR_THRESHOLD = 0.7


def _gold_sources(row: dict) -> set[str]:
    return set(row.get("gold_sources") or row.get("expect_sources") or [])


def _search_query(row: dict) -> str:
    query = row["query"]
    expand = row.get("expand") or {}
    if expand:
        query = expand_query(
            query,
            None,
            last_course_query=expand.get("last_course_query"),
            last_course_name=expand.get("last_course_name"),
        )
    return query


def _course_code(row: dict) -> str | None:
    code = row.get("course_code")
    return str(code).strip() if code else None


def test_rag_golden() -> None:
    failures: list[str] = []
    for row in load_jsonl("rag.jsonl"):
        tenant = row.get("tenant_id") or "demo"
        set_current_tenant(tenant)
        query = _search_query(row)
        docs = retrieve_course_docs(
            query,
            tenant_id=tenant,
            course_code=_course_code(row),
        )
        sources = {str(doc.metadata.get("source")) for doc in docs}
        missing = [name for name in row.get("expect_sources") or [] if name not in sources]
        forbidden = [name for name in row.get("forbid_sources") or [] if name in sources]
        if missing:
            msg = f"{row['id']}: missing {missing}, got {sorted(sources)}"
            failures.append(msg)
            maybe_harvest_eval_failure(test_name="rag", case_id=row["id"], row=row, message=msg)
        if forbidden:
            msg = f"{row['id']}: leaked {forbidden}, got {sorted(sources)}"
            failures.append(msg)
            maybe_harvest_eval_failure(test_name="rag", case_id=row["id"], row=row, message=msg)
        if row.get("expect_empty") and docs:
            msg = f"{row['id']}: expected empty, got {sorted(sources)}"
            failures.append(msg)
            maybe_harvest_eval_failure(test_name="rag", case_id=row["id"], row=row, message=msg)
    assert not failures, "\n".join(failures)


def test_rag_retrieval_metrics() -> None:
    failures: list[str] = []
    hit_total = 0
    hit_ok = 0
    mrr_total = 0.0
    mrr_count = 0

    for row in load_jsonl("rag.jsonl"):
        tenant = row.get("tenant_id") or "demo"
        set_current_tenant(tenant)
        query = _search_query(row)
        eval_k = int(row.get("eval_k") or 8)
        result = search_tenant_documents(
            query,
            tenant_id=tenant,
            k=eval_k,
            course_code=_course_code(row),
        )
        min_score = get_settings().rag_min_score
        valid_hits = [hit for hit in result.hits if hit.passes_min_score(min_score)]
        ranked_sources = [str(hit.document.metadata.get("source") or "") for hit in valid_hits]
        gold = _gold_sources(row)

        if row.get("expect_empty"):
            if valid_hits:
                failures.append(f"{row['id']}: leak {sorted(set(ranked_sources))}")
            continue

        if not gold:
            continue

        hit_total += 1
        if hit_at_k(ranked_sources, gold, eval_k):
            hit_ok += 1
        else:
            msg = f"{row['id']}: miss at k={eval_k}, got {ranked_sources}"
            failures.append(msg)
            maybe_harvest_eval_failure(test_name="rag", case_id=row["id"], row=row, message=msg)

        mrr_total += mrr(ranked_sources, gold)
        mrr_count += 1

        recall = recall_at_k(ranked_sources, gold, eval_k)
        if recall < 1.0 and hit_at_k(ranked_sources, gold, eval_k):
            failures.append(f"{row['id']}: late_rank recall={recall:.2f} sources={ranked_sources}")

    hit_rate = hit_ok / hit_total if hit_total else 1.0
    avg_mrr = mrr_total / mrr_count if mrr_count else 1.0
    if hit_rate < HIT_RATE_THRESHOLD:
        failures.append(f"HitRate@{8}={hit_rate:.2f} < {HIT_RATE_THRESHOLD}")
    if avg_mrr < MRR_THRESHOLD:
        failures.append(f"MRR={avg_mrr:.2f} < {MRR_THRESHOLD}")

    assert not failures, "\n".join(failures)
