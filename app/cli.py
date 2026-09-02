"""命令行入口：serve / ingest / seed / chat / graph / eval。"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.config import get_settings
from app.concurrency import achat
from app.db.seed import seed_if_empty
from app.db.session import init_db, session_scope
from app.graph import export_graph_visual, get_graph
from app.rag.vectorstore import IndexUnavailableError, ingest_indexes, require_index_for_serve
from app.tenancy import normalize_tenant_id, set_current_tenant


def _ensure_db() -> None:
    """建表并在空库时灌入示例数据；不构建向量索引。"""
    init_db()
    with session_scope() as session:
        seed_if_empty(session)


def main(argv: list[str] | None = None) -> None:
    """解析子命令并执行对应动作，默认启动 HTTP 服务。"""
    settings = get_settings()
    from app.observability import configure_logging

    configure_logging(settings.log_level, settings.log_format)

    parser = argparse.ArgumentParser(description="智能客服系统")
    parser.add_argument(
        "--tenant",
        default=settings.default_tenant,
        help=f"租户 ID（默认 {settings.default_tenant}）",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="启动 HTTP 服务")
    ingest_parser = sub.add_parser("ingest", help="增量更新 FAISS 索引（先写新 chunk，再软删旧切片）")
    ingest_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="全量重建，忽略增量与软删除",
    )
    sub.add_parser("seed", help="初始化数据库与示例数据")
    sub.add_parser("graph", help="导出 LangGraph 可视化流程图")
    eval_parser = sub.add_parser("eval", help="跑路由/检索黄金集（keyword）；加 --llm 跑 nightly")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")
    run_parser = eval_sub.add_parser("run", help="跑黄金集（默认）")
    run_parser.add_argument(
        "--llm",
        action="store_true",
        help="额外跑需要 DEEPSEEK_API_KEY 的 LLM 路由评测",
    )
    eval_sub.add_parser("harvest", help="跑检索/忠实度评测并将失败样本导出到 evals/pending")
    merge_parser = eval_sub.add_parser("merge-pending", help="预览或合并 pending 到正式 evals")
    merge_parser.add_argument(
        "--apply",
        action="store_true",
        help="将 pending 追加到 evals/*.jsonl 并清空 pending",
    )
    eval_parser.add_argument(
        "--llm",
        action="store_true",
        help="（兼容旧用法）同 eval run --llm",
    )
    chat_parser = sub.add_parser("chat", help="命令行对话")
    chat_parser.add_argument("message", nargs="+", help="用户问题")
    chat_parser.add_argument(
        "--session",
        default=None,
        help="会话 ID；相同 session 可跨多次调用续聊",
    )
    chat_parser.add_argument(
        "--user",
        default="anonymous",
        help="用户 ID；登录用户只能查自己的订单，非 anonymous 时写入长期记忆",
    )

    args = parser.parse_args(argv)
    command = args.command or "serve"
    tenant_id = normalize_tenant_id(args.tenant)
    set_current_tenant(tenant_id)

    if command == "seed":
        _ensure_db()
        print("数据库已就绪。向量索引请执行 python main.py ingest")
        return
    if command == "ingest":
        stores = ingest_indexes(persist=True, rebuild=bool(getattr(args, "rebuild", False)))
        total = sum(store.index.ntotal for store in stores.values())
        print(f"已写入 FAISS 索引，向量数：{total}，租户：{', '.join(sorted(stores))}")
        return
    if command == "graph":
        path = export_graph_visual(get_graph())
        print(path.read_text(encoding="utf-8"))
        print(f"已写入 {path}")
        return
    if command == "eval":
        import os

        import pytest

        subcmd = getattr(args, "eval_command", None) or "run"
        if subcmd == "harvest":
            os.environ["EVAL_HARVEST"] = "1"
            raise SystemExit(
                pytest.main(
                    [
                        "-q",
                        "tests/test_eval_rag.py",
                        "tests/test_eval_faithfulness.py",
                    ]
                )
            )
        if subcmd == "merge-pending":
            from app.evals.harvest import merge_pending

            for line in merge_pending(apply=bool(getattr(args, "apply", False))):
                print(line)
            return

        use_llm = bool(getattr(args, "llm", False))
        pytest_args = [
            "-q",
            "tests/test_eval_routing.py",
            "tests/test_eval_rag.py",
            "tests/test_eval_faithfulness.py",
            "tests/test_eval_order.py",
            "tests/test_eval_react.py",
            "tests/test_eval_security.py",
        ]
        if use_llm:
            os.environ["RUN_LLM_EVAL"] = "1"
            pytest_args = ["-q", "-m", "llm"]
        raise SystemExit(pytest.main(pytest_args))
    if command == "chat":
        _ensure_db()
        try:
            require_index_for_serve()
        except IndexUnavailableError as exc:
            logging.getLogger("cs.cli").error("%s", exc)
            raise SystemExit(1) from exc
        message = " ".join(args.message)
        response = asyncio.run(
            achat(
                message,
                tenant_id=tenant_id,
                session_id=args.session,
                user_id=args.user,
            )
        )
        if response.session_id:
            print(f"[session_id={response.session_id}]")
        print(response.answer)
        return

    import uvicorn

    from app.session_cache import RedisUnavailableError, require_redis_for_serve

    try:
        require_redis_for_serve()
        require_index_for_serve()
    except (RedisUnavailableError, IndexUnavailableError) as exc:
        logging.getLogger("cs.serve").error("%s", exc)
        raise SystemExit(1) from exc
    settings = get_settings()
    workers = max(1, settings.uvicorn_workers)
    from app.observability.metrics import configure_metrics_multiprocess_from_settings

    configure_metrics_multiprocess_from_settings(settings)
    uvicorn.run(
        "app.app:app",
        host=settings.app_host,
        port=settings.app_port,
        workers=workers,
        reload=False,
    )


if __name__ == "__main__":
    main()
