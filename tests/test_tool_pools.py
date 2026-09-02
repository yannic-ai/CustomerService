import threading

from app.tenancy import get_current_tenant, set_current_tenant
from app.executors import run_order, shutdown_tool_pools


def setup_function() -> None:
    shutdown_tool_pools()


def teardown_function() -> None:
    shutdown_tool_pools()


def test_run_order_uses_named_thread() -> None:
    names: list[str] = []

    def job() -> str:
        names.append(threading.current_thread().name)
        return "ok"

    assert run_order(job) == "ok"
    assert names and names[0].startswith("cs-order")



def test_run_order_copies_tenant_context() -> None:
    seen: list[str] = []
    set_current_tenant("acme")

    def job() -> None:
        seen.append(get_current_tenant())

    run_order(job)
    assert seen == ["acme"]
    set_current_tenant("demo")


def test_nested_run_order_does_not_deadlock() -> None:
    def inner() -> str:
        return threading.current_thread().name

    def outer() -> str:
        return run_order(inner)

    name = run_order(outer)
    assert name.startswith("cs-order")


def test_workers_zero_runs_inline(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "app.executors.get_settings",
        lambda: SimpleNamespace(
            order_tool_workers=0,
            order_tool_timeout_seconds=10,
            tool_pool_queue_factor=2,
            tool_pool_queue_wait_seconds=5,
        ),
    )
    shutdown_tool_pools()
    caller = threading.current_thread().name

    def job() -> str:
        return threading.current_thread().name

    assert run_order(job) == caller
