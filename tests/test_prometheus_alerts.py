"""校验 deploy/prometheus 告警规则与 README 建议对齐（不跑真实 Prometheus）。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROM_DIR = ROOT / "deploy" / "prometheus"
RULES = PROM_DIR / "rules" / "cs.yml"
PROM_YML = PROM_DIR / "prometheus.yml"


def test_rules_file_exists() -> None:
    assert RULES.is_file()
    assert PROM_YML.is_file()
    assert (PROM_DIR / "alertmanager.yml").is_file()


def test_prometheus_yml_loads_cs_rules() -> None:
    text = PROM_YML.read_text(encoding="utf-8")
    assert "rules/cs.yml" in text
    assert "metrics_path: /metrics" in text


def test_recording_and_alert_names() -> None:
    text = RULES.read_text(encoding="utf-8")
    assert "record: cs:order_query:success_ratio" in text
    for name in (
        "CsOrderQuerySuccessRatioLow",
        "CsOrderQueryTimeoutSpike",
        "CsAgentRunawaySustained",
        "CsReactLimitHitSustained",
        "CsLlmCallsPerTurnP95High",
    ):
        assert f"alert: {name}" in text


def test_exprs_reference_expected_metrics() -> None:
    text = RULES.read_text(encoding="utf-8")
    assert "cs_tool_calls_total" in text
    assert "cs_agent_runaway_total" in text
    assert "cs_react_limit_hit_total" in text
    assert "cs_llm_calls_per_turn_bucket" in text


def test_success_ratio_denominator_matches_readme() -> None:
    text = RULES.read_text(encoding="utf-8")
    # README：ok/(ok+not_found+deny+timeout+error)；busy 不进分母
    assert 'outcome=~"ok|not_found|deny|timeout|error"' in text
    assert "not_found|deny" in text
    assert 'outcome=~"ok|not_found|deny|timeout|error|busy"' not in text
