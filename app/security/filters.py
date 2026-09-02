"""安全过滤：注入检测、PII 脱敏、长度限制。"""

from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
    r"you\s+are\s+now",
    r"developer\s+mode",
    r"jailbreak",
    r"system\s+prompt",
    r"忽略(以上|之前|先前|前面)(的)?(所有)?(指令|提示|规则)",
    r"无视(所有)?(指令|规则|限制)",
    r"越狱",
    r"扮演.*不受限制",
]

PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("bank_card", re.compile(r"(?<!\d)\d{13,19}(?!\d)")),
    ("phone", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]

MAX_INPUT_CHARS = 4000


@dataclass
class SecurityVerdict:
    """一次安全检查的结论。"""

    allowed: bool
    reason: str = ""
    sanitized_text: str = ""
    findings: list[str] | None = None


def _contains_injection(text: str) -> str | None:
    """检测提示注入；命中则返回对应正则，否则返回 None。"""
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return pattern
    return None


def mask_pii(text: str) -> tuple[str, list[str]]:
    """脱敏身份证、银行卡、手机号、邮箱，返回 (替换后文本, 命中类型列表)。

    匹配顺序：长模式优先（id_card→bank_card→phone），避免短模式误吞长串数字
    （例如身份证前 11 位被 phone 先截断，导致 id_card 后续正则无法识别）。
    """
    findings: list[str] = []
    masked = text
    for label, pattern in PII_PATTERNS:
        if pattern.search(masked):
            findings.append(label)
            masked = pattern.sub(f"[{label.upper()}_MASKED]", masked)
    return masked, findings


def inspect_input(text: str) -> SecurityVerdict:
    """检查用户输入：空值/超长拦截、注入拦截，通过后做 PII 脱敏。"""
    raw = (text or "").strip()
    if not raw:
        return SecurityVerdict(False, "输入为空", "")
    if len(raw) > MAX_INPUT_CHARS:
        return SecurityVerdict(False, f"输入超过 {MAX_INPUT_CHARS} 字符限制", raw[:MAX_INPUT_CHARS])

    hit = _contains_injection(raw)
    if hit:
        return SecurityVerdict(False, "检测到提示注入风险", raw, findings=["prompt_injection"])

    sanitized, findings = mask_pii(raw)
    return SecurityVerdict(True, "", sanitized, findings)


def inspect_output(text: str) -> str:
    """检查模型输出，避免把 PII 原样返回给用户。"""
    masked, _ = mask_pii(text or "")
    return masked
