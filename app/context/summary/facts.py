"""结构化滚动摘要：规则抽取、合并与渲染。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage

from app.context.slots import MISSING_COURSE_NAMES, SessionSlots
from app.context.window import message_text

DEFAULT_SESSION_SUMMARY_MAX_CHARS = 800

ORDER_NO_RE = re.compile(r"(?:订单|#|NO\.?)?\s*(\d{8,})", re.IGNORECASE)
_COURSE_HEADING_RE = re.compile(r"(?m)^##\s+(.+)$")
_COURSE_BOOK_RE = re.compile(r"《([^》]{2,30})》")
_COURSE_LABEL_RE = re.compile(r"课程名称[：:]\s*([^\n]{2,30})")
_COURSE_INLINE_RE = re.compile(r"课程\s*([A-Za-z0-9一-鿿]{2,20})")
_SKIP_COURSE_NAMES = frozenset({"模块", "名称", "咨询", "大纲", "未找到课程"})
_SECTION_RE = re.compile(r"^(订单|课程|意图|已确认|未决)[：:]\s*(.*)$")
_INTENT_LABELS = {
    "order": "查订单",
    "course_consult": "课程咨询",
    "mixed": "课程+订单",
    "handoff": "转人工",
}
_ORDER_CONSTRAINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"退款"), "用户要退款"),
    (re.compile(r"物流|快递|发货|到哪|到货"), "在追物流"),
    (re.compile(r"取消"), "要取消订单"),
    (re.compile(r"进度"), "在问进度"),
)
_COURSE_CONSTRAINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"不适合.*零基础|零基础.*不"), "已确认不适合零基础"),
    (re.compile(r"(?<!不)适合零基础|零基础可以|零基础友好"), "适合零基础"),
    (re.compile(r"太难|很难|偏难"), "用户觉得难"),
    (re.compile(r"多少钱|价格|学费"), "在问价格"),
    (re.compile(r"大纲|模块|学什么|讲什么"), "在问大纲"),
)
_PENDING_ORDER_NO = "待提供订单号"
_PENDING_COURSE = "待确认课程"
_PENDING_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"请提供订单号|需要订单号|完整订单编号"), _PENDING_ORDER_NO),
    (re.compile(r"请.*课程名|哪门课"), _PENDING_COURSE),
    (re.compile(r"转接人工|已为您转接"), "等待人工"),
)
_EXCLUSIVE_LABEL_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"已确认不适合零基础", "适合零基础"}),
    frozenset({"退款已完成", "退款审核中"}),
)
_ASSISTANT_CONCLUSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"已退款"), "退款已完成"),
    (re.compile(r"退款审核"), "退款审核中"),
    (re.compile(r"已发货"), "已发货"),
    (re.compile(r"已开通"), "课程已开通"),
    (re.compile(r"已取消"), "订单已取消"),
)


def _resolve_summary_max_chars(max_chars: int | None) -> int:
    if max_chars is None:
        try:
            from app.config import get_settings

            max_chars = get_settings().session_summary_max_chars
        except Exception:
            max_chars = DEFAULT_SESSION_SUMMARY_MAX_CHARS
    return max(80, int(max_chars))


def _clip_summary(text: str, max_chars: int | None = None) -> str:
    combined = (text or "").strip()
    limit = _resolve_summary_max_chars(max_chars)
    if len(combined) <= limit:
        return combined
    return combined[-limit:]


def _split_facts(raw: str) -> list[str]:
    return [p.strip() for p in re.split(r"[；;]+", raw or "") if p.strip()]


def _prepend_unique(existing: list[str], incoming: list[str], *, cap: int) -> list[str]:
    out = list(existing)
    for item in incoming:
        text = (item or "").strip()
        if not text:
            continue
        out = [x for x in out if x != text]
        out.append(text)
    return out[-cap:]


def _rivals_of(label: str) -> frozenset[str]:
    for group in _EXCLUSIVE_LABEL_GROUPS:
        if label in group:
            return group - {label}
    return frozenset()


def _apply_exclusive_sequence(labels: list[str], *, cap: int | None = None) -> list[str]:
    out: list[str] = []
    for item in labels:
        text = (item or "").strip()
        if not text:
            continue
        rivals = _rivals_of(text)
        out = [x for x in out if x != text and x not in rivals]
        out.append(text)
    if cap is not None:
        return out[-cap:]
    return out


@dataclass
class StructuredSummary:
    """滚动结构化摘要：记约束和结论，而不是对话原文。"""

    orders: list[str] = field(default_factory=list)
    courses: list[str] = field(default_factory=list)
    intent: str = ""
    confirmed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    def merge(self, other: StructuredSummary) -> StructuredSummary:
        return StructuredSummary(
            orders=_prepend_unique(self.orders, other.orders, cap=3),
            courses=_prepend_unique(self.courses, other.courses, cap=3),
            intent=(other.intent or self.intent).strip(),
            confirmed=_apply_exclusive_sequence(
                list(self.confirmed) + list(other.confirmed), cap=6
            ),
            pending=_prepend_unique(self.pending, other.pending, cap=4),
        )

    def resolve(self, *, slots: SessionSlots | None = None) -> StructuredSummary:
        pending = list(self.pending)
        has_order = bool(self.orders) or bool(
            (slots.last_order_no or "").strip() if slots else ""
        )
        has_course = bool(self.courses) or bool(
            (slots.last_course_name or "").strip() if slots else ""
        )
        if has_order:
            pending = [item for item in pending if item != _PENDING_ORDER_NO]
        if has_course:
            pending = [item for item in pending if item != _PENDING_COURSE]
        return StructuredSummary(
            orders=list(self.orders),
            courses=list(self.courses),
            intent=self.intent,
            confirmed=_apply_exclusive_sequence(self.confirmed, cap=6),
            pending=pending,
        )

    def has_content(self) -> bool:
        return bool(self.orders or self.courses or self.intent or self.confirmed or self.pending)

    def novel_against(self, existing: StructuredSummary) -> bool:
        known_orders = set(existing.orders)
        known_courses = set(existing.courses)
        known_confirmed = set(existing.confirmed)
        known_pending = set(existing.pending)
        return bool(
            any(item not in known_orders for item in self.orders)
            or any(item not in known_courses for item in self.courses)
            or (self.intent and self.intent != existing.intent)
            or any(item not in known_confirmed for item in self.confirmed)
            or any(item not in known_pending for item in self.pending)
        )

    def render(self, max_chars: int | None = None) -> str:
        limit = _resolve_summary_max_chars(max_chars) if max_chars is not None else None
        orders = list(self.orders)
        courses = list(self.courses)
        confirmed = list(self.confirmed)
        pending = list(self.pending)
        intent = self.intent

        def _lines() -> str:
            rows: list[str] = []
            if orders:
                rows.append("订单：" + "；".join(orders))
            if courses:
                rows.append("课程：" + "；".join(courses))
            if intent:
                rows.append(f"意图：{intent}")
            if confirmed:
                rows.append("已确认：" + "；".join(confirmed))
            if pending:
                rows.append("未决：" + "；".join(pending))
            return "\n".join(rows)

        text = _lines()
        if limit is None or len(text) <= limit:
            return text
        while len(text) > limit:
            if len(confirmed) > 1:
                confirmed = confirmed[1:]
            elif len(pending) > 1:
                pending = pending[1:]
            elif len(orders) > 1:
                orders = orders[1:]
            elif len(courses) > 1:
                courses = courses[1:]
            else:
                return text[-limit:]
            text = _lines()
        return text

    @classmethod
    def parse(cls, text: str | None) -> StructuredSummary:
        raw = (text or "").strip()
        if not raw:
            return cls()
        parsed = cls()
        leftover: list[str] = []
        saw_section = False
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            match = _SECTION_RE.match(line)
            if not match:
                leftover.append(line)
                continue
            saw_section = True
            key, value = match.group(1), match.group(2).strip()
            if key == "订单":
                parsed.orders = _split_facts(value)
            elif key == "课程":
                parsed.courses = _split_facts(value)
            elif key == "意图":
                parsed.intent = value
            elif key == "已确认":
                parsed.confirmed = _split_facts(value)
            elif key == "未决":
                parsed.pending = _split_facts(value)
        if leftover and not saw_section:
            parsed.confirmed = leftover
        elif leftover:
            parsed.confirmed = _prepend_unique(parsed.confirmed, leftover, cap=6)
        return parsed


def _extract_order_no(text: str) -> str | None:
    match = ORDER_NO_RE.search(text or "")
    return match.group(1) if match else None


def extract_course_names(text: str) -> list[str]:
    names: list[str] = []
    for pattern in (_COURSE_HEADING_RE, _COURSE_BOOK_RE, _COURSE_LABEL_RE, _COURSE_INLINE_RE):
        for match in pattern.finditer(text or ""):
            name = match.group(1).strip().strip("#").strip()
            if (
                name
                and name not in MISSING_COURSE_NAMES
                and name not in _SKIP_COURSE_NAMES
                and name not in names
            ):
                names.append(name)
    return names


def _match_labels(text: str, patterns: tuple[tuple[re.Pattern[str], str], ...]) -> list[str]:
    found: list[str] = []
    for pattern, label in patterns:
        if pattern.search(text) and label not in found:
            found.append(label)
    return found


def extract_summary_facts(
    overflow: list[BaseMessage] | None,
    *,
    slots: SessionSlots | None = None,
) -> StructuredSummary:
    acc = StructuredSummary()
    for message in overflow or []:
        text = message_text(message).strip()
        if text:
            acc = acc.merge(_facts_from_one(text, slots=slots))
    slot_intent = (slots.last_intent or "").strip() if slots else ""
    if slot_intent and slot_intent in _INTENT_LABELS:
        acc.intent = _INTENT_LABELS[slot_intent]
    elif not acc.intent:
        acc.intent = _infer_intent(acc)
    return acc.resolve(slots=slots)


def _facts_from_one(text: str, *, slots: SessionSlots | None) -> StructuredSummary:
    slot_order = (slots.last_order_no or "").strip() if slots else ""
    slot_course = (slots.last_course_name or "").strip() if slots else ""

    order_constraints = _match_labels(text, _ORDER_CONSTRAINTS)
    course_constraints = _match_labels(text, _COURSE_CONSTRAINTS)
    conclusions = _match_labels(text, _ASSISTANT_CONCLUSIONS)
    pending = _match_labels(text, _PENDING_PATTERNS)

    orders: list[str] = []
    order_no = _extract_order_no(text)
    if order_no and order_no != slot_order:
        suffix = "；".join(order_constraints[:2])
        orders.append(f"{order_no} {suffix}".strip() if suffix else order_no)

    courses: list[str] = []
    for name in extract_course_names(text):
        if slot_course and name == slot_course:
            continue
        suffix = "；".join(course_constraints[:2])
        courses.append(f"{name} {suffix}".strip() if suffix else name)

    confirmed = _apply_exclusive_sequence(
        course_constraints + order_constraints + conclusions
    )
    intent = ""
    if order_constraints and course_constraints:
        intent = _INTENT_LABELS["mixed"]
    elif order_constraints:
        intent = _INTENT_LABELS["order"]
    elif course_constraints:
        intent = _INTENT_LABELS["course_consult"]

    return StructuredSummary(
        orders=orders,
        courses=courses,
        intent=intent,
        confirmed=confirmed,
        pending=pending,
    )


def _infer_intent(summary: StructuredSummary) -> str:
    order_labels = {"用户要退款", "在追物流", "要取消订单", "在问进度"}
    course_labels = {"已确认不适合零基础", "适合零基础", "用户觉得难", "在问价格", "在问大纲"}
    has_order = bool(summary.orders) or bool(order_labels & set(summary.confirmed))
    has_course = bool(summary.courses) or bool(course_labels & set(summary.confirmed))
    if has_order and has_course:
        return _INTENT_LABELS["mixed"]
    if has_order:
        return _INTENT_LABELS["order"]
    if has_course:
        return _INTENT_LABELS["course_consult"]
    return ""


def finalize_summary(
    text: str,
    *,
    slots: SessionSlots | None = None,
    max_chars: int | None = None,
) -> str:
    return finalize_structured(
        StructuredSummary.parse(text), slots=slots, max_chars=max_chars
    )


def finalize_structured(
    summary: StructuredSummary,
    *,
    slots: SessionSlots | None = None,
    max_chars: int | None = None,
) -> str:
    resolved = summary.resolve(slots=slots)
    if not resolved.has_content():
        return ""
    return resolved.render(max_chars)


def fold_session_summary(
    existing: str | None,
    overflow: list[BaseMessage] | None,
    *,
    max_chars: int | None = None,
    slots: SessionSlots | None = None,
) -> str:
    current = StructuredSummary.parse(existing)
    delta = extract_summary_facts(overflow, slots=slots)
    return finalize_structured(current.merge(delta), slots=slots, max_chars=max_chars)
