"""订单权限统一入口单测 · 覆盖决策矩阵每一分支。

测试数据依赖 conftest.py 的 seed_if_empty：
  - 订单 20251114001：tenant=demo, user=u10001（其他租户无此单）
  - 匿名用户：anonymous / "" / "anon"
"""

from __future__ import annotations

import pytest

from app.security.order_access import (
    AccessContext,
    OrderDenyReason,
    authorize,
    can_access_order,
    ensure_access,
    make_order_scope_filters,
)
from app.tenancy import set_current_tenant, set_current_user

# ---- fixtures --------------------------------------------------------------

BOUND_ORDER = "20251114001"   # 会话已绑定单号（seed 中 u10001 的单）
OWNER_UID = "u10001"
OTHER_UID = "alice"
ANON_UIDS = ("anonymous", "", "anon", None)


@pytest.fixture(autouse=True)
def _reset_tenancy():
    set_current_tenant("demo")
    set_current_user("anonymous")
    yield
    set_current_tenant("demo")
    set_current_user("anonymous")


# ---- §5.1 决策矩阵 · authorize() 全部 10 条 -------------------------------

class TestAuthorizeDecisionMatrix:
    # 用例 1：登录用户查自己的订单
    def test_owner_direct_query(self):
        decision = authorize(
            BOUND_ORDER, context=AccessContext.DIRECT_QUERY,
            tenant_id="demo", user_id=OWNER_UID,
        )
        assert decision.allowed is True
        assert decision.reason_code is None
        assert decision.order_id is not None
        assert decision.order is not None
        assert decision.order.order_no == BOUND_ORDER

    # 用例 2：登录用户查别人的订单 → NOT_ORDER_OWNER
    def test_not_owner_direct_query(self):
        decision = authorize(
            BOUND_ORDER, context=AccessContext.DIRECT_QUERY,
            tenant_id="demo", user_id=OTHER_UID,
        )
        assert decision.allowed is False
        assert decision.reason_code == OrderDenyReason.NOT_ORDER_OWNER

    # 用例 3：跨租户同 user_id → CROSS_TENANT
    def test_cross_tenant(self):
        decision = authorize(
            BOUND_ORDER, context=AccessContext.DIRECT_QUERY,
            tenant_id="acme", user_id=OWNER_UID,  # demo 的单，acme 来查
        )
        assert decision.allowed is False
        assert decision.reason_code == OrderDenyReason.CROSS_TENANT

    # 用例 4：匿名 + DIRECT_QUERY → 允许（本轮已提及）
    @pytest.mark.parametrize("anon_uid", ANON_UIDS)
    def test_anonymous_direct_query_allowed(self, anon_uid):
        decision = authorize(
            BOUND_ORDER, context=AccessContext.DIRECT_QUERY,
            tenant_id="demo", user_id=anon_uid,
        )
        assert decision.allowed is True, f"anon_uid={anon_uid!r} 应允许 DIRECT_QUERY"
        assert decision.order is not None

    # 用例 5：匿名 + MEMORY_RECALL → 永久拒绝（核心安全点）
    @pytest.mark.parametrize("anon_uid", ANON_UIDS)
    def test_anonymous_memory_recall_always_denied(self, anon_uid):
        decision = authorize(
            BOUND_ORDER, context=AccessContext.MEMORY_RECALL,
            tenant_id="demo", user_id=anon_uid,
        )
        assert decision.allowed is False
        assert decision.reason_code == OrderDenyReason.ANONYMOUS_UNBOUND

    # 用例 6：匿名 + SLOT_FOLLOWUP + session_bound 匹配 → 允许
    def test_anonymous_slot_followup_bound_matches(self):
        decision = authorize(
            BOUND_ORDER, context=AccessContext.SLOT_FOLLOWUP,
            tenant_id="demo", user_id="anonymous",
            session_bound_order_no=BOUND_ORDER,
        )
        assert decision.allowed is True

    # 用例 7：匿名 + SLOT_FOLLOWUP + bound 不匹配 → ANONYMOUS_UNBOUND（槽位偷换）
    def test_anonymous_slot_followup_bound_mismatch(self):
        decision = authorize(
            "20250000002", context=AccessContext.SLOT_FOLLOWUP,
            tenant_id="demo", user_id="anonymous",
            session_bound_order_no=BOUND_ORDER,  # 会话绑的是 20251114001，但请求了别的
        )
        assert decision.allowed is False
        assert decision.reason_code == OrderDenyReason.ANONYMOUS_UNBOUND

    # 用例 8：格式非法（< 8 位 / 含字母）→ MALFORMED_ORDER_NO
    @pytest.mark.parametrize("bad_no", ["1234567", "abc888888", "", "  ", "2025-1114"])
    def test_malformed_order_no(self, bad_no):
        decision = authorize(
            bad_no, context=AccessContext.DIRECT_QUERY,
            tenant_id="demo", user_id=OWNER_UID,
        )
        assert decision.allowed is False
        assert decision.reason_code == OrderDenyReason.MALFORMED_ORDER_NO

    # 用例 9：格式合法但不存在 → ORDER_NOT_FOUND
    def test_order_not_found(self):
        decision = authorize(
            "999999999999", context=AccessContext.DIRECT_QUERY,
            tenant_id="demo", user_id=OWNER_UID,
        )
        assert decision.allowed is False
        assert decision.reason_code == OrderDenyReason.ORDER_NOT_FOUND

    # 用例 10：登录用户记忆回填别人的单 → NOT_ORDER_OWNER（记忆写入保护）
    def test_persistent_user_memory_recall_other_order(self):
        decision = authorize(
            BOUND_ORDER, context=AccessContext.MEMORY_RECALL,
            tenant_id="demo", user_id=OTHER_UID,  # alice 尝试拿 u10001 的单进记忆
        )
        assert decision.allowed is False
        assert decision.reason_code == OrderDenyReason.NOT_ORDER_OWNER

    # 用例 10+：登录用户记忆回填自己的单 → 允许
    def test_persistent_user_memory_recall_self_allowed(self):
        decision = authorize(
            BOUND_ORDER, context=AccessContext.MEMORY_RECALL,
            tenant_id="demo", user_id=OWNER_UID,
        )
        assert decision.allowed is True


# ---- ensure_access() · 断言式语义 ------------------------------------------

class TestEnsureAccess:
    def test_passes_when_allowed(self):
        order = ensure_access(
            BOUND_ORDER, context=AccessContext.DIRECT_QUERY,
            tenant_id="demo", user_id=OWNER_UID,
        )
        assert order.order_no == BOUND_ORDER
        # 关联对象已预加载（selectinload）：不触发额外 SQL
        assert order.course is not None
        assert order.course.name
        assert len(order.events) > 0

    def test_raises_valueerror_on_deny(self):
        with pytest.raises(ValueError) as ei:
            ensure_access(
                BOUND_ORDER, context=AccessContext.DIRECT_QUERY,
                tenant_id="demo", user_id=OTHER_UID,  # alice 无权
            )
        # 对外文案统一：不暴露原因码，只说「未找到订单」
        assert "未找到订单" in str(ei.value)
        assert BOUND_ORDER in str(ei.value)

    def test_user_visible_msg_no_reason_leak(self):
        """拒绝消息不得包含 NOT_ORDER_OWNER / CROSS_TENANT 等内部 reason 字样。"""
        with pytest.raises(ValueError) as ei:
            ensure_access(
                BOUND_ORDER, context=AccessContext.DIRECT_QUERY,
                tenant_id="demo", user_id=OTHER_UID,
            )
        msg = str(ei.value)
        assert "NOT_ORDER_OWNER" not in msg
        assert "无权" not in msg
        assert "不是你的" not in msg


# ---- 兼容 API · can_access_order()（旧代码签名） --------------------------

class TestCanAccessOrderBackwardCompat:
    def test_true_for_owner(self):
        assert can_access_order(BOUND_ORDER, "demo", OWNER_UID) is True

    def test_false_for_other(self):
        assert can_access_order(BOUND_ORDER, "demo", OTHER_UID) is False

    def test_false_for_cross_tenant(self):
        assert can_access_order(BOUND_ORDER, "acme", OWNER_UID) is False

    def test_true_for_anonymous_direct(self):
        assert can_access_order(BOUND_ORDER, "demo", None) is True

    def test_false_for_empty(self):
        assert can_access_order("", "demo", OWNER_UID) is False


# ---- WHERE 过滤器生成器 · 与决策规则同源 ------------------------------------

class TestMakeOrderScopeFilters:
    def test_anonymous_skips_user_id_filter(self):
        flts = make_order_scope_filters(BOUND_ORDER, "demo", user_id=None)
        # SQL 条件数 = 2（tenant + order_no），不包含 user_id
        assert len(flts) == 2

    def test_persistent_user_adds_user_id_filter(self):
        flts = make_order_scope_filters(BOUND_ORDER, "demo", user_id=OWNER_UID)
        # SQL 条件数 = 3（tenant + order_no + user_id）
        assert len(flts) == 3


# ---- 上下文默认值 · 依赖 tenancy ContextVar --------------------------------

class TestTenancyContextDefaults:
    def test_uses_current_tenant_and_user(self):
        set_current_tenant("demo")
        set_current_user(OWNER_UID)
        # 不传 tenant_id / user_id，应从当前上下文拿
        decision = authorize(BOUND_ORDER, context=AccessContext.DIRECT_QUERY)
        assert decision.allowed is True

    def test_context_switch_affects_decision(self):
        set_current_tenant("demo")
        set_current_user(OTHER_UID)  # alice 无权
        decision = authorize(BOUND_ORDER, context=AccessContext.DIRECT_QUERY)
        assert decision.allowed is False
        assert decision.reason_code == OrderDenyReason.NOT_ORDER_OWNER


# ---- Context 枚举 · 字符串兼容 --------------------------------------------

class TestContextStringCompat:
    def test_string_context_is_cast(self):
        # 允许用字符串传 context，内部自动转枚举
        decision = authorize(
            BOUND_ORDER, context="direct_query",
            tenant_id="demo", user_id=OWNER_UID,
        )
        assert decision.allowed is True

    def test_invalid_string_context_raises(self):
        with pytest.raises((ValueError, KeyError)):
            authorize(
                BOUND_ORDER, context="nonexistent_ctx",
                tenant_id="demo", user_id=OWNER_UID,
            )
