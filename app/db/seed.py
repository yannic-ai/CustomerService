"""写入演示用课程与订单数据（多租户）。"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Course, CourseModule, Order, OrderEvent


def seed_if_empty(session: Session) -> None:
    """库为空时写入 demo / acme 示例数据；订单 20251114001 仅属于 demo。"""
    if session.scalar(select(Course.id).limit(1)):
        return

    python = Course(
        tenant_id="demo",
        code="python-intro",
        name="Python入门课",
        summary="面向零基础学员的 Python 编程入门课程，覆盖语法、数据结构、函数、文件与面向对象，并以实战项目收尾。",
        level="入门",
        duration_hours=24,
        oss_key="courses/python-intro.md",
        modules=[
            CourseModule(
                seq=1,
                name="环境搭建与基础语法",
                topics="安装解释器,变量与类型,输入输出,运算符,注释规范",
                duration_hours=4,
                outcome="能独立运行第一段 Python 程序",
            ),
            CourseModule(
                seq=2,
                name="流程控制",
                topics="条件判断,for/while 循环,break/continue,简单算法题",
                duration_hours=4,
                outcome="能编写分支与循环逻辑",
            ),
            CourseModule(
                seq=3,
                name="数据结构",
                topics="列表,元组,字典,集合,切片与推导式",
                duration_hours=5,
                outcome="能选择合适的数据结构解决问题",
            ),
            CourseModule(
                seq=4,
                name="函数与模块",
                topics="函数定义,参数与返回值,作用域,模块导入,虚拟环境",
                duration_hours=4,
                outcome="能拆分可复用函数并组织项目结构",
            ),
            CourseModule(
                seq=5,
                name="文件、异常与面向对象",
                topics="文件读写,异常处理,类与对象,封装继承",
                duration_hours=4,
                outcome="能读写文件并用类建模简单业务",
            ),
            CourseModule(
                seq=6,
                name="综合实战：命令行待办工具",
                topics="需求拆解,数据持久化,命令解析,测试与发布",
                duration_hours=3,
                outcome="完成一个可运行的命令行小项目",
            ),
        ],
    )
    analysis = Course(
        tenant_id="demo",
        code="data-analysis",
        name="数据分析实战",
        summary="基于 Python 的数据分析入门，学习 Pandas、可视化与一份业务分析报告。",
        level="进阶",
        duration_hours=20,
        oss_key="courses/data-analysis.md",
        modules=[
            CourseModule(seq=1, name="NumPy 基础", topics="ndarray,广播,统计运算", duration_hours=4, outcome="掌握数组计算"),
            CourseModule(seq=2, name="Pandas 数据处理", topics="DataFrame,清洗,分组聚合", duration_hours=8, outcome="能完成常规数据清洗"),
            CourseModule(seq=3, name="可视化与报告", topics="matplotlib,seaborn,业务洞察", duration_hours=8, outcome="输出分析报告"),
        ],
    )
    ml = Course(
        tenant_id="demo",
        code="ml-basics",
        name="机器学习基础",
        summary="监督学习入门，覆盖特征工程、经典模型和模型评估。",
        level="进阶",
        duration_hours=28,
        oss_key="courses/ml-basics.md",
        modules=[
            CourseModule(seq=1, name="机器学习导论", topics="任务类型,训练集,过拟合", duration_hours=4, outcome="建立 ML 基本概念"),
            CourseModule(seq=2, name="特征工程", topics="缺失值,编码,标准化", duration_hours=6, outcome="能准备训练数据"),
            CourseModule(seq=3, name="经典模型", topics="线性回归,逻辑回归,决策树,随机森林", duration_hours=12, outcome="能训练并比较模型"),
            CourseModule(seq=4, name="评估与调参", topics="交叉验证,指标选择,网格搜索", duration_hours=6, outcome="能评估并优化模型"),
        ],
    )
    acme_course = Course(
        tenant_id="acme",
        code="acme-onboarding",
        name="Acme 入职培训",
        summary="Acme 租户专属入职课程，用于验证跨租户隔离（不含 demo 订单）。",
        level="入门",
        duration_hours=4,
        oss_key="courses/acme-onboarding.md",
        modules=[
            CourseModule(
                seq=1,
                name="公司介绍",
                topics="组织架构,产品线,协作规范",
                duration_hours=2,
                outcome="了解 Acme 基本情况",
            ),
            CourseModule(
                seq=2,
                name="工具与流程",
                topics="工单系统,知识库,安全须知",
                duration_hours=2,
                outcome="能按流程处理日常请求",
            ),
        ],
    )
    session.add_all([python, analysis, ml, acme_course])
    session.flush()

    order = Order(
        tenant_id="demo",
        order_no="20251114001",
        user_id="u10001",
        course_id=python.id,
        order_type="purchase",
        status="refund_reviewing",
        amount=199.00,
        events=[
            OrderEvent(
                seq=1,
                name="下单支付",
                status="completed",
                occurred_at=datetime(2025, 11, 14, 10, 21),
                note="微信支付成功",
            ),
            OrderEvent(
                seq=2,
                name="开通课程",
                status="completed",
                occurred_at=datetime(2025, 11, 14, 10, 22),
                note="已开通 Python入门课 学习权限",
            ),
            OrderEvent(
                seq=3,
                name="申请退款",
                status="completed",
                occurred_at=datetime(2025, 11, 20, 19, 8),
                note="用户提交退款申请，原因：时间冲突",
            ),
            OrderEvent(
                seq=4,
                name="退款审核",
                status="in_progress",
                occurred_at=datetime(2025, 11, 21, 9, 30),
                note="财务审核中，预计 1-3 个工作日",
            ),
            OrderEvent(
                seq=5,
                name="退款到账",
                status="pending",
                occurred_at=None,
                note="审核通过后原路退回",
            ),
        ],
    )
    session.add(order)
    session.commit()
