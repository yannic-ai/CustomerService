"""订单、课程相关 ORM 模型。"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类。"""

    pass


class Course(Base):
    """课程主数据（按租户隔离）。"""

    __tablename__ = "courses"
    __table_args__ = (UniqueConstraint("tenant_id", "code", name="uq_courses_tenant_code"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")
    code: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    level: Mapped[str] = mapped_column(String(32), default="入门")
    duration_hours: Mapped[float] = mapped_column(Float, default=0)
    oss_key: Mapped[str | None] = mapped_column(String(255), nullable=True)

    modules: Mapped[list["CourseModule"]] = relationship(back_populates="course")
    orders: Mapped[list["Order"]] = relationship(back_populates="course")


class CourseModule(Base):
    """课程模块，对应大纲中的一章。"""

    __tablename__ = "course_modules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=1)
    name: Mapped[str] = mapped_column(String(128))
    topics: Mapped[str] = mapped_column(Text, default="")
    duration_hours: Mapped[float] = mapped_column(Float, default=0)
    outcome: Mapped[str] = mapped_column(Text, default="")

    course: Mapped[Course] = relationship(back_populates="modules")


class Order(Base):
    """用户订单，关联课程与进度事件（按租户隔离）。"""

    __tablename__ = "orders"
    __table_args__ = (UniqueConstraint("tenant_id", "order_no", name="uq_orders_tenant_order_no"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")
    order_no: Mapped[str] = mapped_column(String(32), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"))
    order_type: Mapped[str] = mapped_column(String(32), default="purchase")
    status: Mapped[str] = mapped_column(String(32), index=True)
    amount: Mapped[float] = mapped_column(Float, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    course: Mapped[Course] = relationship(back_populates="orders")
    events: Mapped[list["OrderEvent"]] = relationship(
        back_populates="order", order_by="OrderEvent.seq"
    )


class OrderEvent(Base):
    """订单进度节点，如支付、开通、退款审核。"""

    __tablename__ = "order_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer, default=1)
    name: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    order: Mapped[Order] = relationship(back_populates="events")


class UserProfile(Base):
    """用户长期档案冷存储（Redis miss 回源）。"""

    __tablename__ = "user_profiles"
    __table_args__ = (UniqueConstraint("tenant_id", "user_id", name="uq_user_profiles_tenant_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    profile_json: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class SessionArchive(Base):
    """会话公共状态冷存储（messages / 槽位 / 摘要；Redis miss 回源）。"""

    __tablename__ = "session_archives"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "user_id",
            "session_id",
            name="uq_session_archives_tenant_user_session",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    thread_id: Mapped[str] = mapped_column(String(255), default="", index=True)
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
