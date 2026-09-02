"""数据库模型、Session 与示例数据。"""

from app.db.courses import load_course_catalog, resolve_course_code
from app.db.models import Course, CourseModule, Order, OrderEvent, SessionArchive, UserProfile
from app.db.seed import seed_if_empty
from app.db.session import get_session, init_db, session_scope

__all__ = [
    "Course",
    "CourseModule",
    "Order",
    "OrderEvent",
    "SessionArchive",
    "UserProfile",
    "get_session",
    "init_db",
    "load_course_catalog",
    "resolve_course_code",
    "session_scope",
    "seed_if_empty",
]
