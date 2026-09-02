"""兼容旧入口：`uvicorn app.api:app` 与 `app.app:app` 指向同一应用。"""

from app.app import app, chat_api, run

__all__ = ["app", "chat_api", "run"]
