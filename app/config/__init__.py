"""应用配置：环境变量 / ``.env`` 按主题拆 mixin，对外仍 ``from app.config import Settings``。"""

from app.config.capacity import CapacitySettings
from app.config.llm import LLMSettings
from app.config.memory import MemorySettings
from app.config.paths import DATA_DIR, KNOWLEDGE_DIR, ROOT_DIR
from app.config.rag import EmbeddingBackend, RAGSettings
from app.config.resilience import ResilienceSettings
from app.config.runtime import LogFormat, LogLevel, RuntimeSettings
from app.config.settings import Settings, get_settings
from app.config.telemetry import TelemetrySettings

__all__ = [
    "CapacitySettings",
    "DATA_DIR",
    "EmbeddingBackend",
    "KNOWLEDGE_DIR",
    "LLMSettings",
    "LogFormat",
    "LogLevel",
    "MemorySettings",
    "RAGSettings",
    "ResilienceSettings",
    "ROOT_DIR",
    "RuntimeSettings",
    "Settings",
    "TelemetrySettings",
    "get_settings",
]
