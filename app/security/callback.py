"""LangChain BaseCallbackHandler：审计 LLM / Tool 调用，适配 v1.0 callbacks 协议。"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from app.observability import get_request_id, log_event

logger = logging.getLogger("cs.security")


class SecurityCallbackHandler(BaseCallbackHandler):
    """自定义安全回调，可挂到 create_agent / graph.stream 的 config.callbacks。"""

    raise_error = False

    def __init__(self, user_id: str = "anonymous", tenant_id: str = "demo") -> None:
        """绑定用户与租户 ID，用于审计日志关联。"""
        super().__init__()
        self.user_id = user_id
        self.tenant_id = tenant_id
        self.events: list[dict[str, Any]] = []

    def _record(self, event: str, **payload: Any) -> None:
        """记录一条安全审计事件。"""
        item = {
            "event": event,
            "request_id": get_request_id(),
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            **payload,
        }
        self.events.append(item)
        log_event(logger, "security_event", **item)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """LLM 开始调用时记录模型名与输入预览。"""
        preview = ""
        if messages and messages[0]:
            preview = str(messages[0][-1].content)[:200]
        self._record(
            "llm_start",
            run_id=str(run_id),
            model=serialized.get("name") or serialized.get("id"),
            preview=preview,
        )

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """LLM 结束时记录 token 用量。"""
        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage") or {}
        self._record("llm_end", run_id=str(run_id), token_usage=token_usage)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """记录 LLM 调用失败。"""
        self._record("llm_error", run_id=str(run_id), error=str(error))

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """工具开始执行时记录名称与入参预览。"""
        self._record(
            "tool_start",
            run_id=str(run_id),
            tool=serialized.get("name"),
            input_preview=str(input_str)[:300],
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """工具执行成功时记录输出预览。"""
        self._record("tool_end", run_id=str(run_id), output_preview=str(output)[:300])

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """记录工具执行失败。"""
        self._record("tool_error", run_id=str(run_id), error=str(error))

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        """记录链路执行失败。"""
        self._record("chain_error", run_id=str(run_id), error=str(error))
