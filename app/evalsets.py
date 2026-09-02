"""离线评测集加载。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import ROOT_DIR

EVAL_DIR = ROOT_DIR / "evals"


def load_jsonl(name: str) -> list[dict[str, Any]]:
    """读取 ``evals/<name>``，跳过空行与 ``#`` 注释行。"""
    path = EVAL_DIR / name
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        rows.append(json.loads(text))
    return rows
