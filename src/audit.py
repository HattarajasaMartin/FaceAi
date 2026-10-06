from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import VERIFY_LOG_PATH, ensure_data_dirs
from .domain import VerifyResult


def append_verify_log(result: VerifyResult, extra: dict[str, Any] | None = None) -> None:
    ensure_data_dirs()
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "accepted": result.accepted,
        "code": result.code,
        "claimed": result.claimed,
        "best_name": result.best_name,
        "claimed_score": round(result.claimed_score, 4),
        "best_score": round(result.best_score, 4),
        "message": result.message,
        "scores": {item.name: round(item.score, 4) for item in result.scores},
    }
    if extra:
        payload.update(extra)
    with VERIFY_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def recent_logs(limit: int = 5, path: Path | None = None) -> list[dict[str, Any]]:
    file_path = path or VERIFY_LOG_PATH
    if not file_path.is_file():
        return []
    lines = file_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(rows))
