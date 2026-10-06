from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import LOCKOUT_LIMIT, LOCKOUT_PATH, LOCKOUT_SECONDS, ensure_data_dirs


class LockoutStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or LOCKOUT_PATH

    def _load(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict) -> None:
        ensure_data_dirs()
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def remaining_seconds(self, name: str) -> int:
        record = self._load().get(name) or {}
        until = record.get("locked_until")
        if not until:
            return 0
        locked_until = datetime.fromisoformat(until)
        delta = (locked_until - datetime.now(timezone.utc)).total_seconds()
        return max(0, int(delta))

    def register_failure(self, name: str) -> int:
        data = self._load()
        record = data.get(name) or {"failures": 0, "locked_until": None}
        if self.remaining_seconds(name) > 0:
            return self.remaining_seconds(name)
        record["failures"] = int(record.get("failures", 0)) + 1
        if record["failures"] >= LOCKOUT_LIMIT:
            until = datetime.now(timezone.utc) + timedelta(seconds=LOCKOUT_SECONDS)
            record["locked_until"] = until.isoformat()
            record["failures"] = 0
        data[name] = record
        self._save(data)
        return self.remaining_seconds(name)

    def register_success(self, name: str) -> None:
        data = self._load()
        data[name] = {"failures": 0, "locked_until": None}
        self._save(data)
