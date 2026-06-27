from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.db.repositories import get_recent_agent_runs, upsert_agent_run
from src.settings import SETTINGS


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def log_agent_run(event: dict[str, Any], log_path: str | Path | None = None) -> None:
    path = Path(log_path) if log_path else SETTINGS.log_path
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **_json_safe(event),
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    try:
        upsert_agent_run(payload)
    except Exception:
        pass


def load_recent_agent_runs(
    limit: int = 100,
    log_path: str | Path | None = None,
    prefer_db: bool = True,
) -> list[dict[str, Any]]:
    if prefer_db and log_path is None:
        try:
            rows = get_recent_agent_runs(limit=limit)
            if rows:
                return rows
        except Exception:
            pass

    path = Path(log_path) if log_path else SETTINGS.log_path
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rows.sort(key=lambda row: row.get("logged_at", ""), reverse=True)
    return rows[:limit]
