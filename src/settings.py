from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AppSettings:
    vectorstore_dir: str = _env_str("VECTORSTORE_DIR", "data/vectorstore")
    ollama_base_url: str = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    chat_model: str = _env_str("CHAT_MODEL", "llama3.1")
    embed_model: str = _env_str("EMBED_MODEL", "nomic-embed-text")
    database_url: str = _env_str("DATABASE_URL", "sqlite:///data/app.db")
    log_dir: str = _env_str("LOG_DIR", "data/logs")
    request_timeout_sec: int = _env_int("REQUEST_TIMEOUT_SEC", 30)
    rag_top_k: int = _env_int("RAG_TOP_K", 6)
    rag_fetch_k: int = _env_int("RAG_FETCH_K", 30)
    min_evidence_score: int = _env_int("MIN_EVIDENCE_SCORE", 40)
    max_logged_snippet_chars: int = _env_int("MAX_LOGGED_SNIPPET_CHARS", 500)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir) / "agent_runs.jsonl"

    @property
    def sqlite_path(self) -> Path:
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            return Path(self.database_url[len(prefix):])
        return Path("data/app.db")


SETTINGS = AppSettings()
