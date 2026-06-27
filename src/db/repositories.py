from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.db.database import get_connection, init_db


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _row_to_dict(row) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def _text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _safe_metadata(doc) -> dict[str, Any]:
    metadata = getattr(doc, "metadata", {}) or {}
    safe = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, tuple, set)):
            safe[key] = [str(v) for v in value]
        else:
            safe[key] = str(value)
    return safe


def upsert_document_chunks(chunks: list[Any], db_path: str | Path | None = None) -> dict[str, int]:
    if not chunks:
        return {"documents": 0, "chunks": 0}

    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    doc_ids = set()
    chunk_count = 0

    with get_connection(db_path) as conn:
        for chunk in chunks:
            metadata = _safe_metadata(chunk)
            source_url = metadata.get("source") or metadata.get("url") or "Unknown source"
            doc_id = metadata.get("doc_id") or _text_hash(source_url)
            doc_ids.add(doc_id)

            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, source_url, source_type, title, domain, primary_ticker,
                    primary_event, ingested_at, metadata_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_url=excluded.source_url,
                    source_type=excluded.source_type,
                    title=excluded.title,
                    domain=excluded.domain,
                    primary_ticker=excluded.primary_ticker,
                    primary_event=excluded.primary_event,
                    ingested_at=excluded.ingested_at,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    doc_id,
                    source_url,
                    metadata.get("source_type", "url"),
                    metadata.get("title", "Untitled"),
                    metadata.get("domain", "unknown"),
                    metadata.get("primary_ticker", "Unknown"),
                    metadata.get("primary_event", "General Update"),
                    metadata.get("ingested_at", now),
                    _json_dumps(metadata),
                    now,
                ),
            )

            chunk_id = metadata.get("chunk_id") or _text_hash(f"{source_url}:{getattr(chunk, 'page_content', '')}")
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, source_url, chunk_index, text_hash, snippet, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    doc_id=excluded.doc_id,
                    source_url=excluded.source_url,
                    chunk_index=excluded.chunk_index,
                    text_hash=excluded.text_hash,
                    snippet=excluded.snippet,
                    metadata_json=excluded.metadata_json
                """,
                (
                    chunk_id,
                    doc_id,
                    source_url,
                    int(metadata.get("chunk_index", 0) or 0),
                    _text_hash(getattr(chunk, "page_content", "") or ""),
                    " ".join((getattr(chunk, "page_content", "") or "").split())[:800],
                    _json_dumps(metadata),
                ),
            )
            chunk_count += 1

    return {"documents": len(doc_ids), "chunks": chunk_count}


def upsert_agent_run(event: dict[str, Any], db_path: str | Path | None = None) -> None:
    init_db(db_path)

    request = event.get("request", {}) or {}
    evidence = event.get("evidence_metrics", {}) or {}
    critic = event.get("critic", {}) or {}
    output_validation = event.get("output_validation", {}) or {}
    trajectory = event.get("trajectory", []) or []
    sources = event.get("sources", []) or []

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO agent_runs (
                run_id, event_type, logged_at, analysis_mode, ticker, company_name,
                latency_sec, critic_status, confidence, evidence_score, source_count,
                chunk_count, request_json, evidence_json, critic_json, output_validation_json, trajectory_json,
                sources_json, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                event_type=excluded.event_type,
                logged_at=excluded.logged_at,
                analysis_mode=excluded.analysis_mode,
                ticker=excluded.ticker,
                company_name=excluded.company_name,
                latency_sec=excluded.latency_sec,
                critic_status=excluded.critic_status,
                confidence=excluded.confidence,
                evidence_score=excluded.evidence_score,
                source_count=excluded.source_count,
                chunk_count=excluded.chunk_count,
                request_json=excluded.request_json,
                evidence_json=excluded.evidence_json,
                critic_json=excluded.critic_json,
                output_validation_json=excluded.output_validation_json,
                trajectory_json=excluded.trajectory_json,
                sources_json=excluded.sources_json,
                raw_json=excluded.raw_json
            """,
            (
                event.get("run_id", ""),
                event.get("event_type", ""),
                event.get("logged_at", ""),
                request.get("analysis_mode", ""),
                request.get("ticker", ""),
                request.get("company_name", ""),
                event.get("latency_sec"),
                critic.get("status", ""),
                critic.get("confidence", ""),
                int(evidence.get("evidence_score", 0) or 0),
                int(evidence.get("unique_source_count", 0) or 0),
                int(evidence.get("retrieved_chunk_count", 0) or 0),
                _json_dumps(request),
                _json_dumps(evidence),
                _json_dumps(critic),
                _json_dumps(output_validation),
                _json_dumps(trajectory),
                _json_dumps(sources),
                _json_dumps(event),
            ),
        )


def get_recent_agent_runs(limit: int = 100, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM agent_runs
            ORDER BY logged_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    results = []
    for row in rows:
        item = _row_to_dict(row)
        item["request"] = json.loads(item.pop("request_json", "{}") or "{}")
        item["evidence_metrics"] = json.loads(item.pop("evidence_json", "{}") or "{}")
        item["critic"] = json.loads(item.pop("critic_json", "{}") or "{}")
        item["output_validation"] = json.loads(item.pop("output_validation_json", "{}") or "{}")
        item["trajectory"] = json.loads(item.pop("trajectory_json", "[]") or "[]")
        item["sources"] = json.loads(item.pop("sources_json", "[]") or "[]")
        results.append(item)
    return results


def get_document_counts(db_path: str | Path | None = None) -> dict[str, int]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        documents = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
        chunks = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
    return {"documents": int(documents), "chunks": int(chunks)}


def get_agent_run_counts(db_path: str | Path | None = None) -> dict[str, int]:
    init_db(db_path)
    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM agent_runs").fetchone()["count"]
        blocked = conn.execute(
            "SELECT COUNT(*) AS count FROM agent_runs WHERE critic_status = 'blocked'"
        ).fetchone()["count"]
        caution = conn.execute(
            "SELECT COUNT(*) AS count FROM agent_runs WHERE critic_status = 'caution'"
        ).fetchone()["count"]
    return {"runs": int(total), "blocked": int(blocked), "caution": int(caution)}
