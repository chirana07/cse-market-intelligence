import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

from src.config import CHAT_MODEL

CACHE_SCHEMA_VERSION = "v1"

CACHE_DIR = Path("data/cache")
ANNOUNCEMENTS_DIR = CACHE_DIR / "announcements"
REPORTS_DIR = CACHE_DIR / "reports"
STOCKS_DIR = CACHE_DIR / "stocks"
PORTFOLIO_DIR = CACHE_DIR / "portfolio"
MEMOS_DIR = CACHE_DIR / "memos"
BENCHMARKS_DIR = CACHE_DIR / "benchmarks"

for d in [ANNOUNCEMENTS_DIR, REPORTS_DIR, STOCKS_DIR, PORTFOLIO_DIR, MEMOS_DIR, BENCHMARKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def _safe_hash(*args) -> str:
    combined = "|".join(str(a) for a in args).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()[:16]


def artifact_exists(path: Path) -> bool:
    return path.exists()


def save_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: Path, text: str):
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


# --- Cache Key Builders ---

def build_announcement_cache_key(company_name: str, title: str, source_url: str) -> str:
    return _safe_hash(company_name, title, source_url)

def build_report_cache_key(company_name: str, ticker: str, report_label_or_url: str) -> str:
    return _safe_hash(company_name, ticker, report_label_or_url)

def build_stock_cache_key(company_name: str, ticker: str, deps_hash: str = "") -> str:
    return _safe_hash(company_name, ticker, deps_hash)

def build_portfolio_cache_key(snapshot_df) -> str:
    if snapshot_df.empty:
        return "empty"
    # Stable snapshot hash based on canonical symbol and weight or quantity
    sorted_df = snapshot_df.sort_values("canonical_symbol")
    components = [
        f"{row['canonical_symbol']}_{round(row.get('weight_pct', 0), 2)}"
        for _, row in sorted_df.iterrows()
    ]
    return _safe_hash(*components)

def build_memo_cache_key(company_name: str, ticker: str, analysis_mode: str, query: str) -> str:
    return _safe_hash(company_name, ticker, analysis_mode, query)


# --- Wrappers ---

def save_announcement_artifacts(cache_key: str, text: str, summary: str, event: dict, financials: dict, meta: dict):
    dir_path = ANNOUNCEMENTS_DIR / cache_key
    dir_path.mkdir(exist_ok=True)
    
    save_text(dir_path / "text.txt", text)
    save_text(dir_path / "summary.txt", summary)
    save_json(dir_path / "event.json", event)
    save_json(dir_path / "financials.json", financials)
    
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    meta["version"] = CACHE_SCHEMA_VERSION
    meta["model"] = CHAT_MODEL
    save_json(dir_path / "meta.json", meta)


def load_announcement_artifacts(cache_key: str):
    dir_path = ANNOUNCEMENTS_DIR / cache_key
    if not dir_path.exists():
        return None
        
    return {
        "text": load_text(dir_path / "text.txt"),
        "summary": load_text(dir_path / "summary.txt"),
        "event": load_json(dir_path / "event.json"),
        "financials": load_json(dir_path / "financials.json"),
        "meta": load_json(dir_path / "meta.json"),
    }


def save_report_artifacts(cache_key: str, text: str, summary: str, event: dict, financials: dict, meta: dict):
    dir_path = REPORTS_DIR / cache_key
    dir_path.mkdir(exist_ok=True)
    
    save_text(dir_path / "text.txt", text)
    save_text(dir_path / "summary.txt", summary)
    save_json(dir_path / "event.json", event)
    save_json(dir_path / "financials.json", financials)
    
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    meta["version"] = CACHE_SCHEMA_VERSION
    meta["model"] = CHAT_MODEL
    save_json(dir_path / "meta.json", meta)


def load_report_artifacts(cache_key: str):
    dir_path = REPORTS_DIR / cache_key
    if not (dir_path / "summary.txt").exists():
        return None
        
    return {
        "text": load_text(dir_path / "text.txt"),
        "summary": load_text(dir_path / "summary.txt"),
        "event": load_json(dir_path / "event.json"),
        "financials": load_json(dir_path / "financials.json"),
        "meta": load_json(dir_path / "meta.json"),
    }


def save_stock_ai_view(cache_key: str, ai_view: str, meta: dict):
    dir_path = STOCKS_DIR / cache_key
    dir_path.mkdir(exist_ok=True)
    
    save_text(dir_path / "ai_view.md", ai_view)
    
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    meta["version"] = CACHE_SCHEMA_VERSION
    meta["model"] = CHAT_MODEL
    save_json(dir_path / "meta.json", meta)


def load_stock_ai_view(cache_key: str):
    dir_path = STOCKS_DIR / cache_key
    if not (dir_path / "ai_view.md").exists():
        return None
        
    return {
        "ai_view": load_text(dir_path / "ai_view.md"),
        "meta": load_json(dir_path / "meta.json"),
    }


def save_portfolio_review(cache_key: str, review: str, meta: dict):
    dir_path = PORTFOLIO_DIR / cache_key
    dir_path.mkdir(exist_ok=True)
    
    save_text(dir_path / "review.md", review)
    
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    meta["version"] = CACHE_SCHEMA_VERSION
    meta["model"] = CHAT_MODEL
    save_json(dir_path / "meta.json", meta)


def load_portfolio_review(cache_key: str):
    dir_path = PORTFOLIO_DIR / cache_key
    if not (dir_path / "review.md").exists():
        return None
        
    return {
        "review": load_text(dir_path / "review.md"),
        "meta": load_json(dir_path / "meta.json"),
    }


def save_memo_artifact(cache_key: str, memo_text: str, meta: dict):
    dir_path = MEMOS_DIR / cache_key
    dir_path.mkdir(exist_ok=True)
    
    save_text(dir_path / "memo.md", memo_text)
    
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    meta["version"] = CACHE_SCHEMA_VERSION
    meta["model"] = CHAT_MODEL
    save_json(dir_path / "meta.json", meta)


def load_recent_memos(limit: int = 10):
    memos = []
    for dir_path in MEMOS_DIR.iterdir():
        if dir_path.is_dir() and (dir_path / "meta.json").exists():
            meta = load_json(dir_path / "meta.json")
            memos.append({
                "key": dir_path.name,
                "memo": load_text(dir_path / "memo.md"),
                "meta": meta,
                "created_at": meta.get("created_at", ""),
            })
            
    # Sort by created_at descending
    memos.sort(key=lambda x: x["created_at"], reverse=True)
    return memos[:limit]

def save_benchmark_results(results: dict):
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = BENCHMARKS_DIR / "latest_results.json"
    save_json(file_path, results)

def load_latest_benchmark() -> dict:
    file_path = BENCHMARKS_DIR / "latest_results.json"
    if file_path.exists():
        return load_json(file_path)
    return {}
