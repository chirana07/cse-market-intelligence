from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import uuid

import pandas as pd

from src.cse_announcements import CSEAnnouncementsClient
from src.yahoo_prices import YahooCSEClient


DEFAULT_ALERTS_FILE = Path("data/alerts_store.json")

RULE_TYPES = {
    "PRICE_ABOVE": "Price Above",
    "PRICE_BELOW": "Price Below",
    "RETURN_1M_ABOVE": "1M Return Above %",
    "AVG_VOLUME_20D_ABOVE": "Avg 20D Volume Above",
    "ANY_DISCLOSURE_7D": "Any Disclosure in Last 7D",
    "HIGH_PRIORITY_DISCLOSURE_30D": "High-Priority Disclosure in Last 30D",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_change(current, previous):
    current = _num(current)
    previous = _num(previous)
    if current is None or previous in (None, 0):
        return None
    return ((current - previous) / previous) * 100


def _return_from_days(df: pd.DataFrame, days: int):
    if df is None or df.empty or "Close" not in df.columns:
        return None

    work = df.copy()
    date_col = "Date" if "Date" in work.columns else work.columns[0]
    work = work.rename(columns={date_col: "Date"})
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if work.empty:
        return None

    current = _num(work["Close"].iloc[-1])
    if current is None:
        return None

    cutoff = work["Date"].iloc[-1] - pd.Timedelta(days=days)
    earlier = work[work["Date"] <= cutoff]
    if earlier.empty:
        return None

    previous = _num(earlier["Close"].iloc[-1])
    return _pct_change(current, previous)


def _avg_volume_20d(df: pd.DataFrame):
    if df is None or df.empty or "Volume" not in df.columns:
        return None
    work = df.copy()
    work["Volume"] = pd.to_numeric(work["Volume"], errors="coerce")
    work = work.dropna(subset=["Volume"]).reset_index(drop=True)
    if work.empty:
        return None
    return float(work["Volume"].tail(20).mean())


def _map_company_to_symbol(company_name: str, universe_df: pd.DataFrame) -> str:
    if not company_name or universe_df.empty:
        return ""

    target = str(company_name).strip().upper()

    exact = universe_df[universe_df["company_name"].str.upper() == target]
    if not exact.empty:
        return exact.iloc[0]["symbol"]

    root = (
        target.replace(" PLC", "")
        .replace(" LIMITED", "")
        .replace(" LTD", "")
        .replace(" THE ", " ")
        .strip()
    )

    broad = universe_df[
        universe_df["company_name"].str.upper().str.contains(root, na=False)
    ]
    if not broad.empty:
        return broad.iloc[0]["symbol"]

    return ""


def _disclosure_lookup(universe_df: pd.DataFrame, announcements_df: pd.DataFrame) -> pd.DataFrame:
    if announcements_df is None or announcements_df.empty:
        return pd.DataFrame(columns=["symbol", "any_7d_count", "high_30d_count"])

    work = announcements_df.copy()
    work["symbol"] = work["company_name"].apply(lambda x: _map_company_to_symbol(str(x), universe_df))

    if "announcement_date_parsed" not in work.columns:
        work["announcement_date_parsed"] = pd.to_datetime(
            work.get("announcement_date", ""),
            errors="coerce",
            dayfirst=True,
        )

    def score_importance(title: str, category: str) -> str:
        text = f"{title} {category}".upper()
        high_keywords = [
            "RIGHTS ISSUE",
            "ACQUISITION",
            "MERGER",
            "TAKEOVER",
            "DIVIDEND",
            "INTERIM FINANCIAL STATEMENTS",
            "ANNUAL REPORT",
            "PROFIT WARNING",
            "BOARD MEETING",
            "SHARE SPLIT",
            "DELIST",
            "BONUS ISSUE",
            "MATERIAL",
            "CAPITAL",
        ]
        if any(k in text for k in high_keywords):
            return "High"
        return "Other"

    work["importance_label"] = work.apply(
        lambda row: score_importance(
            str(row.get("announcement_title", "")),
            str(row.get("category", "")),
        ),
        axis=1,
    )

    now = pd.Timestamp.now().normalize()
    cutoff_7d = now - pd.Timedelta(days=7)
    cutoff_30d = now - pd.Timedelta(days=30)

    recent_any = (
        work[
            (work["symbol"].astype(str).str.len() > 0)
            & (work["announcement_date_parsed"] >= cutoff_7d)
        ]
        .groupby("symbol")
        .size()
        .reset_index(name="any_7d_count")
    )

    recent_high = (
        work[
            (work["symbol"].astype(str).str.len() > 0)
            & (work["announcement_date_parsed"] >= cutoff_30d)
            & (work["importance_label"] == "High")
        ]
        .groupby("symbol")
        .size()
        .reset_index(name="high_30d_count")
    )

    merged = pd.merge(recent_any, recent_high, on="symbol", how="outer").fillna(0)
    merged["any_7d_count"] = merged["any_7d_count"].astype(int)
    merged["high_30d_count"] = merged["high_30d_count"].astype(int)
    return merged


def load_alerts(file_path: str | Path = DEFAULT_ALERTS_FILE) -> list[dict]:
    file_path = Path(file_path)
    if not file_path.exists():
        return []

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_alerts(alerts: list[dict], file_path: str | Path = DEFAULT_ALERTS_FILE) -> None:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(alerts, indent=2), encoding="utf-8")


def add_alert(
    symbol: str,
    company_name: str,
    canonical_symbol: str,
    rule_type: str,
    threshold_value: float | None = None,
    notes: str = "",
    file_path: str | Path = DEFAULT_ALERTS_FILE,
) -> dict:
    alerts = load_alerts(file_path)

    alert = {
        "id": str(uuid.uuid4())[:8],
        "symbol": symbol,
        "company_name": company_name,
        "canonical_symbol": canonical_symbol,
        "rule_type": rule_type,
        "threshold_value": threshold_value,
        "notes": notes,
        "is_enabled": True,
        "created_at": _utc_now_iso(),
        "last_evaluated_at": "",
        "last_triggered_at": "",
    }

    alerts.append(alert)
    save_alerts(alerts, file_path)
    return alert


def update_alert(
    alert_id: str,
    updates: dict,
    file_path: str | Path = DEFAULT_ALERTS_FILE,
) -> None:
    alerts = load_alerts(file_path)
    for alert in alerts:
        if alert.get("id") == alert_id:
            alert.update(updates)
            break
    save_alerts(alerts, file_path)


def delete_alert(alert_id: str, file_path: str | Path = DEFAULT_ALERTS_FILE) -> None:
    alerts = load_alerts(file_path)
    alerts = [a for a in alerts if a.get("id") != alert_id]
    save_alerts(alerts, file_path)


def alerts_to_df(alerts: list[dict]) -> pd.DataFrame:
    if not alerts:
        return pd.DataFrame(
            columns=[
                "id",
                "company_name",
                "canonical_symbol",
                "rule_type",
                "threshold_value",
                "is_enabled",
                "created_at",
                "last_triggered_at",
            ]
        )
    return pd.DataFrame(alerts)


def evaluate_alerts(
    universe_path: str | Path,
    file_path: str | Path = DEFAULT_ALERTS_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alerts = load_alerts(file_path)
    alerts_df = alerts_to_df(alerts)

    if alerts_df.empty:
        return alerts_df, pd.DataFrame()

    client = YahooCSEClient(universe_path=universe_path)
    universe_df = client.load_universe()

    try:
        announcements_df = CSEAnnouncementsClient().fetch_announcements("All")
    except Exception:
        announcements_df = pd.DataFrame()

    disclosure_df = _disclosure_lookup(universe_df, announcements_df)

    disclosure_map = {}
    if not disclosure_df.empty:
        disclosure_map = disclosure_df.set_index("symbol").to_dict(orient="index")

    active_symbols = sorted(
        {
            str(a.get("canonical_symbol", "")).strip().upper()
            for a in alerts
            if a.get("is_enabled") and str(a.get("canonical_symbol", "")).strip()
        }
    )

    market_map = {}
    for symbol in active_symbols:
        try:
            quote = client.get_quote(symbol)
            hist = client.get_history(symbol, period="6mo", interval="1d")

            market_map[symbol] = {
                "quote": quote,
                "return_1m_pct": _return_from_days(hist, 30),
                "avg_volume_20d": _avg_volume_20d(hist),
            }
        except Exception:
            market_map[symbol] = {
                "quote": None,
                "return_1m_pct": None,
                "avg_volume_20d": None,
            }

    triggered_rows = []
    updated_alerts = []

    for alert in alerts:
        alert = dict(alert)
        alert["last_evaluated_at"] = _utc_now_iso()

        if not alert.get("is_enabled"):
            updated_alerts.append(alert)
            continue

        symbol = str(alert.get("canonical_symbol", "")).strip().upper()
        rule_type = str(alert.get("rule_type", "")).strip()
        threshold_value = _num(alert.get("threshold_value"))
        company_name = alert.get("company_name", symbol)

        market = market_map.get(symbol, {})
        quote = market.get("quote")
        disclosure = disclosure_map.get(symbol, {"any_7d_count": 0, "high_30d_count": 0})

        triggered = False
        reason = ""
        current_value = None

        last_price = _num(getattr(quote, "last_traded_price", None)) if quote else None
        return_1m_pct = _num(market.get("return_1m_pct"))
        avg_volume_20d = _num(market.get("avg_volume_20d"))
        any_7d_count = int(disclosure.get("any_7d_count", 0))
        high_30d_count = int(disclosure.get("high_30d_count", 0))

        if rule_type == "PRICE_ABOVE":
            current_value = last_price
            if last_price is not None and threshold_value is not None and last_price > threshold_value:
                triggered = True
                reason = f"Last price {last_price:.2f} is above threshold {threshold_value:.2f}."

        elif rule_type == "PRICE_BELOW":
            current_value = last_price
            if last_price is not None and threshold_value is not None and last_price < threshold_value:
                triggered = True
                reason = f"Last price {last_price:.2f} is below threshold {threshold_value:.2f}."

        elif rule_type == "RETURN_1M_ABOVE":
            current_value = return_1m_pct
            if return_1m_pct is not None and threshold_value is not None and return_1m_pct > threshold_value:
                triggered = True
                reason = f"1M return {return_1m_pct:.2f}% is above threshold {threshold_value:.2f}%."

        elif rule_type == "AVG_VOLUME_20D_ABOVE":
            current_value = avg_volume_20d
            if avg_volume_20d is not None and threshold_value is not None and avg_volume_20d > threshold_value:
                triggered = True
                reason = f"Avg 20D volume {avg_volume_20d:.0f} is above threshold {threshold_value:.0f}."

        elif rule_type == "ANY_DISCLOSURE_7D":
            current_value = any_7d_count
            if any_7d_count > 0:
                triggered = True
                reason = f"{any_7d_count} disclosure(s) found in the last 7 days."

        elif rule_type == "HIGH_PRIORITY_DISCLOSURE_30D":
            current_value = high_30d_count
            if high_30d_count > 0:
                triggered = True
                reason = f"{high_30d_count} high-priority disclosure(s) found in the last 30 days."

        if triggered:
            alert["last_triggered_at"] = _utc_now_iso()
            triggered_rows.append(
                {
                    "alert_id": alert.get("id"),
                    "company_name": company_name,
                    "canonical_symbol": symbol,
                    "rule_type": rule_type,
                    "threshold_value": threshold_value,
                    "current_value": current_value,
                    "reason": reason,
                    "last_triggered_at": alert["last_triggered_at"],
                }
            )

        updated_alerts.append(alert)

    save_alerts(updated_alerts, file_path)

    return alerts_to_df(updated_alerts), pd.DataFrame(triggered_rows)