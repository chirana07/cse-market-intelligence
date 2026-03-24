from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.yahoo_prices import YahooCSEClient
from src.cse_announcements import CSEAnnouncementsClient


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


def _annualized_volatility(df: pd.DataFrame):
    if df is None or df.empty or "Close" not in df.columns:
        return None

    work = df.copy()
    work["Close"] = pd.to_numeric(work["Close"], errors="coerce")
    work = work.dropna(subset=["Close"]).reset_index(drop=True)
    if len(work) < 20:
        return None

    returns = work["Close"].pct_change().dropna()
    if returns.empty:
        return None

    return float(returns.std() * (252 ** 0.5) * 100)


def _avg_volume_20d(df: pd.DataFrame):
    if df is None or df.empty or "Volume" not in df.columns:
        return None
    work = df.copy()
    work["Volume"] = pd.to_numeric(work["Volume"], errors="coerce")
    work = work.dropna(subset=["Volume"]).reset_index(drop=True)
    if work.empty:
        return None
    return float(work["Volume"].tail(20).mean())


def build_announcement_lookup(universe_df: pd.DataFrame, announcements_df: pd.DataFrame) -> pd.DataFrame:
    if announcements_df is None or announcements_df.empty:
        return pd.DataFrame(columns=["symbol", "announcement_count", "high_priority_count"])

    def lookup_symbol(company_name: str) -> str:
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
        broad = universe_df[universe_df["company_name"].str.upper().str.contains(root, na=False)]
        if not broad.empty:
            return broad.iloc[0]["symbol"]

        return ""

    work = announcements_df.copy()
    work["symbol"] = work["company_name"].apply(lookup_symbol)

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

    grouped = (
        work[work["symbol"].astype(str).str.len() > 0]
        .groupby("symbol")
        .agg(
            announcement_count=("symbol", "size"),
            high_priority_count=("importance_label", lambda s: int((s == "High").sum())),
        )
        .reset_index()
    )

    return grouped


def build_screening_dataset(
    universe_df: pd.DataFrame,
    universe_path: str | Path,
    announcements_df: pd.DataFrame | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    if universe_df is None or universe_df.empty:
        return pd.DataFrame()

    client = YahooCSEClient(universe_path=universe_path)
    ann_lookup = build_announcement_lookup(universe_df, announcements_df if announcements_df is not None else pd.DataFrame())

    rows = []
    subset = universe_df.head(limit).copy()

    for _, row in subset.iterrows():
        symbol = str(row["symbol"]).strip().upper()
        company_name = str(row["company_name"]).strip()

        try:
            quote = client.get_quote(symbol)
            hist = client.get_history(symbol, period="1y", interval="1d")

            last_price = quote.last_traded_price
            market_cap = quote.market_cap
            vol_20d = _avg_volume_20d(hist)
            ret_1m = _return_from_days(hist, 30)
            ret_3m = _return_from_days(hist, 90)
            ret_6m = _return_from_days(hist, 180)
            volatility = _annualized_volatility(hist)

            rows.append(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "yahoo_symbol": quote.yahoo_symbol,
                    "last_price": last_price,
                    "market_cap": market_cap,
                    "avg_volume_20d": vol_20d,
                    "return_1m_pct": ret_1m,
                    "return_3m_pct": ret_3m,
                    "return_6m_pct": ret_6m,
                    "volatility_pct": volatility,
                }
            )
        except Exception:
            rows.append(
                {
                    "symbol": symbol,
                    "company_name": company_name,
                    "yahoo_symbol": "",
                    "last_price": None,
                    "market_cap": None,
                    "avg_volume_20d": None,
                    "return_1m_pct": None,
                    "return_3m_pct": None,
                    "return_6m_pct": None,
                    "volatility_pct": None,
                }
            )

    df = pd.DataFrame(rows)

    if not ann_lookup.empty:
        df = df.merge(ann_lookup, on="symbol", how="left")
    else:
        df["announcement_count"] = 0
        df["high_priority_count"] = 0

    df["announcement_count"] = df["announcement_count"].fillna(0).astype(int)
    df["high_priority_count"] = df["high_priority_count"].fillna(0).astype(int)

    return df


def apply_nl_screener_hint(df: pd.DataFrame, prompt: str) -> pd.DataFrame:
    if df.empty or not prompt.strip():
        return df

    q = prompt.strip().lower()
    work = df.copy()

    if "dividend" in q:
        work = work[work["high_priority_count"] >= 1]

    if "momentum" in q or "strong performers" in q or "winners" in q:
        work = work[work["return_3m_pct"].fillna(-999) > 0]

    if "low volatility" in q or "stable" in q:
        work = work[work["volatility_pct"].fillna(999) < 40]

    if "liquid" in q or "high volume" in q:
        work = work[work["avg_volume_20d"].fillna(0) > 100000]

    if "recent disclosures" in q or "news" in q or "announcements" in q:
        work = work[work["announcement_count"] > 0]

    if "high conviction" in q or "important disclosures" in q:
        work = work[work["high_priority_count"] > 0]

    return work.reset_index(drop=True)