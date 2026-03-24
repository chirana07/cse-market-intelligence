from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from langchain_ollama import ChatOllama

from src.config import CHAT_MODEL, OLLAMA_BASE_URL
from src.yahoo_prices import YahooCSEClient


@dataclass
class HoldingSnapshot:
    symbol: str
    company_name: str
    quantity: float
    avg_cost: float
    canonical_symbol: str
    yahoo_symbol: str
    last_price: float | None
    market_value: float | None
    cost_basis: float | None
    unrealized_pnl: float | None
    unrealized_pnl_pct: float | None


def _num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_mul(a, b):
    a = _num(a)
    b = _num(b)
    if a is None or b is None:
        return None
    return a * b


def _safe_pct(current, base):
    current = _num(current)
    base = _num(base)
    if current is None or base in (None, 0):
        return None
    return (current / base) * 100


def normalize_holdings_csv(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    original_columns = list(work.columns)
    work.columns = [str(c).strip().lower() for c in work.columns]

    required_aliases = {
        "symbol": ["symbol", "ticker"],
        "quantity": ["quantity", "qty", "shares"],
        "avg_cost": ["avg_cost", "average_cost", "cost", "buy_price", "avg price"],
    }

    mapped = {}
    for standard, aliases in required_aliases.items():
        for alias in aliases:
            if alias in work.columns:
                mapped[standard] = alias
                break

    # -- Fallback for headerless CSVs --
    if set(mapped.keys()) != {"symbol", "quantity", "avg_cost"}:
        if len(work.columns) == 3:
            # If we didn't find headers, but we have 3 columns, check if the first row (the 'columns')
            # looks like data rather than headers.
            # pd.read_csv(StringIO("JKH.N,100,150")) results in columns=["JKH.N", "100", "150"]
            # If the last two 'columns' are numbers, then it's highly likely headerless.
            try:
                # Check if columns 1 and 2 (quantity and cost) are numeric-like
                float(str(original_columns[1]).replace(",", ""))
                float(str(original_columns[2]).replace(",", ""))
                
                # If so, create a new row from the current columns and prepend it
                headers_as_data = pd.DataFrame([original_columns], columns=range(3))
                work.columns = range(3)
                work = pd.concat([headers_as_data, work], ignore_index=True)
                
                mapped = {"symbol": 0, "quantity": 1, "avg_cost": 2}
            except (ValueError, TypeError, IndexError):
                return pd.DataFrame(columns=["symbol", "quantity", "avg_cost"])
        else:
            return pd.DataFrame(columns=["symbol", "quantity", "avg_cost"])

    cleaned = pd.DataFrame()
    cleaned["symbol"] = work[mapped["symbol"]].astype(str).str.strip().str.upper()
    cleaned["quantity"] = pd.to_numeric(work[mapped["quantity"]], errors="coerce")
    cleaned["avg_cost"] = pd.to_numeric(work[mapped["avg_cost"]], errors="coerce")

    cleaned = cleaned.dropna(subset=["symbol", "quantity", "avg_cost"])
    cleaned = cleaned[cleaned["symbol"].str.len() > 0].reset_index(drop=True)
    return cleaned


def build_portfolio_snapshot(
    holdings_df: pd.DataFrame,
    universe_path: str | Path,
) -> pd.DataFrame:
    if holdings_df is None or holdings_df.empty:
        return pd.DataFrame()

    client = YahooCSEClient(universe_path=universe_path)
    universe_df = client.load_universe()

    rows = []
    for _, row in holdings_df.iterrows():
        requested_symbol = str(row["symbol"]).strip().upper()
        quantity = _num(row["quantity"])
        avg_cost = _num(row["avg_cost"])

        try:
            canonical_symbol = client.resolve_symbol_from_universe(requested_symbol, universe_df)
            quote = client.get_quote(canonical_symbol)
            company_name = quote.company_name or client.get_company_name(canonical_symbol, universe_df) or canonical_symbol
            last_price = _num(quote.last_traded_price)

            market_value = _safe_mul(quantity, last_price)
            cost_basis = _safe_mul(quantity, avg_cost)
            unrealized_pnl = None
            if market_value is not None and cost_basis is not None:
                unrealized_pnl = market_value - cost_basis

            unrealized_pnl_pct = None
            if unrealized_pnl is not None and cost_basis not in (None, 0):
                unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100

            rows.append(
                {
                    "symbol": requested_symbol,
                    "company_name": company_name,
                    "quantity": quantity,
                    "avg_cost": avg_cost,
                    "canonical_symbol": canonical_symbol,
                    "yahoo_symbol": quote.yahoo_symbol,
                    "last_price": last_price,
                    "market_value": market_value,
                    "cost_basis": cost_basis,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct,
                }
            )
        except Exception:
            rows.append(
                {
                    "symbol": requested_symbol,
                    "company_name": requested_symbol,
                    "quantity": quantity,
                    "avg_cost": avg_cost,
                    "canonical_symbol": requested_symbol,
                    "yahoo_symbol": "",
                    "last_price": None,
                    "market_value": None,
                    "cost_basis": _safe_mul(quantity, avg_cost),
                    "unrealized_pnl": None,
                    "unrealized_pnl_pct": None,
                }
            )

    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        return snapshot

    total_market_value = snapshot["market_value"].fillna(0).sum()
    snapshot["weight_pct"] = snapshot["market_value"].apply(
        lambda x: (x / total_market_value * 100) if total_market_value and pd.notna(x) else None
    )

    return snapshot.sort_values("market_value", ascending=False, na_position="last").reset_index(drop=True)


def portfolio_summary_metrics(snapshot_df: pd.DataFrame) -> dict:
    if snapshot_df is None or snapshot_df.empty:
        return {
            "holdings": 0,
            "total_market_value": 0,
            "total_cost_basis": 0,
            "total_unrealized_pnl": 0,
            "total_unrealized_pnl_pct": None,
            "top_weight_pct": None,
        }

    total_market_value = snapshot_df["market_value"].fillna(0).sum()
    total_cost_basis = snapshot_df["cost_basis"].fillna(0).sum()
    total_unrealized_pnl = snapshot_df["unrealized_pnl"].fillna(0).sum()
    total_unrealized_pnl_pct = None

    if total_cost_basis not in (None, 0):
        total_unrealized_pnl_pct = (total_unrealized_pnl / total_cost_basis) * 100

    top_weight_pct = None
    if "weight_pct" in snapshot_df.columns and not snapshot_df["weight_pct"].dropna().empty:
        top_weight_pct = float(snapshot_df["weight_pct"].dropna().max())

    return {
        "holdings": int(len(snapshot_df)),
        "total_market_value": float(total_market_value),
        "total_cost_basis": float(total_cost_basis),
        "total_unrealized_pnl": float(total_unrealized_pnl),
        "total_unrealized_pnl_pct": total_unrealized_pnl_pct,
        "top_weight_pct": top_weight_pct,
    }


def concentration_flags(snapshot_df: pd.DataFrame) -> list[str]:
    if snapshot_df is None or snapshot_df.empty:
        return []

    flags = []

    weights = snapshot_df["weight_pct"].dropna().tolist() if "weight_pct" in snapshot_df.columns else []
    if not weights:
        return flags

    top1 = max(weights)
    top3 = sum(sorted(weights, reverse=True)[:3])

    if top1 >= 35:
        flags.append("Top holding exceeds 35% of portfolio value.")
    elif top1 >= 25:
        flags.append("Top holding exceeds 25% of portfolio value.")

    if top3 >= 70:
        flags.append("Top 3 holdings exceed 70% of portfolio value.")
    elif top3 >= 55:
        flags.append("Top 3 holdings exceed 55% of portfolio value.")

    losers = snapshot_df["unrealized_pnl_pct"].dropna()
    if not losers.empty and (losers < -15).sum() >= 2:
        flags.append("Multiple holdings are down more than 15% versus average cost.")

    return flags


def _get_llm():
    return ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


def generate_portfolio_review(snapshot_df: pd.DataFrame) -> str:
    if snapshot_df is None or snapshot_df.empty:
        return "No portfolio data available."

    llm = _get_llm()

    preview_cols = [
        "canonical_symbol",
        "company_name",
        "weight_pct",
        "market_value",
        "unrealized_pnl_pct",
    ]
    preview = snapshot_df[preview_cols].copy()
    preview = preview.fillna("")
    portfolio_table = preview.to_string(index=False)

    metrics = portfolio_summary_metrics(snapshot_df)
    flags = concentration_flags(snapshot_df)
    flags_text = "\n".join(f"- {flag}" for flag in flags) if flags else "No major concentration flags detected."

    prompt = f"""
You are an equity research assistant focused on the Colombo Stock Exchange.

Review the following portfolio snapshot.
Use only the information provided.
Do not invent facts.

Return the answer in this exact structure:

1. Portfolio Snapshot
- 3 to 6 bullet points

2. Concentration Risk
- Bullet points

3. Strongest Holdings
- Bullet points

4. Weakest Holdings
- Bullet points

5. Rebalance / Monitoring Ideas
- Bullet points

6. Questions For The Investor
- Bullet points

Portfolio Metrics:
- Holdings: {metrics['holdings']}
- Total Market Value: {metrics['total_market_value']}
- Total Cost Basis: {metrics['total_cost_basis']}
- Total Unrealized PnL: {metrics['total_unrealized_pnl']}
- Total Unrealized PnL %: {metrics['total_unrealized_pnl_pct']}
- Top Weight %: {metrics['top_weight_pct']}

Concentration Flags:
{flags_text}

Portfolio Table:
{portfolio_table}

Answer:
""".strip()

    result = llm.invoke(prompt)
    return result.content if hasattr(result, "content") else str(result)