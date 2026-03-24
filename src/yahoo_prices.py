from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass
class Quote:
    requested_symbol: str
    canonical_symbol: str
    yahoo_symbol: str
    company_name: str | None = None
    currency: str | None = None
    last_traded_price: float | None = None
    change: float | None = None
    change_pct: float | None = None
    open_price: float | None = None
    high: float | None = None
    low: float | None = None
    previous_close: float | None = None
    volume: float | None = None
    market_cap: float | None = None
    raw_info: dict[str, Any] | None = None


class YahooCSEClient:
    def __init__(self, universe_path: str | Path = "data/cse_universe.csv"):
        self.universe_path = Path(universe_path)

    # -------------------------
    # Universe helpers
    # -------------------------
    def load_universe(self) -> pd.DataFrame:
        if not self.universe_path.exists():
            return pd.DataFrame(columns=["symbol", "company_name"])

        df = pd.read_csv(self.universe_path)
        expected = {"symbol", "company_name"}
        if not expected.issubset(df.columns):
            return pd.DataFrame(columns=["symbol", "company_name"])

        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df["company_name"] = df["company_name"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)
        return df

    def normalize_symbol_text(self, value: str) -> str:
        value = (value or "").strip().upper()
        while ".." in value:
            value = value.replace("..", ".")
        return value

    def symbol_root(self, value: str) -> str:
        return self.normalize_symbol_text(value).split(".")[0].strip()

    def resolve_symbol_from_universe(self, user_symbol: str, universe_df: pd.DataFrame | None = None) -> str:
        user_symbol = self.normalize_symbol_text(user_symbol)
        if not user_symbol:
            return ""

        if universe_df is None:
            universe_df = self.load_universe()

        if universe_df.empty:
            return user_symbol

        exact = universe_df[universe_df["symbol"] == user_symbol]
        if not exact.empty:
            return exact.iloc[0]["symbol"]

        root = self.symbol_root(user_symbol)
        root_matches = universe_df[
            universe_df["symbol"].astype(str).str.upper().apply(self.symbol_root) == root
        ]
        if not root_matches.empty:
            return root_matches.iloc[0]["symbol"]

        return user_symbol

    def get_company_name(self, canonical_symbol: str, universe_df: pd.DataFrame | None = None) -> str | None:
        if universe_df is None:
            universe_df = self.load_universe()

        if universe_df.empty:
            return None

        row = universe_df[universe_df["symbol"] == canonical_symbol]
        if row.empty:
            return None
        return row.iloc[0]["company_name"]

    # -------------------------
    # Symbol mapping
    # -------------------------
    def cse_to_yahoo_symbol(self, canonical_symbol: str) -> str:
        canonical_symbol = self.normalize_symbol_text(canonical_symbol)
        if canonical_symbol.endswith(".CM"):
            return canonical_symbol
        return canonical_symbol.replace(".", "-") + ".CM"

    # -------------------------
    # Quote/history helpers
    # -------------------------
    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_quote(self, user_symbol: str) -> Quote:
        universe_df = self.load_universe()
        canonical_symbol = self.resolve_symbol_from_universe(user_symbol, universe_df)
        yahoo_symbol = self.cse_to_yahoo_symbol(canonical_symbol)

        ticker = yf.Ticker(yahoo_symbol)

        info = {}
        try:
            info = ticker.info or {}
        except Exception:
            info = {}

        hist = pd.DataFrame()
        try:
            hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        except Exception:
            hist = pd.DataFrame()

        company_name = (
            info.get("longName")
            or info.get("shortName")
            or self.get_company_name(canonical_symbol, universe_df)
        )

        currency = info.get("currency")

        last_price = change = change_pct = open_price = high = low = previous_close = volume = None

        if not hist.empty:
            hist = hist.dropna(how="all")
            if not hist.empty:
                last_row = hist.iloc[-1]

                last_price = self._safe_float(last_row.get("Close"))
                open_price = self._safe_float(last_row.get("Open"))
                high = self._safe_float(last_row.get("High"))
                low = self._safe_float(last_row.get("Low"))
                volume = self._safe_float(last_row.get("Volume"))

                if len(hist) >= 2:
                    prev_row = hist.iloc[-2]
                    previous_close = self._safe_float(prev_row.get("Close"))
                else:
                    previous_close = self._safe_float(info.get("previousClose"))

                if last_price is not None and previous_close not in (None, 0):
                    change = last_price - previous_close
                    change_pct = (change / previous_close) * 100

        if previous_close is None:
            previous_close = self._safe_float(info.get("previousClose"))

        market_cap = self._safe_float(info.get("marketCap"))

        return Quote(
            requested_symbol=self.normalize_symbol_text(user_symbol),
            canonical_symbol=canonical_symbol,
            yahoo_symbol=yahoo_symbol,
            company_name=company_name,
            currency=currency,
            last_traded_price=last_price,
            change=change,
            change_pct=change_pct,
            open_price=open_price,
            high=high,
            low=low,
            previous_close=previous_close,
            volume=volume,
            market_cap=market_cap,
            raw_info=info,
        )

    def get_history(self, user_symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        universe_df = self.load_universe()
        canonical_symbol = self.resolve_symbol_from_universe(user_symbol, universe_df)
        yahoo_symbol = self.cse_to_yahoo_symbol(canonical_symbol)

        ticker = yf.Ticker(yahoo_symbol)
        hist = ticker.history(period=period, interval=interval, auto_adjust=False)

        if hist is None or hist.empty:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

        hist = hist.reset_index()
        return hist

    def get_watchlist_quotes(self, symbols: list[str]) -> pd.DataFrame:
        rows = []
        for symbol in symbols:
            quote = self.get_quote(symbol)
            rows.append(
                {
                    "requested_symbol": quote.requested_symbol,
                    "canonical_symbol": quote.canonical_symbol,
                    "yahoo_symbol": quote.yahoo_symbol,
                    "company_name": quote.company_name,
                    "currency": quote.currency,
                    "last_traded_price": quote.last_traded_price,
                    "change": quote.change,
                    "change_pct": quote.change_pct,
                    "open_price": quote.open_price,
                    "high": quote.high,
                    "low": quote.low,
                    "previous_close": quote.previous_close,
                    "volume": quote.volume,
                    "market_cap": quote.market_cap,
                }
            )
        return pd.DataFrame(rows)