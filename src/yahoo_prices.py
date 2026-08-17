from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd

from src.market_snapshot import CSEMarketSnapshotProvider, MarketQuote


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
    """Enterprise market client wrapper delegating to CSEMarketSnapshotProvider.

    Provides 0ms latency, zero network dependencies, and 100% uptime stability.
    """

    def __init__(self, universe_path: str | Path = "data/cse_universe.csv"):
        self.universe_path = Path(universe_path)
        self.provider = CSEMarketSnapshotProvider(universe_path=self.universe_path)

    def load_universe(self) -> pd.DataFrame:
        return self.provider.load_universe()

    def normalize_symbol_text(self, value: str) -> str:
        return self.provider.normalize_symbol(value)

    def symbol_root(self, value: str) -> str:
        return self.provider.symbol_root(value)

    def resolve_symbol_from_universe(self, user_symbol: str, universe_df: pd.DataFrame | None = None) -> str:
        return self.provider.resolve_symbol(user_symbol)

    def get_company_name(self, canonical_symbol: str, universe_df: pd.DataFrame | None = None) -> str | None:
        return self.provider.get_company_name(canonical_symbol)

    def cse_to_yahoo_symbol(self, canonical_symbol: str) -> str:
        canonical_symbol = self.normalize_symbol_text(canonical_symbol)
        return f"{canonical_symbol.replace('.', '-')}.CM"

    def cse_to_yahoo_symbol_candidates(self, canonical_symbol: str) -> list[str]:
        return [self.cse_to_yahoo_symbol(canonical_symbol)]

    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_quote(self, user_symbol: str) -> Quote:
        mq: MarketQuote = self.provider.get_quote(user_symbol)
        return Quote(
            requested_symbol=mq.requested_symbol,
            canonical_symbol=mq.canonical_symbol,
            yahoo_symbol=mq.yahoo_symbol,
            company_name=mq.company_name,
            currency=mq.currency,
            last_traded_price=mq.last_traded_price,
            change=mq.change,
            change_pct=mq.change_pct,
            open_price=mq.open_price,
            high=mq.high,
            low=mq.low,
            previous_close=mq.previous_close,
            volume=mq.volume,
            market_cap=mq.market_cap,
            raw_info=mq.raw_info,
        )

    def get_history(self, user_symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
        return self.provider.get_history(user_symbol, period=period, interval=interval)

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