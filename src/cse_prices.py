from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


@dataclass
class Quote:
    symbol: str
    matched_symbol: str | None = None
    name: str | None = None
    last_traded_price: float | None = None
    change: float | None = None
    change_pct: float | None = None
    high: float | None = None
    low: float | None = None
    open_price: float | None = None
    previous_close: float | None = None
    volume: float | None = None
    trades: float | None = None
    turnover: float | None = None
    market_cap: float | None = None
    raw: dict[str, Any] | None = None


class CSEPriceClient:
    def debug_symbol_matches(self, symbol: str, limit: int = 30) -> dict:
        """
        Inspect how a requested symbol compares to rows returned by the live feed.
        """
        requested = self._clean_symbol_input(symbol)
        requested_root = self._symbol_root(requested)
        rows = self._market_rows()

        symbols = []
        root_matches = []

        for row in rows:
            row_symbol = self._row_symbol(row)
            if not row_symbol:
                continue
            symbols.append(row_symbol)

            if self._symbol_root(row_symbol) == requested_root:
                root_matches.append({
                    "row_symbol": row_symbol,
                    "row_name": self._row_name(row),
                    "row": row,
                })

        return {
            "requested": requested,
            "requested_root": requested_root,
            "row_count": len(rows),
            "sample_symbols": symbols[:limit],
            "root_matches": root_matches[:limit],
        }
    def __init__(self, base_url: str = "https://www.cse.lk/api/", timeout: int = 20):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.cse.lk/",
                "Origin": "https://www.cse.lk",
            }
        )

    # -------------------------
    # Helpers
    # -------------------------
    def _clean_symbol_input(self, symbol: str) -> str:
        symbol = (symbol or "").strip().upper()
        while ".." in symbol:
            symbol = symbol.replace("..", ".")
        return symbol

    def _symbol_root(self, symbol: str) -> str:
        symbol = self._clean_symbol_input(symbol)
        return symbol.split(".")[0].strip()

    def _post(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        response = self.session.post(
            self.base_url + endpoint,
            data=data or {},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _extract_list(self, payload: Any, preferred_keys: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        if payload is None:
            return []

        if isinstance(payload, list):
            if payload and isinstance(payload[0], list):
                return [item for item in payload[0] if isinstance(item, dict)]
            return [item for item in payload if isinstance(item, dict)]

        if isinstance(payload, dict):
            for key in preferred_keys:
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
                if isinstance(value, dict):
                    for nested in value.values():
                        if isinstance(nested, list):
                            return [item for item in nested if isinstance(item, dict)]

            for value in payload.values():
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
                if isinstance(value, dict):
                    for nested in value.values():
                        if isinstance(nested, list):
                            return [item for item in nested if isinstance(item, dict)]

        return []

    def _safe_float(self, value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _pick(self, row: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in row and row.get(key) not in (None, ""):
                return row.get(key)
        return None

    def _row_symbol(self, row: dict[str, Any]) -> str:
        value = self._pick(
            row,
            "symbol",
            "stockSymbol",
            "displaySymbol",
            "securityCode",
            "symbolCode",
        )
        return self._clean_symbol_input(str(value or ""))

    def _row_name(self, row: dict[str, Any]) -> str | None:
        value = self._pick(
            row,
            "name",
            "companyName",
            "securityName",
            "displayName",
            "shortName",
        )
        return str(value) if value else None

    def _normalize_quote(self, row: dict[str, Any], requested_symbol: str) -> Quote:
        matched_symbol = self._row_symbol(row) or requested_symbol

        return Quote(
            symbol=requested_symbol,
            matched_symbol=matched_symbol,
            name=self._row_name(row),
            last_traded_price=self._safe_float(
                self._pick(row, "lastTradedPrice", "price", "ltp", "lastPrice", "close")
            ),
            change=self._safe_float(self._pick(row, "change", "priceChange")),
            change_pct=self._safe_float(
                self._pick(row, "changePercentage", "changePercent", "pctChange")
            ),
            high=self._safe_float(self._pick(row, "high", "highPrice")),
            low=self._safe_float(self._pick(row, "low", "lowPrice")),
            open_price=self._safe_float(self._pick(row, "open", "openPrice")),
            previous_close=self._safe_float(
                self._pick(row, "previousClose", "prevClose", "closePrice")
            ),
            volume=self._safe_float(self._pick(row, "shareVolume", "tradeVolume", "volume", "qty")),
            trades=self._safe_float(self._pick(row, "trades", "tradeCount")),
            turnover=self._safe_float(self._pick(row, "turnover", "tradeValue")),
            market_cap=self._safe_float(self._pick(row, "marketCap", "marketCapitalization")),
            raw=row,
        )

    # -------------------------
    # Raw endpoints
    # -------------------------
    def get_market_status(self) -> dict[str, Any]:
        return self._post("marketStatus")

    def get_market_summary(self) -> dict[str, Any]:
        return self._post("marketSummery")

    def get_aspi(self) -> dict[str, Any]:
        return self._post("aspiData")

    def get_snp(self) -> dict[str, Any]:
        return self._post("snpData")

    def get_trade_summary_raw(self) -> Any:
        return self._post("tradeSummary")

    def get_today_share_price_raw(self) -> Any:
        return self._post("todaySharePrice")

    def get_top_gainers_raw(self) -> Any:
        return self._post("topGainers")

    def get_top_losers_raw(self) -> Any:
        return self._post("topLooses")

    def get_most_active_trades_raw(self) -> Any:
        return self._post("mostActiveTrades")

    def get_company_chart_data_by_stock_raw(self, symbol: str) -> Any:
        return self._post("companyChartDataByStock", {"symbol": self._clean_symbol_input(symbol)})

    def get_chart_data_raw(self, symbol: str, chart_id: str = "1", period: str = "1") -> Any:
        return self._post(
            "chartData",
            {
                "symbol": self._clean_symbol_input(symbol),
                "chartId": chart_id,
                "period": period,
            },
        )

    # -------------------------
    # Normalized list loaders
    # -------------------------
    def get_today_share_prices(self) -> list[dict[str, Any]]:
        payload = self.get_today_share_price_raw()
        return self._extract_list(
            payload,
            preferred_keys=("todaySharePrice", "reqTodaySharePrice", "reqTradeSummery", "data"),
        )

    def get_trade_summary(self) -> list[dict[str, Any]]:
        payload = self.get_trade_summary_raw()
        return self._extract_list(payload, preferred_keys=("reqTradeSummery", "data"))

    def get_top_gainers(self) -> list[dict[str, Any]]:
        payload = self.get_top_gainers_raw()
        return self._extract_list(payload, preferred_keys=("topGainers", "data"))

    def get_top_losers(self) -> list[dict[str, Any]]:
        payload = self.get_top_losers_raw()
        return self._extract_list(payload, preferred_keys=("topLooses", "data"))

    def get_most_active_trades(self) -> list[dict[str, Any]]:
        payload = self.get_most_active_trades_raw()
        return self._extract_list(payload, preferred_keys=("mostActiveTrades", "data"))

    def _market_rows(self) -> list[dict[str, Any]]:
        rows = self.get_today_share_prices()
        if rows:
            return rows
        rows = self.get_trade_summary()
        return rows

    def get_symbol_universe(self, limit: int = 100) -> list[str]:
        rows = self._market_rows()
        symbols = []
        for row in rows:
            sym = self._row_symbol(row)
            if sym:
                symbols.append(sym)
        return sorted(set(symbols))[:limit]

    # -------------------------
    # Matching
    # -------------------------
    def _find_best_row_for_symbol(self, symbol: str) -> dict[str, Any] | None:
        target = self._clean_symbol_input(symbol)
        target_root = self._symbol_root(symbol)

        rows = self._market_rows()
        if not rows:
            return None

        # 1. exact match
        for row in rows:
            row_symbol = self._row_symbol(row)
            if row_symbol == target:
                return row

        # 2. root match
        for row in rows:
            row_symbol = self._row_symbol(row)
            if self._symbol_root(row_symbol) == target_root:
                return row

        # 3. startswith fallback
        for row in rows:
            row_symbol = self._row_symbol(row)
            if row_symbol.startswith(target_root):
                return row

        return None

    # -------------------------
    # Public quote helpers
    # -------------------------
    def get_quote(self, symbol: str) -> Quote:
        requested = self._clean_symbol_input(symbol)
        row = self._find_best_row_for_symbol(requested)

        if row is None:
            return Quote(symbol=requested)

        return self._normalize_quote(row, requested_symbol=requested)

    def get_watchlist_quotes(self, symbols: list[str]) -> list[Quote]:
        quotes: list[Quote] = []
        for sym in symbols:
            clean = self._clean_symbol_input(sym)
            if not clean:
                continue
            try:
                quotes.append(self.get_quote(clean))
            except Exception:
                quotes.append(Quote(symbol=clean))
        return quotes

    def get_price_series(self, symbol: str) -> pd.DataFrame:
        requested = self._clean_symbol_input(symbol)
        quote = self.get_quote(requested)
        chart_symbol = quote.matched_symbol or requested

        series: list[dict[str, Any]] = []

        try:
            payload = self.get_company_chart_data_by_stock_raw(chart_symbol)
            if isinstance(payload, dict):
                req_trade_summary = payload.get("reqTradeSummery", {})
                if isinstance(req_trade_summary, dict):
                    chart_data = req_trade_summary.get("chartData", [])
                    if isinstance(chart_data, list):
                        series = [row for row in chart_data if isinstance(row, dict)]
        except Exception:
            series = []

        if not series:
            try:
                payload = self.get_chart_data_raw(chart_symbol)
                if isinstance(payload, list):
                    series = [row for row in payload if isinstance(row, dict)]
                elif isinstance(payload, dict):
                    series = self._extract_list(payload, preferred_keys=("chartData", "data"))
            except Exception:
                series = []

        rows = []
        for item in series:
            ts = item.get("t") or item.get("time") or item.get("timestamp")
            price = self._safe_float(self._pick(item, "p", "price", "close", "lastTradedPrice"))
            if ts is None or price is None:
                continue

            rows.append(
                {
                    "timestamp": pd.to_datetime(ts, unit="ms", errors="coerce"),
                    "price": price,
                    "change": self._safe_float(self._pick(item, "c", "change")),
                    "change_pct": self._safe_float(self._pick(item, "pc", "changePercentage")),
                    "volume": self._safe_float(self._pick(item, "s", "volume", "qty")),
                    "trades": self._safe_float(self._pick(item, "q", "trades")),
                }
            )

        # IMPORTANT FIX: guard against empty rows before dropna
        if not rows:
            return pd.DataFrame(columns=["timestamp", "price", "change", "change_pct", "volume", "trades"])

        df = pd.DataFrame(rows)

        required_cols = {"timestamp", "price"}
        if not required_cols.issubset(df.columns):
            return pd.DataFrame(columns=["timestamp", "price", "change", "change_pct", "volume", "trades"])

        df = df.dropna(subset=["timestamp", "price"])
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "price", "change", "change_pct", "volume", "trades"])

        return df.sort_values("timestamp").reset_index(drop=True)

    def quotes_to_dataframe(self, quotes: list[Quote]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "symbol": q.symbol,
                    "matched_symbol": q.matched_symbol,
                    "name": q.name,
                    "last_traded_price": q.last_traded_price,
                    "change": q.change,
                    "change_pct": q.change_pct,
                    "high": q.high,
                    "low": q.low,
                    "open_price": q.open_price,
                    "previous_close": q.previous_close,
                    "volume": q.volume,
                    "trades": q.trades,
                    "turnover": q.turnover,
                    "market_cap": q.market_cap,
                }
                for q in quotes
            ]
        )