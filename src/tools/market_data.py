from __future__ import annotations

from src.services import RetryConfig, run_with_retry
from src.tools.schemas import MarketQuoteInput, MarketQuoteOutput, ToolError
from src.yahoo_prices import YahooCSEClient


def get_market_quote(input_data: MarketQuoteInput, client: YahooCSEClient | None = None) -> MarketQuoteOutput | ToolError:
    client = client or YahooCSEClient()
    result = run_with_retry(
        lambda: client.get_quote(input_data.symbol),
        RetryConfig(attempts=2, backoff_sec=0.4),
    )

    if result.ok and result.value is not None:
        quote = result.value
        return MarketQuoteOutput(
            requested_symbol=quote.requested_symbol,
            canonical_symbol=quote.canonical_symbol,
            company_name=quote.company_name,
            currency=quote.currency,
            last_traded_price=quote.last_traded_price,
            change_pct=quote.change_pct,
            attempts=result.attempts,
            elapsed_sec=result.elapsed_sec,
        )

    exc = result.error or RuntimeError("Unknown market quote error")
    return ToolError(
        tool_name="get_market_quote",
        error_type=exc.__class__.__name__,
        message=str(exc),
        retryable=True,
        attempts=result.attempts,
        elapsed_sec=result.elapsed_sec,
    )
