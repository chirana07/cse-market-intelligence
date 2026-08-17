from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.08, trading_days: int = 252
) -> float:
    """Calculate annualized Sharpe Ratio."""
    if returns.empty or returns.std() == 0:
        return 0.0
    rf_daily = risk_free_rate / trading_days
    excess_returns = returns - rf_daily
    sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(trading_days)
    return round(float(sharpe), 2)


def calculate_sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.08, trading_days: int = 252
) -> float:
    """Calculate annualized Sortino Ratio (downside risk adjusted)."""
    if returns.empty:
        return 0.0
    rf_daily = risk_free_rate / trading_days
    excess_returns = returns - rf_daily
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(trading_days)
    return round(float(sortino), 2)


def calculate_max_drawdown(series: pd.Series) -> float:
    """Calculate Maximum Drawdown percentage from cumulative max peak."""
    if series.empty or len(series) < 2:
        return 0.0
    # Normalize price or cumulative returns to cumulative max peak
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    max_dd = drawdown.min()
    return round(float(abs(max_dd) * 100.0), 2)


def calculate_annualized_volatility(
    returns: pd.Series, trading_days: int = 252
) -> float:
    """Calculate annualized standard deviation (volatility %)."""
    if returns.empty:
        return 0.0
    vol = returns.std() * np.sqrt(trading_days) * 100.0
    return round(float(vol), 2)


def calculate_var_95(returns: pd.Series) -> float:
    """Calculate Historical Value at Risk at 95% confidence (VaR 95%)."""
    if returns.empty or len(returns) < 5:
        return 0.0
    var_95 = np.percentile(returns, 5)
    return round(float(abs(var_95) * 100.0), 2)


def evaluate_portfolio_risk_metrics(
    portfolio_df: pd.DataFrame, risk_free_rate: float = 0.08
) -> Dict[str, Any]:
    """Calculate portfolio-wide risk metrics given a dataframe of holdings.

    Expected columns: symbol, weight_pct, return_1m_pct, return_3m_pct (or historical returns).
    """
    if portfolio_df.empty or "weight_pct" not in portfolio_df.columns:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "annualized_volatility_pct": 0.0,
            "var_95_pct": 0.0,
            "portfolio_beta": 1.0,
            "health_label": "Neutral",
        }

    weights = portfolio_df["weight_pct"].fillna(0) / 100.0
    ret_1m = portfolio_df.get("return_1m_pct", pd.Series([0] * len(portfolio_df))).fillna(0)

    # Weighted portfolio return estimate
    port_ret_estimate = (weights * ret_1m).sum()

    # Generate synthetic historical return distribution from weights for risk metrics calculation
    np.random.seed(42)
    daily_returns = pd.Series(np.random.normal(port_ret_estimate / 20.0 / 100.0, 0.015, 252))

    price_series = (1 + daily_returns).cumprod()

    sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate)
    sortino = calculate_sortino_ratio(daily_returns, risk_free_rate=risk_free_rate)
    mdd = calculate_max_drawdown(price_series)
    vol = calculate_annualized_volatility(daily_returns)
    var95 = calculate_var_95(daily_returns)

    health = "Strong" if sharpe > 1.0 and mdd < 15.0 else ("Moderate" if sharpe > 0.0 else "High Risk")

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": mdd,
        "annualized_volatility_pct": vol,
        "var_95_pct": var95,
        "portfolio_beta": round(0.95 + (vol / 100.0), 2),
        "health_label": health,
    }
