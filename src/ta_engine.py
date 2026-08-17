from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def compute_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average (SMA)."""
    return series.rolling(window=period).mean()


def compute_ema(series: pd.Series, period: int = 12) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    for i in range(period, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_macd(
    series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence (MACD).

    Returns: (macd_line, signal_line, histogram)
    """
    fast_ema = compute_ema(series, fast_period)
    slow_ema = compute_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = compute_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns: (upper_band, middle_sma, lower_band)
    """
    middle_sma = compute_sma(series, period)
    rolling_std = series.rolling(window=period).std()
    upper_band = middle_sma + (rolling_std * num_std)
    lower_band = middle_sma - (rolling_std * num_std)
    return upper_band, middle_sma, lower_band


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR). Required columns: High, Low, Close."""
    if not all(col in df.columns for col in ["High", "Low", "Close"]):
        return pd.Series(index=df.index, dtype=float)

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def compute_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (%K, %D). Required columns: High, Low, Close."""
    if not all(col in df.columns for col in ["High", "Low", "Close"]):
        empty = pd.Series(index=df.index, dtype=float)
        return empty, empty

    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()

    percent_k = 100.0 * ((df["Close"] - low_min) / ((high_max - low_min).replace(0, np.nan)))
    percent_d = percent_k.rolling(window=d_period).mean()
    return percent_k.fillna(50.0), percent_d.fillna(50.0)


def generate_technical_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate technical signals for a historical price dataframe."""
    if df.empty or "Close" not in df.columns or len(df) < 14:
        return {
            "rsi": 50.0,
            "rsi_signal": "Neutral",
            "macd_signal": "Neutral",
            "golden_cross": False,
            "death_cross": False,
            "bollinger_position": "Middle",
            "volatility_atr": 0.0,
        }

    close = df["Close"]
    latest_price = float(close.iloc[-1])

    # RSI
    rsi_series = compute_rsi(close)
    latest_rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
    if latest_rsi < 35.0:
        rsi_signal = "Oversold (Bullish Rebound Candidate)"
    elif latest_rsi > 70.0:
        rsi_signal = "Overbought (Caution)"
    else:
        rsi_signal = "Neutral"

    # MACD
    macd, signal, hist = compute_macd(close)
    if not hist.empty and len(hist) >= 2:
        latest_hist = float(hist.iloc[-1])
        prev_hist = float(hist.iloc[-2])
        if prev_hist < 0 and latest_hist > 0:
            macd_signal = "Bullish Crossover"
        elif prev_hist > 0 and latest_hist < 0:
            macd_signal = "Bearish Crossover"
        elif latest_hist > 0:
            macd_signal = "Bullish Momentum"
        else:
            macd_signal = "Bearish Momentum"
    else:
        macd_signal = "Neutral"

    # SMAs (Golden Cross / Death Cross)
    sma_50 = compute_sma(close, 50)
    sma_200 = compute_sma(close, min(200, max(20, len(close) - 1)))
    golden_cross = False
    death_cross = False

    if not sma_50.empty and not sma_200.empty:
        val_50 = sma_50.dropna().iloc[-1] if not sma_50.dropna().empty else None
        val_200 = sma_200.dropna().iloc[-1] if not sma_200.dropna().empty else None
        if val_50 is not None and val_200 is not None:
            if val_50 > val_200:
                golden_cross = True
            elif val_50 < val_200:
                death_cross = True

    # Bollinger Bands
    upper, mid, lower = compute_bollinger_bands(close)
    if not upper.empty and not lower.empty:
        u_val = float(upper.iloc[-1])
        l_val = float(lower.iloc[-1])
        if latest_price >= u_val:
            bb_pos = "Above Upper Band (Breakout)"
        elif latest_price <= l_val:
            bb_pos = "Below Lower Band (Oversold)"
        else:
            bb_pos = "Within Bands"
    else:
        bb_pos = "Within Bands"

    # ATR
    atr_series = compute_atr(df)
    latest_atr = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else 0.0

    return {
        "rsi": round(latest_rsi, 2),
        "rsi_signal": rsi_signal,
        "macd_signal": macd_signal,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "bollinger_position": bb_pos,
        "volatility_atr": round(latest_atr, 2),
    }
