import unittest
import numpy as np
import pandas as pd

from src.ta_engine import (
    compute_sma,
    compute_ema,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    generate_technical_signals,
)
from src.portfolio_risk import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_annualized_volatility,
    evaluate_portfolio_risk_metrics,
)


class TestTechnicalAnalysisAndRisk(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
        close_prices = 100.0 + np.cumsum(np.random.normal(0.5, 1.5, 100))
        self.df = pd.DataFrame(
            {
                "Date": dates,
                "Close": close_prices,
                "High": close_prices + 2.0,
                "Low": close_prices - 2.0,
                "Volume": np.random.randint(1000, 50000, 100),
            }
        )

    def test_sma_and_ema(self):
        sma_20 = compute_sma(self.df["Close"], 20)
        ema_12 = compute_ema(self.df["Close"], 12)
        self.assertEqual(len(sma_20), 100)
        self.assertEqual(len(ema_12), 100)
        self.assertFalse(sma_20.dropna().empty)
        self.assertFalse(ema_12.dropna().empty)

    def test_rsi(self):
        rsi = compute_rsi(self.df["Close"], 14)
        self.assertEqual(len(rsi), 100)
        # RSI values must be bounded between 0 and 100
        self.assertTrue((rsi >= 0.0).all())
        self.assertTrue((rsi <= 100.0).all())

    def test_macd(self):
        macd, signal, hist = compute_macd(self.df["Close"])
        self.assertEqual(len(macd), 100)
        self.assertEqual(len(signal), 100)
        self.assertEqual(len(hist), 100)

    def test_bollinger_bands(self):
        upper, mid, lower = compute_bollinger_bands(self.df["Close"], 20, 2.0)
        valid_idx = upper.dropna().index
        self.assertTrue((upper.loc[valid_idx] >= mid.loc[valid_idx]).all())
        self.assertTrue((mid.loc[valid_idx] >= lower.loc[valid_idx]).all())

    def test_technical_signals(self):
        signals = generate_technical_signals(self.df)
        self.assertIn("rsi", signals)
        self.assertIn("macd_signal", signals)
        self.assertIn("golden_cross", signals)
        self.assertIsInstance(signals["rsi"], float)

    def test_sharpe_and_sortino_ratios(self):
        returns = self.df["Close"].pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns)
        sortino = calculate_sortino_ratio(returns)
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(sortino, float)

    def test_max_drawdown(self):
        mdd = calculate_max_drawdown(self.df["Close"])
        self.assertGreaterEqual(mdd, 0.0)

    def test_portfolio_risk_evaluation(self):
        portfolio_df = pd.DataFrame(
            {
                "symbol": ["JKH.N0000", "COMB.N0000"],
                "weight_pct": [60.0, 40.0],
                "return_1m_pct": [5.2, -1.8],
            }
        )
        metrics = evaluate_portfolio_risk_metrics(portfolio_df)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("max_drawdown_pct", metrics)
        self.assertIn("annualized_volatility_pct", metrics)


if __name__ == "__main__":
    unittest.main()
