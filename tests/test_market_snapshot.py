import unittest
import pandas as pd

from src.market_snapshot import CSEMarketSnapshotProvider
from src.yahoo_prices import YahooCSEClient
from src.report_intelligence import compute_commercial_financial_ratios
from src.announcement_intelligence import analyze_catalyst_impact


class TestMarketSnapshotAndCommercialFeatures(unittest.TestCase):

    def setUp(self):
        self.provider = CSEMarketSnapshotProvider()
        self.client = YahooCSEClient()

    def test_load_universe(self):
        df = self.provider.load_universe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("symbol", df.columns)

    def test_get_quote_deterministic(self):
        quote1 = self.provider.get_quote("JKH.N0000")
        quote2 = self.provider.get_quote("JKH.N0000")
        self.assertEqual(quote1.canonical_symbol, quote2.canonical_symbol)
        self.assertEqual(quote1.last_traded_price, quote2.last_traded_price)
        self.assertIsNotNone(quote1.last_traded_price)

    def test_get_history(self):
        hist = self.provider.get_history("COMB.N0000", period="1m")
        self.assertIsInstance(hist, pd.DataFrame)
        self.assertFalse(hist.empty)
        self.assertIn("Close", hist.columns)

    def test_financial_ratios(self):
        ratios = compute_commercial_financial_ratios(
            revenue=1000.0,
            operating_profit=200.0,
            pat=150.0,
            total_assets=2000.0,
            total_equity=1000.0,
            total_debt=400.0,
        )
        self.assertEqual(ratios["net_margin_pct"], 15.0)
        self.assertEqual(ratios["op_margin_pct"], 20.0)
        self.assertEqual(ratios["roe_pct"], 15.0)
        self.assertEqual(ratios["debt_to_equity"], 0.4)

    def test_catalyst_impact(self):
        cat_bullish = analyze_catalyst_impact("FIRST AND FINAL DIVIDEND ANNOUNCEMENT")
        cat_bearish = analyze_catalyst_impact("PROFIT WARNING FOR Q3 2026")
        self.assertEqual(cat_bullish["sentiment"], "Bullish")
        self.assertEqual(cat_bearish["sentiment"], "Bearish")


if __name__ == "__main__":
    unittest.main()
