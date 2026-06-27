import unittest

import pandas as pd

from src.portfolio_intelligence import concentration_flags, normalize_holdings_csv, portfolio_summary_metrics


class PortfolioIntelligenceTests(unittest.TestCase):
    def test_normalize_headerless_holdings_csv(self):
        raw = pd.DataFrame([["COMB.N0000", "50", "100"]], columns=["JKH.N0000", "10", "150"])

        normalized = normalize_holdings_csv(raw)

        self.assertEqual(list(normalized.columns), ["symbol", "quantity", "avg_cost"])
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized.iloc[0]["symbol"], "JKH.N0000")
        self.assertEqual(normalized.iloc[0]["quantity"], 10)

    def test_concentration_flags_for_top_holding(self):
        snapshot = pd.DataFrame(
            [
                {"weight_pct": 40, "unrealized_pnl_pct": 5},
                {"weight_pct": 20, "unrealized_pnl_pct": -20},
                {"weight_pct": 15, "unrealized_pnl_pct": -18},
            ]
        )

        flags = concentration_flags(snapshot)

        self.assertIn("Top holding exceeds 35% of portfolio value.", flags)
        self.assertIn("Top 3 holdings exceed 70% of portfolio value.", flags)
        self.assertIn("Multiple holdings are down more than 15% versus average cost.", flags)

    def test_portfolio_summary_metrics_empty(self):
        metrics = portfolio_summary_metrics(pd.DataFrame())

        self.assertEqual(metrics["holdings"], 0)
        self.assertEqual(metrics["total_market_value"], 0)


if __name__ == "__main__":
    unittest.main()

