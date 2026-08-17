import unittest

from src.report_intelligence import generate_executive_tear_sheet
from src.announcement_intelligence import analyze_catalyst_impact


class TestCompetitionShowstopperFeatures(unittest.TestCase):

    def test_executive_tear_sheet_generation(self):
        sample_report_text = """
        John Keells Holdings PLC Quarterly Financial Statements for Q3 2026.
        Revenue increased by 18.5% YoY to LKR 48.2 Billion compared to LKR 40.6 Billion.
        Profit After Tax (PAT) expanded by 24.2% YoY to LKR 6.8 Billion.
        Operating Margin expanded from 12.1% to 14.5% driven by port operations and leisure recovery.
        Borrowings reduced by LKR 2.1 Billion. Cash and cash equivalents stand at LKR 18.4 Billion.
        """
        tear_sheet = generate_executive_tear_sheet(
            company_name="John Keells Holdings PLC",
            ticker="JKH.N0000",
            report_text=sample_report_text,
        )
        self.assertIn("company_name", tear_sheet)
        self.assertEqual(tear_sheet["ticker"], "JKH.N0000")
        self.assertIn("verdict", tear_sheet)
        self.assertIn("markdown_tear_sheet", tear_sheet)
        self.assertGreater(len(tear_sheet["markdown_tear_sheet"]), 50)

    def test_catalyst_radar_sentiments(self):
        div_cat = analyze_catalyst_impact("INTERIM DIVIDEND ANNOUNCEMENT LKR 2.50 PER SHARE")
        warning_cat = analyze_catalyst_impact("PROFIT WARNING FOR QUARTER ENDED MARCH 2026")
        routine_cat = analyze_catalyst_impact("NOTICE OF ANNUAL GENERAL MEETING")

        self.assertEqual(div_cat["sentiment"], "Bullish")
        self.assertEqual(warning_cat["sentiment"], "Bearish")
        self.assertEqual(routine_cat["sentiment"], "Neutral")


if __name__ == "__main__":
    unittest.main()
