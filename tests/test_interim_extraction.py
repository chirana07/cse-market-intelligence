import unittest
from unittest.mock import MagicMock, patch

from src.report_intelligence import extract_interim_key_figures


class InterimExtractionTests(unittest.TestCase):
    @patch("src.report_intelligence.ChatOllama")
    def test_extract_interim_key_figures_prompt_structure(self, mock_chat_ollama_cls):
        mock_llm_inst = MagicMock()
        mock_response = MagicMock()
        mock_response.content = """### Sample Company – Q3 2024

| Metric | Current Quarter | Previous Quarter | QoQ | Same Quarter Last Year | YoY |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Revenue | Rs 100M | Rs 80M | +25% | Rs 70M | +42.8% |
| PAT | Rs 15M | Rs (5M) | Turned profitable | Rs 8M | +87.5% |

### Key Takeaways
* Revenue grew 25% QoQ and 42.8% YoY.
* Company turned profitable QoQ.

### Investor Snapshot
Overall performance: Strong
Reason: Revenue expansion and margin recovery drove solid quarterly profitability.
"""
        mock_llm_inst.invoke.return_value = mock_response
        mock_chat_ollama_cls.return_value = mock_llm_inst

        result = extract_interim_key_figures(
            company_name="Sample Corp",
            ticker="SAMP.N0000",
            report_text="Quarter ended 31 Dec 2024 Revenue Rs 100M PAT Rs 15M...",
        )

        self.assertIn("Sample Company – Q3 2024", result)
        self.assertIn("Turned profitable", result)
        self.assertIn("Investor Snapshot", result)

        # Check prompt invocation
        mock_llm_inst.invoke.assert_called_once()
        invoked_prompt = mock_llm_inst.invoke.call_args[0][0]
        self.assertIn("First identify the reporting periods correctly", invoked_prompt)
        self.assertIn("Turned profitable", invoked_prompt)
        self.assertIn("Sample Corp", invoked_prompt)

    @patch("src.report_intelligence.ChatOllama")
    def test_extract_interim_key_figures_handles_empty_text(self, mock_chat_ollama_cls):
        mock_llm_inst = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "QoQ comparison not available from this report"
        mock_llm_inst.invoke.return_value = mock_response
        mock_chat_ollama_cls.return_value = mock_llm_inst

        result = extract_interim_key_figures(
            company_name="",
            ticker="",
            report_text="",
        )

        self.assertEqual(result, "QoQ comparison not available from this report")


if __name__ == "__main__":
    unittest.main()
