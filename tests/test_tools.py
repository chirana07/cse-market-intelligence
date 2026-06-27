import unittest

from langchain_core.documents import Document

from src.tools.retrieval import evaluate_retrieved_evidence
from src.tools.schemas import MarketQuoteInput, RetrievalInput


class ToolSchemaTests(unittest.TestCase):
    def test_market_quote_input_normalizes_symbol(self):
        parsed = MarketQuoteInput(symbol=" jkh.n0000 ")

        self.assertEqual(parsed.symbol, "JKH.N0000")

    def test_retrieval_tool_returns_metrics_and_filters(self):
        output = evaluate_retrieved_evidence(
            [
                Document(
                    page_content="A sufficiently long disclosure snippet " * 10,
                    metadata={
                        "title": "Disclosure",
                        "source": "https://cse.lk/a",
                        "domain": "cse.lk",
                        "chunk_id": "chunk_a",
                    },
                )
            ],
            RetrievalInput(
                question="What happened?",
                selected_domain="cse.lk",
                selected_ticker="JKH",
            ),
        )

        self.assertEqual(len(output.formatted_docs), 1)
        self.assertEqual(output.metrics["retrieved_chunk_count"], 1)
        self.assertEqual(output.metrics["requested_filters"]["selected_ticker"], "JKH")


if __name__ == "__main__":
    unittest.main()

