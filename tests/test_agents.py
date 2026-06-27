import unittest

from src.agents.critic import INSUFFICIENT_EVIDENCE_MESSAGE, apply_grounding_gate, evaluate_grounding
from src.agents.state import AnalystRequest


class AnalystAgentTests(unittest.TestCase):
    def test_enriched_question_contains_filters(self):
        request = AnalystRequest(
            company_name="John Keells Holdings",
            ticker="JKH.N0000",
            analysis_mode="Catalysts & Risks",
            research_query="What changed?",
            selected_domain="cse.lk",
            selected_source="https://example.com/report.pdf",
            selected_ticker="JKH",
            selected_event="Dividend",
        )

        enriched = request.enriched_question()

        self.assertIn("Company Focus: John Keells Holdings", enriched)
        self.assertIn("Ticker Focus: JKH.N0000", enriched)
        self.assertIn("Analysis Mode: Catalysts & Risks", enriched)
        self.assertIn("Domain Filter: cse.lk", enriched)
        self.assertIn("Research Question:", enriched)

    def test_grounding_blocks_zero_evidence(self):
        critic = evaluate_grounding(
            answer="This company has strong revenue growth.",
            evidence_metrics={"retrieved_chunk_count": 0, "evidence_score": 0},
        )

        self.assertEqual(critic["status"], "blocked")
        self.assertTrue(critic["should_replace_answer"])
        self.assertEqual(apply_grounding_gate("unsupported answer", critic), INSUFFICIENT_EVIDENCE_MESSAGE)

    def test_grounding_adds_caveat_for_low_evidence(self):
        critic = evaluate_grounding(
            answer="A cautious view is warranted.",
            evidence_metrics={
                "retrieved_chunk_count": 1,
                "evidence_score": 20,
                "confidence_label": "Low",
                "gaps_or_warnings": ["Evidence comes from only one source."],
            },
        )

        gated = apply_grounding_gate("A cautious view is warranted.", critic)

        self.assertEqual(critic["status"], "caution")
        self.assertTrue(gated.startswith("Evidence caveat:"))


if __name__ == "__main__":
    unittest.main()

