import unittest

from src.guardrails.output_safety import apply_output_safety_notice, validate_analyst_answer


VALID_ANSWER = """
1. Direct Answer
- The retrieved evidence supports a cautious view.

2. Why It Matters
- The development may affect investor expectations.

3. Key Evidence
- The disclosure mentions a material event.

4. Risks / Unknowns
- The evidence does not include forward guidance.

5. Follow-up Questions
- What changed versus the prior period?
""".strip()


class OutputSafetyTests(unittest.TestCase):
    def test_valid_answer_passes_structure(self):
        validation = validate_analyst_answer(VALID_ANSWER)

        self.assertTrue(validation.passed)
        self.assertEqual(validation.structure_score, 100)
        self.assertEqual(validation.status, "passed")

    def test_missing_sections_are_reported(self):
        validation = validate_analyst_answer("1. Direct Answer\n- Short answer only.")

        self.assertFalse(validation.passed)
        self.assertEqual(validation.status, "format_warning")
        self.assertIn("2. Why It Matters", validation.missing_sections)

    def test_direct_advice_language_adds_notice(self):
        answer = VALID_ANSWER + "\n\nYou should buy this stock."
        validation = validate_analyst_answer(answer)
        gated = apply_output_safety_notice(answer, validation)

        self.assertEqual(validation.status, "caution")
        self.assertTrue(gated.startswith("Compliance note:"))


if __name__ == "__main__":
    unittest.main()
