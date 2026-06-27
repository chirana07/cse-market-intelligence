import unittest

from src.rag_evaluation import compute_retrieval_metrics


class RagEvaluationTests(unittest.TestCase):
    def test_zero_retrieval_is_low_confidence(self):
        metrics = compute_retrieval_metrics([])

        self.assertEqual(metrics["retrieved_chunk_count"], 0)
        self.assertEqual(metrics["confidence_label"], "Low")
        self.assertIn("No evidence retrieved.", metrics["gaps_or_warnings"])

    def test_multiple_sources_raise_evidence_score(self):
        docs = [
            {
                "source_url": "https://cse.lk/a",
                "domain": "cse.lk",
                "snippet_length": 500,
                "has_title": True,
                "has_source": True,
            },
            {
                "source_url": "https://example.com/b",
                "domain": "example.com",
                "snippet_length": 500,
                "has_title": True,
                "has_source": True,
            },
            {
                "source_url": "https://example.com/c",
                "domain": "example.com",
                "snippet_length": 500,
                "has_title": True,
                "has_source": True,
            },
        ]

        metrics = compute_retrieval_metrics(docs)

        self.assertGreaterEqual(metrics["unique_source_count"], 3)
        self.assertGreater(metrics["evidence_score"], 40)


if __name__ == "__main__":
    unittest.main()

