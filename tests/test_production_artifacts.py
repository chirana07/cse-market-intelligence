import unittest
from pathlib import Path

from pydantic import ValidationError

from src.api.schemas import ResearchQueryRequest


class ProductionArtifactTests(unittest.TestCase):
    def test_deployment_and_docs_files_exist(self):
        expected_paths = [
            ".github/workflows/ci.yml",
            ".dockerignore",
            "Dockerfile",
            "docker-compose.yml",
            "docs/architecture.md",
            "docs/evaluation.md",
            "docs/demo_script.md",
            "docs/deployment.md",
            "src/api/main.py",
        ]

        for path in expected_paths:
            self.assertTrue(Path(path).exists(), path)

    def test_api_research_query_schema_validates_question(self):
        payload = ResearchQueryRequest(question="Summarize JKH risks.", ticker="JKH.N0000")

        self.assertEqual(payload.question, "Summarize JKH risks.")
        self.assertEqual(payload.ticker, "JKH.N0000")

        with self.assertRaises(ValidationError):
            ResearchQueryRequest(question="")


if __name__ == "__main__":
    unittest.main()
