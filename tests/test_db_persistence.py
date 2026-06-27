import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from src.db.repositories import (
    get_agent_run_counts,
    get_document_counts,
    get_recent_agent_runs,
    upsert_agent_run,
    upsert_document_chunks,
)


class DbPersistenceTests(unittest.TestCase):
    def test_upsert_document_chunks_records_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "app.db"
            chunks = [
                Document(
                    page_content="Company announced a dividend.",
                    metadata={
                        "doc_id": "doc_1",
                        "chunk_id": "chunk_1",
                        "source": "https://cse.lk/doc1",
                        "source_type": "url",
                        "title": "Dividend Announcement",
                        "domain": "cse.lk",
                        "primary_ticker": "COMB.N0000",
                        "primary_event": "Dividend",
                        "chunk_index": 0,
                    },
                )
            ]

            stats = upsert_document_chunks(chunks, db_path=db_path)
            counts = get_document_counts(db_path=db_path)

            self.assertEqual(stats, {"documents": 1, "chunks": 1})
            self.assertEqual(counts, {"documents": 1, "chunks": 1})

    def test_upsert_agent_run_can_be_read_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "app.db"
            upsert_agent_run(
                {
                    "run_id": "run_1",
                    "event_type": "analyst_workflow",
                    "logged_at": "2026-01-01T00:00:00+00:00",
                    "latency_sec": 1.25,
                    "request": {
                        "analysis_mode": "News Summary",
                        "ticker": "JKH.N0000",
                        "company_name": "John Keells Holdings",
                    },
                    "evidence_metrics": {
                        "evidence_score": 80,
                        "unique_source_count": 2,
                        "retrieved_chunk_count": 5,
                    },
                    "critic": {"status": "approved", "confidence": "High"},
                    "output_validation": {"status": "passed", "structure_score": 100},
                    "trajectory": [{"node": "test", "status": "completed"}],
                    "sources": [{"source_url": "https://example.com"}],
                },
                db_path=db_path,
            )

            rows = get_recent_agent_runs(limit=5, db_path=db_path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["run_id"], "run_1")
            self.assertEqual(rows[0]["critic"]["status"], "approved")
            self.assertEqual(rows[0]["output_validation"]["status"], "passed")
            self.assertEqual(rows[0]["evidence_metrics"]["evidence_score"], 80)
            self.assertEqual(get_agent_run_counts(db_path=db_path)["runs"], 1)


if __name__ == "__main__":
    unittest.main()
