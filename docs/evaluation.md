# Evaluation Plan

## Automated Checks

Run:

```bash
python3 -m unittest discover tests
python3 -m compileall src scripts app.py
python3 scripts/run_benchmarks.py
```

## Metrics

- Task success: benchmark pass rate and expected topic coverage.
- Retrieval quality: chunk count, source count, domain diversity, evidence score.
- Groundedness: unsupported-answer blocking, low-evidence caveats, source warnings.
- Output quality: required five-section format and direct-advice language flags.
- Tool correctness: typed tool outputs, retry metadata, structured tool errors.
- Latency: agent run `latency_sec` captured in SQLite and JSONL.
- Reliability: unit tests for portfolio parsing, RAG metrics, DB persistence, retries, input guardrails, and output guardrails.

## Human Review

Use the Agent Observability page to inspect run traces, critic decisions, evidence metrics, output-validation status, sources, and trajectory. This is the primary technical-defense view for judges or code reviewers.
