# Demo Script

## One-Minute Flow

1. Open Command Center and show high-priority CSE disclosures, alerts, and portfolio context.
2. Select a company and send it to AI Analyst Copilot.
3. Ask a catalyst/risk question grounded in disclosures or reports.
4. Show the five-section answer, evidence quality score, sources, and memo export.
5. Open Agent Observability to show the run id, evidence metrics, grounding critic, output validation, and full trajectory.

## Judge Questions

**Where is the agentic AI?**
The workflow routes the request, checks guardrails, retrieves evidence, synthesizes the answer, evaluates grounding, validates output structure, and records a trace.

**How do you reduce hallucination?**
The answer is constrained to retrieved context, evidence metrics are scored, weak retrieval triggers caveats or blocking, and every run exposes sources and warnings.

**Why use RAG?**
CSE disclosures and company reports change over time. RAG grounds the analyst response in ingested, inspectable documents instead of model memory.

**How is it production-ready?**
The project includes typed tools, retry handling, SQLite persistence, JSONL logs, observability UI, guardrails, unit tests, CI, Docker, and deployment docs.

**What is the commercial angle?**
This can become a Sri Lanka-focused analyst copilot for retail investors, finance students, broker teams, and small funds that need low-cost research automation.
