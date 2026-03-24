import json
import time
from src.persistence import save_benchmark_results
from src.evidence_formatter import format_evidence
from src.rag_evaluation import compute_retrieval_metrics
from src.benchmark_metrics import evaluate_benchmark_case

def load_benchmark_cases(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_single_benchmark_case(case: dict, vectorstore, chain, use_llm_grader: bool = False):
    question = case["question"]
    
    # Emulate the Analyst Workspace injection template securely.
    enriched_question = (
        f"Company Focus: {case.get('company_name', 'Not specified')}\n"
        f"Ticker Focus: {case.get('ticker', 'Not specified')}\n"
        f"Analysis Mode: {case.get('analysis_mode', 'General Search')}\n\n"
        f"Research Question:\n{question}"
    )
    
    start_time = time.time()
    try:
        result = chain.invoke({"question": enriched_question})
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])
    except Exception as e:
        answer = f"Error during execution: {str(e)}"
        source_docs = []
    end_time = time.time()
    
    formatted_docs = format_evidence(source_docs)
    evidence_metrics = compute_retrieval_metrics(formatted_docs)
    
    b_metrics = evaluate_benchmark_case(case, answer, evidence_metrics, use_llm_grader)
    
    return {
        "case_id": case["id"],
        "question": question,
        "answer": answer,
        "execution_time_sec": round(end_time - start_time, 2),
        "source_count": evidence_metrics["unique_source_count"],
        "chunk_count": evidence_metrics["retrieved_chunk_count"],
        "evidence_metrics": evidence_metrics,
        "benchmark_metrics": b_metrics
    }

def run_benchmark_suite(cases: list, vectorstore, chain, use_llm_grader: bool = False) -> dict:
    results = []
    for case in cases:
        print(f"Running Case [{case['id']}] - Category: {case['category']}")
        res = run_single_benchmark_case(case, vectorstore, chain, use_llm_grader)
        results.append(res)
    return summarize_benchmark_results(results)

def summarize_benchmark_results(results: list) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r["benchmark_metrics"]["passed"])
    strong = sum(1 for r in results if r["benchmark_metrics"]["overall_label"] == "Strong")
    acceptable = sum(1 for r in results if r["benchmark_metrics"]["overall_label"] == "Acceptable")
    weak = sum(1 for r in results if "Weak" in r["benchmark_metrics"]["overall_label"])
    
    avg_structure = sum(r["benchmark_metrics"]["structure_score"] for r in results) / max(total, 1)
    avg_coverage = sum(r["benchmark_metrics"]["topic_coverage_score"] for r in results) / max(total, 1)
    
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cases": total,
        "passed": passed,
        "pass_rate_pct": int((passed / max(total, 1)) * 100),
        "label_counts": {
            "Strong": strong,
            "Acceptable": acceptable,
            "Weak": weak
        },
        "averages": {
            "structure_score": round(avg_structure, 1),
            "topic_coverage_score": round(avg_coverage, 1)
        },
        "case_results": results
    }
    
    save_benchmark_results(summary)
    return summary
