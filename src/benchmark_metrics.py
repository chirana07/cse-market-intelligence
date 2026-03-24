from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from src.config import CHAT_MODEL, OLLAMA_BASE_URL

def evaluate_structure(answer: str) -> dict:
    sections = [
        "1. Direct Answer",
        "2. Why It Matters",
        "3. Key Evidence",
        "4. Risks / Unknowns",
        "5. Follow-up Questions"
    ]
    found = [s for s in sections if s.lower() in answer.lower()]
    score = int((len(found) / len(sections)) * 100)
    return {"score": score, "missing_sections": [s for s in sections if s not in found]}

def evaluate_topic_coverage(answer: str, topics: List[str]) -> dict:
    if not topics:
        return {"score": 100, "missing_topics": []}
    found = [t for t in topics if t.lower() in answer.lower()]
    score = int((len(found) / len(topics)) * 100)
    return {"score": score, "missing_topics": [t for t in topics if t not in found]}

def compute_groundedness(evidence_metrics: dict, structure_score: int, coverage_score: int) -> dict:
    e_score = evidence_metrics.get("evidence_score", 0)
    
    # High risk if topics matched but there is zero or minimal evidence
    hallucination_risk = False
    if e_score <= 20 and coverage_score > 50:
        hallucination_risk = True

    proxy_score = int(0.5 * e_score + 0.3 * coverage_score + 0.2 * structure_score)
    
    if hallucination_risk:
        overall_label = "Weak (High Hallucination Risk)"
    elif proxy_score >= 75:
        overall_label = "Strong"
    elif proxy_score >= 40:
        overall_label = "Acceptable"
    else:
        overall_label = "Weak"
        
    return {
        "proxy_score": proxy_score,
        "hallucination_risk_flag": hallucination_risk,
        "overall_label": overall_label
    }

def grade_alignment_with_llm(question: str, answer: str, expected_signals: List[str]) -> str:
    if not expected_signals or not answer.strip():
        return "Not Graded"
        
    llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    signals_str = "\n".join([f"- {s}" for s in expected_signals])
    prompt = PromptTemplate.from_template(
        "You are an evaluator checking if an Answer hits the Expected Signals for a specific Question.\n"
        "Output ONLY ONE of these exact labels: 'Well-aligned', 'Partially aligned', or 'Weakly aligned'.\n"
        "Do not explain your reasoning.\n\n"
        "Question: {question}\n\nExpected Signals:\n{signals}\n\nAnswer:\n{answer}"
    )
    
    try:
        res = (prompt | llm).invoke({"question": question, "signals": signals_str, "answer": answer})
        content = res.content.strip() if hasattr(res, "content") else str(res).strip()
        for valid in ["Well-aligned", "Partially aligned", "Weakly aligned"]:
            if valid.lower() in content.lower():
                return valid
        return "Grading failed"
    except Exception:
        return "Grading error"

def evaluate_benchmark_case(case: dict, answer: str, evidence_metrics: dict, use_llm_grader: bool = False) -> dict:
    structure_eval = evaluate_structure(answer)
    coverage_eval = evaluate_topic_coverage(answer, case.get("must_include_topics", []))
    groundedness_eval = compute_groundedness(evidence_metrics, structure_eval["score"], coverage_eval["score"])
    
    llm_alignment = "Not Graded"
    if use_llm_grader:
        llm_alignment = grade_alignment_with_llm(case.get("question", ""), answer, case.get("expected_signals", []))
        
    passed = groundedness_eval["overall_label"] in ["Strong", "Acceptable"] and structure_eval["score"] >= 80
    
    return {
        "passed": passed,
        "structure_score": structure_eval["score"],
        "topic_coverage_score": coverage_eval["score"],
        "groundedness_proxy_score": groundedness_eval["proxy_score"],
        "hallucination_risk_flag": groundedness_eval["hallucination_risk_flag"],
        "overall_label": groundedness_eval["overall_label"],
        "llm_alignment_grade": llm_alignment,
        "missing_sections": structure_eval["missing_sections"],
        "missing_topics": coverage_eval["missing_topics"]
    }
