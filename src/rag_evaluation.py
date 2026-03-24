from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from src.config import CHAT_MODEL, OLLAMA_BASE_URL

def compute_retrieval_metrics(formatted_docs: list) -> dict:
    if not formatted_docs:
        return {
            "retrieved_chunk_count": 0,
            "unique_source_count": 0,
            "unique_domain_count": 0,
            "evidence_score": 0,
            "retrieval_score": 0,
            "source_diversity_score": 0,
            "coverage_label": "Zero",
            "confidence_label": "Low",
            "gaps_or_warnings": ["No evidence retrieved."],
        }
        
    retrieved_chunk_count = len(formatted_docs)
    unique_sources = len(set(d["source_url"] for d in formatted_docs))
    unique_domains = len(set(d["domain"] for d in formatted_docs))
    
    # Calculate bounded sub-scores (Max 100)
    retrieval_score = min(100, int((retrieved_chunk_count / 5.0) * 100))
    diversity_score = min(100, int((unique_domains / 2.0) * 50 + (unique_sources / 3.0) * 50))
    
    # Blended score weighting diversity slightly higher
    evidence_score = int(0.4 * retrieval_score + 0.6 * diversity_score)
    
    # Gap checks
    warnings = []
    if unique_sources == 1 and retrieved_chunk_count > 1:
        warnings.append("Evidence comes from only one source.")
    if unique_domains == 1 and unique_sources > 1:
        warnings.append("Limited source diversity (all evidence mapped to a single domain).")
        
    short_chunks = sum(1 for d in formatted_docs if d.get("snippet_length", 0) < 100)
    if short_chunks > 0:
        warnings.append(f"{short_chunks} retrieved chunks are abnormally short or potentially weak.")
        
    missing_meta = sum(1 for d in formatted_docs if not d.get("has_title") or not d.get("has_source"))
    if missing_meta > 0:
        warnings.append("Metadata coverage is incomplete for some chunks.")
        
    if evidence_score >= 80:
        confidence = "High"
        coverage = "Strong"
    elif evidence_score >= 40:
        confidence = "Medium"
        coverage = "Moderate"
    else:
        confidence = "Low"
        coverage = "Limited"
        warnings.append("Overall retrieved evidence is considered weak or limited.")
        
    return {
        "retrieved_chunk_count": retrieved_chunk_count,
        "unique_source_count": unique_sources,
        "unique_domain_count": unique_domains,
        "retrieval_score": retrieval_score,
        "source_diversity_score": diversity_score,
        "evidence_score": evidence_score,
        "coverage_label": coverage,
        "confidence_label": confidence,
        "gaps_or_warnings": warnings,
    }


def grade_answer_support(question: str, answer: str, formatted_docs: list) -> str:
    """
    Lightweight, optional LLM check leveraging a single forward pass.
    """
    if not formatted_docs or not answer.strip():
        return "Unverifiable"
        
    llm = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )
    
    snippets = "\n\n".join([f"Source {i+1}:\n{doc.get('snippet', '')}" for i, doc in enumerate(formatted_docs)])
    
    prompt = PromptTemplate.from_template(
        "You are an evaluator grading whether a provided 'Answer' is supported by the context 'Evidence'.\n"
        "Read the Evidence. Read the Answer.\n"
        "Output ONLY ONE of these exactly: 'Well-supported', 'Partially supported', 'Weakly supported'.\n"
        "Do not explain your reasoning. Just output the label.\n\n"
        "Question: {question}\n\nEvidence:\n{snippets}\n\nAnswer:\n{answer}"
    )
    
    chain = prompt | llm
    try:
        res = chain.invoke({"question": question, "snippets": snippets, "answer": answer})
        content = res.content.strip() if hasattr(res, "content") else str(res).strip()
        
        # Clean any hallucinations to ensure tight category matching
        for valid in ["Well-supported", "Partially supported", "Weakly supported"]:
            if valid.lower() in content.lower():
                return valid
        return "Unverifiable"
    except Exception:
        return "Evaluation Failed"
