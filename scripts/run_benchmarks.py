import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vectorstore import load_vectorstore
from src.rag_chain import build_qa_chain
from src.benchmark_runner import load_benchmark_cases, run_benchmark_suite

def main():
    print("Loading Vectorstore...")
    vectorstore = load_vectorstore()
    
    print("Building QA Chain...")
    chain = build_qa_chain(vectorstore)
    
    cases_path = "data/benchmarks/cse_eval_set.json"
    print(f"Loading benchmarks from {cases_path}...")
    try:
        cases = load_benchmark_cases(cases_path)
    except FileNotFoundError:
        print(f"Error: Could not find benchmark file at {cases_path}. Are you in the project root?")
        sys.exit(1)
        
    print(f"Starting execution of {len(cases)} cases... (This may take several minutes)")
    summary = run_benchmark_suite(cases, vectorstore, chain, use_llm_grader=True)
    
    print("\n" + "="*40)
    print("Benchmark Execution Complete")
    print("="*40)
    print(f"Total Cases: {summary['total_cases']}")
    print(f"Pass Rate:   {summary['pass_rate_pct']}%")
    print(f"Strong:      {summary['label_counts']['Strong']}")
    print(f"Acceptable:  {summary['label_counts']['Acceptable']}")
    print(f"Weak:        {summary['label_counts']['Weak']}")
    print("="*40)
    print("Results saved successfully to data/cache/benchmarks/latest_results.json")

if __name__ == "__main__":
    main()
