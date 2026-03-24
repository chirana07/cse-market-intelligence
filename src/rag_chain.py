from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from src.config import CHAT_MODEL, OLLAMA_BASE_URL


ANALYST_PROMPT = PromptTemplate(
    input_variables=["summaries", "question"],
    template="""You are an elite equity research analyst assistant analyzing Sri Lankan (CSE) stocks.

Your responses MUST strictly follow this exact 5-part structure:

1. Direct Answer
- Provide a clear, concise answer to the question in 2-4 sentences.

2. Why It Matters
- Explain the business, financial, strategic, or valuation relevance of the evidence.

3. Key Evidence
- List the most critical factual developments using bullet points. Include numbers and dates if available.

4. Risks / Unknowns
- Identify what is explicitly missing from the evidence, what could go wrong, or what remains uncertain. Say clearly when evidence is insufficient.

5. Follow-up Questions
- Provide 2-3 logical follow-up questions for the user to consider next.

CRITICAL RULES:
- Use ONLY the provided context to answer. Do not invent facts, numbers, or external events.
- If the evidence is incomplete, explicitly state "Based on the retrieved context, there is not enough information to..."
- If the context contains nothing relevant, simply say "I do not have sufficient retrieved context to answer this question."
- Always prioritize CSE-specific or company-specific evidence when present.

Context:
{summaries}

Question:
{question}

Structured Analyst Output:
"""
)


def build_qa_chain(vectorstore, domain_filter=None, source_filter=None):
    llm = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    search_kwargs = {
        "k": 6,
        "fetch_k": 30,
    }

    metadata_filter = {}
    if domain_filter and domain_filter != "All":
        metadata_filter["domain"] = domain_filter
    if source_filter and source_filter != "All":
        metadata_filter["source"] = source_filter

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )
    except Exception:
        retriever = base_retriever

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": ANALYST_PROMPT},
    )
    return chain