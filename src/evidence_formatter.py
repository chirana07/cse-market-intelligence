def format_evidence(docs):
    """
    Extracts and standardizes metadata from raw LangChain Document objects 
    for cleaner visualization in the Streamlit UI.
    """
    formatted = []
    
    if not docs:
        return formatted
        
    for doc in docs:
        metadata = getattr(doc, "metadata", {})
        content = getattr(doc, "page_content", "")
        
        has_title = bool(metadata.get("title"))
        has_source = bool(metadata.get("source"))
        
        formatted.append({
            "title": metadata.get("title", "Untitled Document"),
            "source_url": metadata.get("source", "Unknown Source"),
            "domain": metadata.get("domain", "Unknown Domain"),
            "snippet": content[:1000].strip() + ("..." if len(content) > 1000 else ""),
            "snippet_length": len(content),
            "tickers": metadata.get("ticker_candidates_str", ""),
            "events": metadata.get("event_tags_str", ""),
            "ingested_at": metadata.get("ingested_at", ""),
            "chunk_id": metadata.get("chunk_id", "N/A"),
            "has_title": has_title,
            "has_source": has_source,
        })
        
    return formatted
