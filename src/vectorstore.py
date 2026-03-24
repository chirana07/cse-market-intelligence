import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from src.config import EMBED_MODEL, OLLAMA_BASE_URL


def get_embeddings():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def load_vectorstore(save_path: str):
    if not os.path.exists(save_path):
        return None

    try:
        embeddings = get_embeddings()
        return FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None


def _get_existing_chunk_ids(vectorstore) -> set[str]:
    chunk_ids = set()

    if vectorstore is None:
        return chunk_ids

    docstore = getattr(vectorstore, "docstore", None)
    docs_dict = getattr(docstore, "_dict", {}) if docstore else {}

    for doc in docs_dict.values():
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            chunk_ids.add(chunk_id)

    return chunk_ids


def _coerce_label_list(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [v.strip() for v in value.split("|") if v.strip()]
    return []


def get_vectorstore_stats(save_path: str) -> dict:
    vectorstore = load_vectorstore(save_path)
    if vectorstore is None:
        return {
            "exists": False,
            "chunk_count": 0,
            "source_count": 0,
            "domain_count": 0,
            "domains": [],
            "tickers": [],
            "event_tags": [],
            "documents": [],
        }

    docstore = getattr(vectorstore, "docstore", None)
    docs_dict = getattr(docstore, "_dict", {}) if docstore else {}

    documents_by_source = {}

    for doc in docs_dict.values():
        source = doc.metadata.get("source", "Unknown source")

        if source not in documents_by_source:
            documents_by_source[source] = {
                "title": doc.metadata.get("title", "Untitled"),
                "domain": doc.metadata.get("domain", "unknown"),
                "source": source,
                "ingested_at": doc.metadata.get("ingested_at", ""),
                "tickers": set(),
                "event_tags": set(),
            }

        documents_by_source[source]["tickers"].update(
            _coerce_label_list(doc.metadata.get("ticker_candidates", doc.metadata.get("ticker_candidates_str", "")))
        )
        documents_by_source[source]["event_tags"].update(
            _coerce_label_list(doc.metadata.get("event_tags", doc.metadata.get("event_tags_str", "")))
        )

    documents = []
    for item in documents_by_source.values():
        tickers = sorted(item["tickers"])
        event_tags = sorted(item["event_tags"])

        documents.append({
            "title": item["title"],
            "domain": item["domain"],
            "source": item["source"],
            "ingested_at": item["ingested_at"],
            "tickers": tickers,
            "tickers_str": " | ".join(tickers),
            "primary_ticker": tickers[0] if tickers else "Unknown",
            "event_tags": event_tags,
            "event_tags_str": " | ".join(event_tags),
            "primary_event": event_tags[0] if event_tags else "General Update",
        })

    documents = sorted(
        documents,
        key=lambda x: (x["domain"], x["title"], x["source"]),
    )

    domains = sorted({item["domain"] for item in documents})
    tickers = sorted({ticker for item in documents for ticker in item["tickers"]})
    event_tags = sorted({tag for item in documents for tag in item["event_tags"]})

    return {
        "exists": True,
        "chunk_count": len(docs_dict),
        "source_count": len(documents),
        "domain_count": len(domains),
        "domains": domains,
        "tickers": tickers,
        "event_tags": event_tags,
        "documents": documents,
    }


def ingest_chunks(chunks, save_path: str):
    os.makedirs(save_path, exist_ok=True)

    vectorstore = load_vectorstore(save_path)
    existing_chunk_ids = _get_existing_chunk_ids(vectorstore)

    new_chunks = []
    skipped_duplicates = 0

    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if chunk_id and chunk_id in existing_chunk_ids:
            skipped_duplicates += 1
            continue
        new_chunks.append(chunk)

    if vectorstore is None:
        if not new_chunks:
            return None, {
                "mode": "empty",
                "new_chunks": 0,
                "skipped_duplicates": skipped_duplicates,
            }

        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(new_chunks, embeddings)
        vectorstore.save_local(save_path)

        return vectorstore, {
            "mode": "created",
            "new_chunks": len(new_chunks),
            "skipped_duplicates": skipped_duplicates,
        }

    if new_chunks:
        vectorstore.add_documents(new_chunks)
        vectorstore.save_local(save_path)

    return vectorstore, {
        "mode": "appended" if new_chunks else "no_new_chunks",
        "new_chunks": len(new_chunks),
        "skipped_duplicates": skipped_duplicates,
    }


def clear_vectorstore(save_path: str):
    if os.path.exists(save_path):
        shutil.rmtree(save_path)