import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _make_chunk_id(source: str, text: str) -> str:
    raw = f"{source}::{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        text = chunk.page_content.strip()

        chunk.metadata["chunk_id"] = _make_chunk_id(source, text)
        chunk.metadata["chunk_index"] = idx

    return chunks