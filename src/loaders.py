import re

from langchain_community.document_loaders import UnstructuredURLLoader, WebBaseLoader

from src.enrichment import enrich_document_metadata
from src.guardrails import validate_ingest_url


URL_REGEX = re.compile(r"https?://[^\s<>\"]+")


def is_valid_url(url: str) -> bool:
    return validate_ingest_url(url).allowed


def _clean_extracted_url(url: str) -> str:
    return url.rstrip(".,);:]}>\"'")


def load_single_url(url: str):
    url = url.strip()
    if not validate_ingest_url(url).allowed:
        return []

    try:
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        if docs and any((doc.page_content or "").strip() for doc in docs):
            for doc in docs:
                enrich_document_metadata(doc, url)
            return docs
    except Exception as e:
        print(f"[UnstructuredURLLoader failed] {url} -> {e}")

    try:
        loader = WebBaseLoader(web_paths=[url])
        docs = loader.load()
        if docs and any((doc.page_content or "").strip() for doc in docs):
            for doc in docs:
                enrich_document_metadata(doc, url)
            return docs
    except Exception as e:
        print(f"[WebBaseLoader failed] {url} -> {e}")

    return []


def load_urls(urls: list[str]):
    cleaned_urls = list(dict.fromkeys([
        u.strip() for u in urls if u.strip() and validate_ingest_url(u).allowed
    ]))

    all_docs = []
    failed_urls = []

    for url in cleaned_urls:
        docs = load_single_url(url)
        if docs:
            all_docs.extend(docs)
        else:
            failed_urls.append(url)

    return all_docs, failed_urls


def parse_uploaded_txt_file(uploaded_file):
    """
    Supports both:
    1. One URL per line
    2. Mixed text files that contain URLs inside paragraphs
    Returns:
        valid_urls, invalid_lines
    """
    if uploaded_file is None:
        return [], []

    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    raw_lines = [line.strip() for line in content.splitlines() if line.strip()]

    extracted_urls = []
    for match in URL_REGEX.findall(content):
        candidate = _clean_extracted_url(match)
        if validate_ingest_url(candidate).allowed:
            extracted_urls.append(candidate)

    valid_urls = list(dict.fromkeys(extracted_urls))

    invalid_lines = []
    for line in raw_lines:
        if not URL_REGEX.search(line):
            invalid_lines.append(line[:250])

    return valid_urls, invalid_lines
