from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin
import re

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


BASE_ANNOUNCEMENTS_URL = "https://www.cse.lk/announcements"

CATEGORY_URLS = {
    "All": BASE_ANNOUNCEMENTS_URL,
    "Corporate Disclosure": f"{BASE_ANNOUNCEMENTS_URL}?category=CORPORATE+DISCLOSURE",
    "Rights Issue": f"{BASE_ANNOUNCEMENTS_URL}?category=RIGHTS+ISSUE",
    "Dealings by Directors": f"{BASE_ANNOUNCEMENTS_URL}?category=DEALINGS+BY+DIRECTORS",
}

DATE_REGEX = re.compile(r"\b\d{2}[-/]\d{2}[-/]\d{4}\b")


class CSEAnnouncementsClient:
    def __init__(self, timeout: int = 20, debug_dir: str | Path = "data/debug"):
        self.timeout = timeout
        self.debug_dir = Path(debug_dir)

    # -------------------------
    # Helpers
    # -------------------------
    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _extract_date(self, text: str) -> str:
        match = DATE_REGEX.search(text or "")
        return match.group(0) if match else ""

    def _normalize_lines(self, text: str) -> list[str]:
        lines = [self._clean_text(line) for line in (text or "").splitlines()]
        lines = [line for line in lines if line]

        noise = {
            "HOME",
            "ANNOUNCEMENTS",
            "VIEW ALL",
            "SEARCH",
            "FILTER",
            "DOWNLOAD",
            "PDF",
            "READ MORE",
            "OPEN PDF",
        }

        normalized = []
        for line in lines:
            if line.upper() in noise:
                continue
            normalized.append(line)

        return normalized

    # -------------------------
    # Browser-rendered fetch
    # -------------------------
    def _render_page(self, url: str) -> tuple[str, str]:
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 2200},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=self.timeout * 1000)
            except PlaywrightTimeoutError:
                pass

            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except PlaywrightTimeoutError:
                pass

            # try waiting for visible announcement-ish text
            waited = False
            for selector in [
                "text=View Details",
                "text=Corporate Disclosure",
                "text=Rights Issue",
                "text=Dealings by Directors",
            ]:
                try:
                    page.locator(selector).first.wait_for(timeout=5000)
                    waited = True
                    break
                except Exception:
                    continue

            if not waited:
                page.wait_for_timeout(2000)

            # scroll a bit to trigger lazy rendering
            for y in [500, 1200, 1800, 0]:
                try:
                    page.evaluate(f"window.scrollTo(0, {y})")
                    page.wait_for_timeout(600)
                except Exception:
                    pass

            html = page.content()

            try:
                body_text = page.locator("body").inner_text(timeout=3000)
            except Exception:
                body_text = ""

            # debug artifacts
            (self.debug_dir / "last_cse_announcements_rendered.html").write_text(
                html, encoding="utf-8"
            )
            (self.debug_dir / "last_cse_announcements_visible_text.txt").write_text(
                body_text, encoding="utf-8"
            )
            try:
                page.screenshot(path=str(self.debug_dir / "last_cse_announcements_page.png"), full_page=True)
            except Exception:
                pass

            context.close()
            browser.close()

            return html, body_text

    # -------------------------
    # Link extraction
    # -------------------------
    def _extract_links_from_html(self, html: str, source_url: str) -> tuple[list[str], list[str]]:
        soup = BeautifulSoup(html, "html.parser")

        detail_links = []
        pdf_links = []

        for a in soup.find_all("a", href=True):
            href = urljoin(source_url, a.get("href", "").strip())
            text = self._clean_text(a.get_text(" ", strip=True)).upper()
            href_lower = href.lower()

            if ("VIEW DETAILS" in text or "READ MORE" in text) and href not in detail_links:
                detail_links.append(href)

            if href_lower.endswith(".pdf") and href not in pdf_links:
                pdf_links.append(href)

        return detail_links, pdf_links

    # -------------------------
    # Visible-text parsing
    # -------------------------
    def _build_blocks_from_view_details(self, lines: list[str]) -> list[list[str]]:
        blocks = []

        for i, line in enumerate(lines):
            if "VIEW DETAILS" in line.upper():
                start = max(0, i - 6)
                window = lines[start : i + 1]
                if window:
                    blocks.append(window)

        # dedupe
        deduped = []
        seen = set()
        for block in blocks:
            key = " | ".join(block)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(block)

        return deduped

    def _build_blocks_from_dates(self, lines: list[str]) -> list[list[str]]:
        blocks = []
        current = []

        for line in lines:
            if DATE_REGEX.search(line):
                if current:
                    blocks.append(current)
                current = [line]
            else:
                if current:
                    current.append(line)

        if current:
            blocks.append(current)

        # keep only meaningful ones
        filtered = []
        for block in blocks:
            text = " ".join(block)
            if len(text) >= 40:
                filtered.append(block)

        return filtered

    def _parse_block(self, block: list[str], category: str) -> dict | None:
        if not block:
            return None

        block_text = " ".join(block)
        announcement_date = self._extract_date(block_text)

        cleaned = []
        for line in block:
            upper = line.upper()
            if "VIEW DETAILS" in upper:
                continue
            if DATE_REGEX.search(line):
                continue
            if line in cleaned:
                continue
            cleaned.append(line)

        if not cleaned:
            return None

        company_name = cleaned[0]
        announcement_title = cleaned[1] if len(cleaned) > 1 else cleaned[0]

        return {
            "announcement_date": announcement_date,
            "company_name": company_name,
            "announcement_title": announcement_title,
            "category": category,
            "raw_block_text": block_text,
        }

    def _parse_from_visible_text(
        self,
        body_text: str,
        category: str,
        detail_links: list[str],
        pdf_links: list[str],
        source_url: str,
    ) -> pd.DataFrame:
        lines = self._normalize_lines(body_text)

        # first try grouping around "View Details"
        blocks = self._build_blocks_from_view_details(lines)

        # fallback to date-based grouping
        if not blocks:
            blocks = self._build_blocks_from_dates(lines)

        rows = []
        for block in blocks:
            parsed = self._parse_block(block, category)
            if parsed:
                rows.append(parsed)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "announcement_date",
                    "company_name",
                    "announcement_title",
                    "category",
                    "detail_url",
                    "pdf_url",
                    "source_page",
                    "raw_block_text",
                ]
            )

        df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

        df["detail_url"] = ""
        df["pdf_url"] = ""
        df["source_page"] = source_url

        for i in range(min(len(df), len(detail_links))):
            df.loc[i, "detail_url"] = detail_links[i]

        for i in range(min(len(df), len(pdf_links))):
            df.loc[i, "pdf_url"] = pdf_links[i]

        return df

    # -------------------------
    # Public API
    # -------------------------
    def fetch_announcements(self, category: str = "All") -> pd.DataFrame:
        source_url = CATEGORY_URLS.get(category, BASE_ANNOUNCEMENTS_URL)

        html, body_text = self._render_page(source_url)
        detail_links, pdf_links = self._extract_links_from_html(html, source_url)

        df = self._parse_from_visible_text(
            body_text=body_text,
            category=category,
            detail_links=detail_links,
            pdf_links=pdf_links,
            source_url=source_url,
        )

        if df.empty:
            return df

        df["announcement_date_parsed"] = pd.to_datetime(
            df["announcement_date"],
            errors="coerce",
            dayfirst=True,
        )

        df = df.sort_values(
            by="announcement_date_parsed",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

        return df