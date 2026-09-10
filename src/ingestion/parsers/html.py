"""HTML parsing via BeautifulSoup4 + lxml (SRS FR-02)."""

from pathlib import Path

from bs4 import BeautifulSoup

from src.ingestion.parsers.base import (
    PageText,
    ParserBase,
    ParserError,
    ParseResult,
    check_min_text,
    normalize_whitespace,
)


class HtmlParser(ParserBase):
    def parse(self, file_path: Path) -> ParseResult:
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ParserError("corrupt_file", f"Failed to read HTML file: {exc}") from exc

        soup = BeautifulSoup(raw, "lxml")
        # Strip non-content tags before extracting text (script/style text
        # is not document content and would pollute chunks/embeddings).
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = normalize_whitespace(soup.get_text(separator="\n"))
        # No page concept for HTML (ARCHITECTURE.md: page_number NULL for
        # formats without pages).
        result = ParseResult(pages=[PageText(page_number=None, text=text)])
        check_min_text(result.full_text)
        return result
