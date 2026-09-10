"""PDF parsing via pdfplumber (SRS FR-02; ADR chosen library)."""

from pathlib import Path

import pdfplumber

from src.ingestion.parsers.base import (
    PageText,
    ParserBase,
    ParserError,
    ParseResult,
    check_min_text,
)


class PdfParser(ParserBase):
    def parse(self, file_path: Path) -> ParseResult:
        try:
            pages: list[PageText] = []
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    pages.append(PageText(page_number=i, text=text))
        except ParserError:
            raise
        except Exception as exc:
            # pdfplumber/pypdfium raise a variety of exception types for
            # corrupt or malformed PDFs; all are treated as one failure
            # mode here (ARCHITECTURE.md §5.2 "Malformed file content").
            raise ParserError("corrupt_file", f"Failed to open PDF: {exc}") from exc

        result = ParseResult(pages=pages)
        # A scanned PDF with no text layer is the documented PARSE_FAILED
        # case (ARCHITECTURE.md §5.2: "Scanned PDF (no text layer)").
        check_min_text(result.full_text)
        return result
