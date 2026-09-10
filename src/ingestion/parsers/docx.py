"""DOCX parsing via python-docx (SRS FR-02)."""

from pathlib import Path

from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError

from src.ingestion.parsers.base import (
    PageText,
    ParserBase,
    ParserError,
    ParseResult,
    check_min_text,
    normalize_whitespace,
)


class DocxParser(ParserBase):
    def parse(self, file_path: Path) -> ParseResult:
        try:
            doc = DocxDocument(str(file_path))
        except PackageNotFoundError as exc:
            # Covers both corrupt files and password-protected DOCX
            # (ARCHITECTURE.md §5.2: "Password-protected DOCX").
            raise ParserError(
                "corrupt_file", "Not a valid DOCX file (corrupt or password-protected)"
            ) from exc
        except Exception as exc:
            raise ParserError("corrupt_file", f"Failed to open DOCX: {exc}") from exc

        text = normalize_whitespace("\n".join(p.text for p in doc.paragraphs))
        # DOCX has no page concept at the parser level (page breaks are a
        # rendering detail, not reliably recoverable from python-docx), so
        # the whole document is one PageText with page_number=None,
        # matching ARCHITECTURE.md's `page_number NULL for formats without
        # pages`.
        result = ParseResult(pages=[PageText(page_number=None, text=text)])
        check_min_text(result.full_text)
        return result
