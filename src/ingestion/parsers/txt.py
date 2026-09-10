"""Plain text parsing (SRS FR-02)."""

from pathlib import Path

from src.ingestion.parsers.base import (
    PageText,
    ParserBase,
    ParserError,
    ParseResult,
    check_min_text,
    normalize_whitespace,
)


class PlainTextParser(ParserBase):
    def parse(self, file_path: Path) -> ParseResult:
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ParserError("corrupt_file", f"Failed to read TXT file: {exc}") from exc

        text = normalize_whitespace(raw)
        result = ParseResult(pages=[PageText(page_number=None, text=text)])
        check_min_text(result.full_text)
        return result
