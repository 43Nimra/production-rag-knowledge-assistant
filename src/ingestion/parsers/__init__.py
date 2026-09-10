"""Parser dispatch by file extension.

Re-exports the public parser API so callers do `from
src.ingestion.parsers import get_parser, ParserError` rather than reaching
into individual submodules.
"""

from pathlib import Path

from src.ingestion.parsers.base import ParserBase, ParserError, ParseResult
from src.ingestion.parsers.docx import DocxParser
from src.ingestion.parsers.html import HtmlParser
from src.ingestion.parsers.pdf import PdfParser
from src.ingestion.parsers.txt import PlainTextParser

_PARSERS_BY_EXTENSION: dict[str, type[ParserBase]] = {
    ".pdf": PdfParser,
    ".docx": DocxParser,
    ".html": HtmlParser,
    ".htm": HtmlParser,
    ".txt": PlainTextParser,
}

SUPPORTED_EXTENSIONS = frozenset(_PARSERS_BY_EXTENSION)


def get_parser(filename: str) -> ParserBase:
    """Returns the parser for filename's extension.

    Raises ParserError("unsupported_format", ...) for anything outside
    the approved format set (SRS §6 constraint: PDF, DOCX, HTML, TXT only).
    """
    extension = Path(filename).suffix.lower()
    parser_cls = _PARSERS_BY_EXTENSION.get(extension)
    if parser_cls is None:
        raise ParserError(
            "unsupported_format",
            f"Unsupported file extension '{extension}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    return parser_cls()


__all__ = [
    "ParseResult",
    "ParserBase",
    "ParserError",
    "get_parser",
    "SUPPORTED_EXTENSIONS",
]
