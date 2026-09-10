"""Shared parser types.

IMPLEMENTATION_PLAN.md sketches `ParseResult(text, page_numbers)`. This
module implements the same intent — extracted text plus, where the format
has one, a page number per unit of text (FR-05) — as a list of per-page
segments (`PageText`) rather than a `dict[int, str]`. That gives the
chunker a straightforward way to attribute a chunk's `page_number` from
the char offset where it starts, and reads cleanly for formats with no
page concept (DOCX/HTML/TXT all produce a single `PageText` with
`page_number=None`). This is an internal data-structure choice, not a
change to the approved parser set, chunking strategy, or schema.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# Below this many non-whitespace characters, extracted text is treated as
# empty (ARCHITECTURE.md §5.2: "len(extracted_text) < MIN_TEXT_THRESHOLD").
MIN_TEXT_THRESHOLD = 20


@dataclass(frozen=True)
class PageText:
    """One page's worth of extracted text. page_number is None for formats
    without a page concept (DOCX, HTML, TXT)."""

    page_number: int | None
    text: str


@dataclass(frozen=True)
class ParseResult:
    pages: list[PageText]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


class ParserError(Exception):
    """Raised on any parse failure. `reason` is a short machine-readable
    code logged and surfaced in the API's PARSE_FAILED response
    (ARCHITECTURE.md §5.2), e.g. 'empty_extraction', 'corrupt_file'."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class ParserBase(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> ParseResult:
        """Extract text from file_path. Raises ParserError on failure —
        never returns an empty/near-empty ParseResult silently."""


def normalize_whitespace(text: str) -> str:
    """Collapses runs of horizontal whitespace and strips trailing
    whitespace per line, without touching paragraph structure (blank
    lines are preserved as sentence/paragraph boundaries for the
    chunker)."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(lines)


def check_min_text(text: str, reason: str = "empty_extraction") -> None:
    if len(text.strip()) < MIN_TEXT_THRESHOLD:
        raise ParserError(reason, "Extracted text is empty or below the minimum length")
