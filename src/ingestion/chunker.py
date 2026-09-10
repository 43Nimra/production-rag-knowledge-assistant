"""Sentence-aware chunking (ADR-004).

Splits text at sentence boundaries (NLTK) rather than at arbitrary
character positions, then groups sentences into chunks up to a token
budget with a sliding-window overlap, so a concept sitting at a chunk
boundary is still fully captured in at least one chunk.

Token counting is injectable (`token_counter` constructor parameter)
rather than hardwired to a module-level tiktoken call. Production code
gets an accurate `tiktoken` (cl100k_base — the encoding
`text-embedding-3-small` uses, per ADR-003) count by default; tests can
inject a cheap counter to stay hermetic and fast, the same testability
principle NFR-22 applies to embedding/LLM calls, extended here to token
counting since tiktoken's rank data is a first-use network download in
any environment that hasn't already cached it (this is a general
characteristic of the library, not specific to any one deployment
target).
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from nltk.tokenize import sent_tokenize

from src.ingestion.parsers.base import ParseResult

TokenCounter = Callable[[str], int]


class ChunkerConfigError(Exception):
    """Raised when a chunker dependency (NLTK sentence data) is missing."""


@lru_cache
def _tiktoken_encoding() -> "object":
    import tiktoken

    return tiktoken.encoding_for_model("text-embedding-3-small")


def default_token_counter(text: str) -> int:
    """Real token count via tiktoken (cl100k_base). This is the counter
    IngestionPipeline uses in production."""
    encoding = _tiktoken_encoding()
    return len(encoding.encode(text))  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ChunkResult:
    text: str
    chunk_index: int
    start_char: int
    token_count: int
    page_number: int | None


@dataclass(frozen=True)
class _Sentence:
    text: str
    start_char: int
    page_number: int | None
    token_count: int


def _sentence_spans(text: str) -> list[tuple[str, int]]:
    """Sentence-tokenizes text and returns each sentence with its start
    char offset in text, via a forward-scanning search rather than
    relying on NLTK's internal span API (whose on-disk pickle format
    differs between punkt/punkt_tab NLTK releases)."""
    try:
        sentences = sent_tokenize(text)
    except LookupError as exc:
        raise ChunkerConfigError(
            "NLTK sentence tokenizer data ('punkt_tab') is not installed. "
            "This must be downloaded once (baked into the Docker image at "
            "build time, or via `python -m nltk.downloader punkt_tab` in "
            "local dev) — see docker/Dockerfile."
        ) from exc

    spans: list[tuple[str, int]] = []
    cursor = 0
    for sentence in sentences:
        idx = text.find(sentence, cursor)
        if idx == -1:
            # Defensive fallback: should not happen given sent_tokenize's
            # contract of returning verbatim substrings, but never silently
            # drop a sentence if it does.
            idx = cursor
        spans.append((sentence, idx))
        cursor = idx + len(sentence)
    return spans


def _page_for_offset(page_boundaries: list[tuple[int, int | None]], offset: int) -> int | None:
    """page_boundaries is a list of (start_char_in_full_text, page_number)
    in ascending order; returns the page_number whose range contains
    offset."""
    page_number: int | None = page_boundaries[0][1] if page_boundaries else None
    for start, page in page_boundaries:
        if start > offset:
            break
        page_number = page
    return page_number


class SentenceAwareChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        token_counter: TokenCounter = default_token_counter,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._count_tokens = token_counter

    def chunk(self, parse_result: ParseResult) -> list[ChunkResult]:
        full_text = parse_result.full_text
        if not full_text.strip():
            return []

        # Char offset (within full_text) where each page begins, derived
        # the same way ParseResult.full_text joins pages ("\n\n").
        page_boundaries: list[tuple[int, int | None]] = []
        cursor = 0
        for page in parse_result.pages:
            if page.text:
                page_boundaries.append((cursor, page.page_number))
                cursor += len(page.text) + 2  # + the "\n\n" join separator

        sentences: list[_Sentence] = []
        for text, offset in _sentence_spans(full_text):
            if not text.strip():
                continue
            sentences.append(
                _Sentence(
                    text=text,
                    start_char=offset,
                    page_number=_page_for_offset(page_boundaries, offset),
                    token_count=self._count_tokens(text),
                )
            )

        return self._group_into_chunks(sentences)

    def _group_into_chunks(self, sentences: list[_Sentence]) -> list[ChunkResult]:
        chunks: list[ChunkResult] = []
        current: list[_Sentence] = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            projected = current_tokens + sentence.token_count
            if current and projected > self.chunk_size:
                chunks.append(self._finalize(current, chunk_index))
                chunk_index += 1
                current = self._carry_over(current)
                current_tokens = sum(s.token_count for s in current)

            current.append(sentence)
            current_tokens += sentence.token_count

        if current:
            chunks.append(self._finalize(current, chunk_index))

        return chunks

    def _carry_over(self, sentences: list[_Sentence]) -> list[_Sentence]:
        """Returns the trailing sentences of the just-finalized chunk that
        together total up to `overlap` tokens, seeding the next chunk so a
        boundary concept is captured in both chunks (ADR-004)."""
        carried: list[_Sentence] = []
        tokens = 0
        for sentence in reversed(sentences):
            if tokens + sentence.token_count > self.overlap and carried:
                break
            carried.insert(0, sentence)
            tokens += sentence.token_count
        return carried

    @staticmethod
    def _finalize(sentences: list[_Sentence], chunk_index: int) -> ChunkResult:
        text = " ".join(s.text for s in sentences)
        return ChunkResult(
            text=text,
            chunk_index=chunk_index,
            start_char=sentences[0].start_char,
            token_count=sum(s.token_count for s in sentences),
            page_number=sentences[0].page_number,
        )
