"""SentenceAwareChunker unit tests (NFR-19).

Uses a simple word-count token counter injected via the constructor
(see src/ingestion/chunker.py's module docstring) so these tests are
hermetic and fast — no tiktoken network dependency, and no coupling to
its exact BPE counts, which aren't what's under test here anyway.
"""

from src.ingestion.chunker import SentenceAwareChunker
from src.ingestion.parsers.base import PageText, ParseResult


def word_count(text: str) -> int:
    return len(text.split())


def make_result(*page_texts: str) -> ParseResult:
    return ParseResult(
        pages=[PageText(page_number=i + 1, text=t) for i, t in enumerate(page_texts)]
    )


class TestEmptyInput:
    def test_empty_text_returns_no_chunks(self) -> None:
        chunker = SentenceAwareChunker(chunk_size=50, overlap=10, token_counter=word_count)
        result = chunker.chunk(make_result(""))
        assert result == []

    def test_whitespace_only_text_returns_no_chunks(self) -> None:
        chunker = SentenceAwareChunker(chunk_size=50, overlap=10, token_counter=word_count)
        result = chunker.chunk(make_result("   \n\n   "))
        assert result == []


class TestSingleSentence:
    def test_single_short_sentence_is_one_chunk(self) -> None:
        chunker = SentenceAwareChunker(chunk_size=50, overlap=10, token_counter=word_count)
        result = chunker.chunk(make_result("This is one sentence."))
        assert len(result) == 1
        assert result[0].text == "This is one sentence."
        assert result[0].chunk_index == 0
        assert result[0].start_char == 0

    def test_sentence_larger_than_chunk_size_is_still_its_own_chunk(self) -> None:
        # A single sentence can't be split further while staying
        # sentence-aware (ADR-004): it becomes an over-budget chunk of one.
        long_sentence = " ".join(f"word{i}" for i in range(100)) + "."
        chunker = SentenceAwareChunker(chunk_size=10, overlap=2, token_counter=word_count)
        result = chunker.chunk(make_result(long_sentence))
        assert len(result) == 1
        assert result[0].token_count == 100


class TestMultiParagraph:
    def test_multiple_sentences_split_at_token_budget(self) -> None:
        # 6 sentences, 10 words ("tokens") each = 60 tokens total.
        sentences = " ".join(
            f"Sentence number {i} has exactly ten words in it right now." for i in range(1, 7)
        )
        chunker = SentenceAwareChunker(chunk_size=25, overlap=0, token_counter=word_count)
        result = chunker.chunk(make_result(sentences))

        assert len(result) > 1
        # Every produced chunk must respect the requested word-count budget
        # (no chunk should exceed it when built from multiple sentences).
        for c in result:
            assert c.token_count <= 25 or c.text.count(".") <= 1

    def test_chunk_indices_are_sequential(self) -> None:
        sentences = " ".join(f"Sentence {i} here now." for i in range(1, 10))
        chunker = SentenceAwareChunker(chunk_size=10, overlap=2, token_counter=word_count)
        result = chunker.chunk(make_result(sentences))
        assert [c.chunk_index for c in result] == list(range(len(result)))

    def test_page_number_attributed_per_chunk(self) -> None:
        page1 = " ".join(f"Page one sentence {i} content here now." for i in range(1, 5))
        page2 = " ".join(f"Page two sentence {i} content here now." for i in range(1, 5))
        chunker = SentenceAwareChunker(chunk_size=15, overlap=0, token_counter=word_count)
        result = chunker.chunk(make_result(page1, page2))

        page_numbers = {c.page_number for c in result}
        assert page_numbers == {1, 2}
        # Chunks should appear in document order: all page-1 chunks before
        # any page-2 chunk.
        pages_in_order = [c.page_number for c in result]
        assert pages_in_order == sorted(pages_in_order)


class TestOverlapBoundary:
    def test_overlap_carries_trailing_sentences_into_next_chunk(self) -> None:
        sentences = [f"This is sentence number {i} in the document." for i in range(1, 8)]
        text = " ".join(sentences)
        # 8-word sentences; overlap=8 should carry ~1 trailing sentence.
        chunker = SentenceAwareChunker(chunk_size=16, overlap=8, token_counter=word_count)
        result = chunker.chunk(make_result(text))

        assert len(result) >= 2
        # The last sentence of chunk N should reappear as the first
        # sentence of chunk N+1 (the overlap).
        first_chunk_last_sentence = result[0].text.split(".")[-2].strip() + "."
        assert first_chunk_last_sentence in result[1].text

    def test_zero_overlap_produces_no_shared_sentences(self) -> None:
        sentences = [f"Sentence {i} content words here now today." for i in range(1, 8)]
        text = " ".join(sentences)
        chunker = SentenceAwareChunker(chunk_size=14, overlap=0, token_counter=word_count)
        result = chunker.chunk(make_result(text))

        assert len(result) >= 2
        combined_text = " ".join(c.text for c in result)
        # No sentence's word-count should be double-counted when overlap=0.
        assert combined_text.count("Sentence 1 ") == 1


class TestConfigValidation:
    def test_overlap_greater_than_chunk_size_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="overlap"):
            SentenceAwareChunker(chunk_size=10, overlap=20)

    def test_negative_chunk_size_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="chunk_size"):
            SentenceAwareChunker(chunk_size=-5, overlap=0)
