"""Test doubles shared across integration tests.

MockEmbedder implements EmbedderBase with no network calls (ADR-003's
testing philosophy: "MockEmbedder ... implement[s] the same interface.
Unit tests run without any API calls."). It returns real 1536-dimension
vectors — matching chunks.embedding's VECTOR(1536) column — so ingestion
integration tests exercise a real pgvector insert, not just a mocked one.
"""

from src.embeddings.base import EmbedderBase

EMBEDDING_DIM = 1536


class MockEmbedder(EmbedderBase):
    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    @staticmethod
    def _vector_for(text: str) -> list[float]:
        seed = sum(text.encode()) % 1000
        return [((seed + i) % 100) / 100 for i in range(EMBEDDING_DIM)]


class FailingEmbedder(EmbedderBase):
    """Always raises — used to test the PENDING-on-embedding-failure path
    (ARCHITECTURE.md §5.2)."""

    def embed_query(self, text: str) -> list[float]:
        raise self._error()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise self._error()

    @staticmethod
    def _error() -> Exception:
        from src.embeddings.base import EmbedderError

        return EmbedderError("simulated persistent embedding failure")
