"""Embedding provider abstraction (ADR-003).

Provider selection is hidden behind this ABC so the ingestion pipeline
and (in a later phase) the query pipeline never call an OpenAI/local SDK
directly — swapping providers is a config change, not an application
code change (NFR-14).
"""

from abc import ABC, abstractmethod


class EmbedderError(Exception):
    """Raised on persistent embedding failure (after retries are
    exhausted) or any non-retryable embedding provider error."""


class EmbedderBase(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Returns a vector."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks. Returns a list of vectors, in
        the same order as `texts`. Implementations must handle batching
        internally (ADR-003 contract)."""
