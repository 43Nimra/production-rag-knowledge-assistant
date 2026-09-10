"""Embedding provider factory (NFR-14: swappable via config, no
application code change)."""

from config import get_settings
from src.embeddings.base import EmbedderBase, EmbedderError

__all__ = ["EmbedderBase", "EmbedderError", "get_embedder"]


def get_embedder() -> EmbedderBase:
    settings = get_settings()
    if settings.embedder_provider == "openai":
        from src.embeddings.openai import OpenAIEmbedder

        return OpenAIEmbedder()

    # ADR-003 documents a local/sentence-transformers alternative
    # (EMBEDDER_PROVIDER=local) as an extension point for data-privacy
    # constraints, but it is not part of the approved Phase 2 scope — no
    # LocalEmbedder implementation exists yet. Failing clearly here (rather
    # than silently falling back to OpenAI) surfaces a misconfiguration
    # immediately instead of masking it.
    raise ValueError(
        f"Unsupported EMBEDDER_PROVIDER={settings.embedder_provider!r}. "
        "Only 'openai' is implemented; 'local' is a documented future "
        "extension point (ADR-003), not yet built."
    )
