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

    if settings.embedder_provider == "local":
        from src.embeddings.local import LocalEmbedder

        return LocalEmbedder()

    raise ValueError(
        f"Unsupported EMBEDDER_PROVIDER={settings.embedder_provider!r}. "
        "Supported values: 'openai', 'local' (ADR-003)."
    )
