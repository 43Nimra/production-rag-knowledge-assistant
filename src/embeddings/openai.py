"""OpenAI embedding provider (ADR-003 default implementation).

Uses the synchronous OpenAI SDK — matching the sync `EmbedderBase`
contract from ADR-003. The async ingestion pipeline runs embedding calls
via `asyncio.to_thread` so they don't block the event loop; that's a
pipeline-level concern, not this class's.
"""

import time
from collections.abc import Callable

import openai

from config import get_settings
from src.embeddings.base import EmbedderBase, EmbedderError
from src.observability.logger import get_logger

logger = get_logger(__name__)

# NFR-26: "Embedding calls during ingestion SHALL be batched (max batch
# size: 100 chunks per API call)".
MAX_BATCH_SIZE = 100

# ARCHITECTURE.md §5.2: "Exponential backoff: 1s, 2s, 4s. After 3
# failures: ... return 503" (translated here to raising EmbedderError,
# which the ingestion pipeline maps to the documented failure handling).
_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


class OpenAIEmbedder(EmbedderBase):
    def __init__(
        self,
        client: openai.OpenAI | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        settings = get_settings()
        self._model = settings.embed_model
        # Injectable for unit test isolation (NFR-22): defaults to a real
        # client built from configured settings, but tests can pass a
        # fake implementing the same `.embeddings.create(...)` surface
        # without needing a live API key or network access.
        self._client = client or openai.OpenAI(api_key=settings.openai_api_key)
        # Injectable so retry/backoff tests don't burn 7 real seconds
        # waiting on time.sleep.
        self._sleep = sleep_fn

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[start : start + MAX_BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt, delay in enumerate((0.0, *_BACKOFF_SECONDS), start=1):
            if delay:
                self._sleep(delay)
            try:
                response = self._client.embeddings.create(model=self._model, input=batch)
                # OpenAI's API returns embeddings in request order, but we
                # sort by `.index` defensively rather than assume it.
                ordered = sorted(response.data, key=lambda item: item.index)
                return [item.embedding for item in ordered]
            except openai.APIError as exc:
                last_error = exc
                logger.warning(
                    "embedding_call_failed",
                    attempt=attempt,
                    max_attempts=len(_BACKOFF_SECONDS) + 1,
                    error=str(exc),
                )

        raise EmbedderError(
            f"Embedding call failed after {len(_BACKOFF_SECONDS) + 1} attempts: {last_error}"
        ) from last_error
