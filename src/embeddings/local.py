"""Local/offline embedding provider (ADR-003's documented extension point).

Runs a small open-weight model entirely on-device via sentence-transformers
— zero API cost, no API key, no rate limits, and (once the model is
cached) no network dependency at all. ADR-003 names this alternative
specifically for cost- or data-privacy-constrained deployments.
"""

from collections.abc import Callable

from config import get_settings
from src.embeddings.base import EmbedderBase, EmbedderError

ModelFactory = Callable[[str], object]


def _default_model_factory(model_name: str) -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class LocalEmbedder(EmbedderBase):
    def __init__(
        self,
        model: object | None = None,
        model_factory: ModelFactory = _default_model_factory,
    ) -> None:
        settings = get_settings()
        self._model_name = settings.local_embed_model
        # Injectable for unit test isolation (NFR-22), the same pattern
        # OpenAIEmbedder uses for its client: production lazily loads the
        # real model on first use (model=None initially), tests pass a
        # fake exposing a compatible .encode() method — no multi-hundred-MB
        # model download needed to test the wiring.
        self._model = model
        self._model_factory = model_factory

    def _get_model(self) -> object:
        if self._model is None:
            try:
                self._model = self._model_factory(self._model_name)
            except Exception as exc:
                raise EmbedderError(
                    f"Failed to load local embedding model {self._model_name!r}: {exc}"
                ) from exc
        return self._model

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        try:
            vectors = model.encode(texts, normalize_embeddings=True)  # type: ignore[attr-defined]
        except Exception as exc:
            raise EmbedderError(f"Local embedding inference failed: {exc}") from exc
        return [vector.tolist() for vector in vectors]
