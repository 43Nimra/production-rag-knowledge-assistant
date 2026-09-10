"""Embedder unit tests.

OpenAIEmbedder's `client` and `sleep_fn` are injectable specifically so
these tests never hit the network or a real API key (NFR-22), and don't
burn real wall-clock time on the 1s/2s/4s backoff.
"""

from dataclasses import dataclass

import httpx
import openai
import pytest

from src.embeddings.base import EmbedderError
from src.embeddings.openai import MAX_BATCH_SIZE, OpenAIEmbedder


@dataclass
class _FakeEmbeddingItem:
    index: int
    embedding: list[float]


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingItem]


class _FakeEmbeddingsAPI:
    """Stands in for `client.embeddings`. `responses` is a list of
    behaviors consumed one per call: an int means "raise a connection
    error", anything else is returned as the batch's vectors."""

    def __init__(self, behaviors: list) -> None:
        self._behaviors = list(behaviors)
        self.calls: list[list[str]] = []

    def create(self, model: str, input: list[str]) -> _FakeEmbeddingResponse:  # noqa: A002
        self.calls.append(input)
        behavior = self._behaviors.pop(0)
        if behavior == "error":
            raise openai.APIConnectionError(
                request=httpx.Request("POST", "https://api.openai.com/v1/embeddings")
            )
        # behavior is a list of vectors, one per input text; returned out
        # of order with explicit `.index` to exercise the reordering logic.
        items = [_FakeEmbeddingItem(index=i, embedding=vec) for i, vec in enumerate(behavior)]
        return _FakeEmbeddingResponse(data=list(reversed(items)))


class _FakeClient:
    def __init__(self, behaviors: list) -> None:
        self.embeddings = _FakeEmbeddingsAPI(behaviors)


def _no_sleep(_seconds: float) -> None:
    return None


class TestEmbedderBaseContract:
    def test_embed_query_returns_single_vector(self) -> None:
        client = _FakeClient(behaviors=[[[0.1, 0.2, 0.3]]])
        embedder = OpenAIEmbedder(client=client, sleep_fn=_no_sleep)  # type: ignore[arg-type]
        vector = embedder.embed_query("hello")
        assert vector == [0.1, 0.2, 0.3]

    def test_embed_documents_returns_vectors_in_input_order(self) -> None:
        client = _FakeClient(behaviors=[[[1.0], [2.0], [3.0]]])
        embedder = OpenAIEmbedder(client=client, sleep_fn=_no_sleep)  # type: ignore[arg-type]
        vectors = embedder.embed_documents(["a", "b", "c"])
        # Response was returned reversed; embedder must restore order via
        # each item's `.index`.
        assert vectors == [[1.0], [2.0], [3.0]]


class TestBatching:
    def test_embed_documents_batches_at_max_batch_size(self) -> None:
        texts = [f"text-{i}" for i in range(MAX_BATCH_SIZE + 50)]
        behaviors = [
            [[float(i)] for i in range(MAX_BATCH_SIZE)],
            [[float(i)] for i in range(50)],
        ]
        client = _FakeClient(behaviors=behaviors)
        embedder = OpenAIEmbedder(client=client, sleep_fn=_no_sleep)  # type: ignore[arg-type]

        vectors = embedder.embed_documents(texts)

        assert len(client.embeddings.calls) == 2
        assert len(client.embeddings.calls[0]) == MAX_BATCH_SIZE
        assert len(client.embeddings.calls[1]) == 50
        assert len(vectors) == len(texts)


class TestRetryAndBackoff:
    def test_retries_on_transient_error_then_succeeds(self) -> None:
        client = _FakeClient(behaviors=["error", "error", [[9.9]]])
        embedder = OpenAIEmbedder(client=client, sleep_fn=_no_sleep)  # type: ignore[arg-type]
        vectors = embedder.embed_documents(["only"])
        assert vectors == [[9.9]]
        assert len(client.embeddings.calls) == 3

    def test_raises_embedder_error_after_exhausting_retries(self) -> None:
        client = _FakeClient(behaviors=["error", "error", "error", "error"])
        embedder = OpenAIEmbedder(client=client, sleep_fn=_no_sleep)  # type: ignore[arg-type]
        with pytest.raises(EmbedderError):
            embedder.embed_documents(["only"])
        # 1 initial attempt + 3 retries = 4 total calls.
        assert len(client.embeddings.calls) == 4

    def test_backoff_delays_follow_documented_schedule(self) -> None:
        delays: list[float] = []
        client = _FakeClient(behaviors=["error", "error", [[1.0]]])
        embedder = OpenAIEmbedder(client=client, sleep_fn=delays.append)  # type: ignore[arg-type]
        embedder.embed_documents(["only"])
        # ARCHITECTURE.md §5.2: "Exponential backoff: 1s, 2s, 4s".
        assert delays == [1.0, 2.0]
