"""get_embedder() factory unit tests (NFR-14: provider swap via config)."""

import pytest

from config import get_settings
from src.embeddings import get_embedder
from src.embeddings.local import LocalEmbedder
from src.embeddings.openai import OpenAIEmbedder


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    # get_settings() is lru_cached; each test mutates env vars via
    # monkeypatch, so the cache must be cleared before and after to avoid
    # leaking one test's provider choice into the next.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestGetEmbedderDispatch:
    def test_openai_provider_returns_openai_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDER_PROVIDER", "openai")
        assert isinstance(get_embedder(), OpenAIEmbedder)

    def test_local_provider_returns_local_embedder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDER_PROVIDER", "local")
        assert isinstance(get_embedder(), LocalEmbedder)

    def test_unsupported_provider_raises_clear_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMBEDDER_PROVIDER", "gemini")
        with pytest.raises(ValueError, match="gemini"):
            get_embedder()
