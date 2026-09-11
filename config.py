"""Application configuration.

Single source of configuration truth (NFR-16): all settings are read from
environment variables via Pydantic Settings. Nothing is hardcoded, and
secrets are never given real defaults (NFR-09).

Only Phase 1 fields are actively consumed in Phase 1 (database, JWT,
logging). Fields for later phases (embeddings, LLM, chunking) are declared
now so `.env.example` stays a complete reference per ADR-006, but no
Phase 1 code path reads them yet.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database (Phase 1) ---
    # Async driver scheme is required because src/db/session.py uses
    # SQLAlchemy's async engine with asyncpg.
    database_url: str = Field(
        default="postgresql+asyncpg://rag:rag@localhost:5432/rag_db",
        description="Async SQLAlchemy connection string (postgresql+asyncpg://...)",
    )

    # --- Auth / JWT (Phase 1) ---
    jwt_secret_key: str = Field(
        default="CHANGE_ME_IN_PRODUCTION",
        description="Secret used to sign JWTs. Must be overridden outside of local dev.",
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_hours: int = Field(default=24, ge=1)

    # Placeholder single-user credentials for the demo auth flow (FR-20).
    # This system has no user registration/management (documented out of
    # scope in ARCHITECTURE.md Security Model, §11).
    auth_username: str = Field(default="admin")
    auth_password: str = Field(default="changeme")

    # --- Observability (Phase 1) ---
    log_level: str = Field(default="INFO")

    # --- Embeddings (declared now, consumed starting Phase 2) ---
    embedder_provider: str = Field(default="openai")
    embed_model: str = Field(default="text-embedding-3-small")
    openai_api_key: str = Field(default="")
    local_embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Model used by LocalEmbedder when EMBEDDER_PROVIDER=local (ADR-003).",
    )

    # --- LLM (declared now, consumed starting Phase 3/4) ---
    llm_provider: str = Field(default="anthropic")
    llm_model: str = Field(default="claude-haiku-4-5")
    anthropic_api_key: str = Field(default="")

    # --- Chunking / retrieval (declared now, consumed starting Phase 2/3) ---
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    top_k: int = Field(default=5, gt=0)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor.

    Cached with lru_cache so environment variables are parsed once per
    process rather than on every access, while remaining easy to override
    in tests via `get_settings.cache_clear()`.
    """
    return Settings()
