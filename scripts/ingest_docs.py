#!/usr/bin/env python3
"""Batch-ingest all supported documents in a directory.

Usage:
    python scripts/ingest_docs.py ./sample_docs/

Uses the same IngestionPipeline as POST /documents — this script exists
so a whole folder can be indexed in one command (UC-08), not because the
ingestion logic differs between the CLI and API entry points.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow `python scripts/ingest_docs.py ...` to import `src`/`config` when
# invoked directly (only the script's own directory is on sys.path by
# default in that invocation style).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.db.session import get_sessionmaker  # noqa: E402
from src.embeddings import get_embedder  # noqa: E402
from src.ingestion.parsers import SUPPORTED_EXTENSIONS, ParserError  # noqa: E402
from src.ingestion.pipeline import (  # noqa: E402
    FileTooLargeError,
    IngestionError,
    IngestionPipeline,
)
from src.observability.logger import configure_logging  # noqa: E402
from src.retrieval.vector_store import PgVectorStore  # noqa: E402


async def ingest_directory(directory: Path) -> int:
    session_factory = get_sessionmaker()
    embedder = get_embedder()

    files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(
            f"No supported documents found in {directory} "
            f"(supported: {sorted(SUPPORTED_EXTENSIONS)})"
        )
        return 0

    indexed = 0
    failed = 0
    for file_path in files:
        async with session_factory() as session:
            pipeline = IngestionPipeline(
                session=session, embedder=embedder, vector_store=PgVectorStore(session)
            )
            try:
                outcome = await pipeline.ingest(file_path, filename=file_path.name)
            except (FileTooLargeError, ParserError, IngestionError) as exc:
                # NFR-05: a failed document parse must not affect other
                # documents in the batch — continue to the next file.
                failed += 1
                print(f"  FAILED        {file_path.name}: {exc}")
                continue

        if outcome.status == "INDEXED":
            indexed += 1
            print(
                f"  INDEXED       {file_path.name} "
                f"({outcome.chunk_count} chunks, {outcome.duration_ms}ms)"
            )
        else:
            failed += 1
            print(f"  PARSE_FAILED  {file_path.name} (reason: {outcome.reason})")

    print(f"\nSummary: {indexed} indexed, {failed} failed, {len(files)} total")
    return 1 if failed and not indexed else 0


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Batch-ingest documents into the RAG knowledge base"
    )
    parser.add_argument("directory", type=Path, help="Directory of documents to ingest")
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        sys.exit(2)

    exit_code = asyncio.run(ingest_directory(args.directory))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
