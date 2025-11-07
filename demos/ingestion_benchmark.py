"""Quick benchmark for the ingestion pipeline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from spindle.ingestion.service import build_config, run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark spindle ingestion")
    parser.add_argument("path", help="Directory of documents to ingest")
    parser.add_argument(
        "--templates",
        nargs="*",
        default=[],
        help="Optional template directories",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        help="Optional SQLite catalog URL",
    )
    parser.add_argument(
        "--vector-store",
        default=None,
        help="Optional vector store directory",
    )
    args = parser.parse_args()

    path = Path(args.path)
    files = sorted(p for p in path.rglob("*") if p.is_file())
    config = build_config(
        template_paths=[Path(p) for p in args.templates],
        catalog_url=args.catalog,
        vector_store_uri=args.vector_store,
    )

    start = time.perf_counter()
    result = run_ingestion(files, config)
    elapsed = time.perf_counter() - start

    docs = result.metrics.processed_documents or 1
    rate = docs / elapsed
    print(
        f"Processed {result.metrics.processed_documents} documents "
        f"({result.metrics.processed_chunks} chunks) in {elapsed:.2f}s "
        f"[{rate:.2f} docs/s]"
    )


if __name__ == "__main__":  # pragma: no cover
    main()

