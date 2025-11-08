"""Command line interface for the spindle ingestion pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator

from spindle.ingestion.service import build_config, run_ingestion
from spindle.observability import attach_persistent_observer, get_event_recorder
from spindle.observability.storage import EventLogStore

RECORDER = get_event_recorder("ingestion.cli")


def _expand_paths(inputs: Iterable[str], recursive: bool) -> Iterator[Path]:
    for input_path in inputs:
        path = Path(input_path).expanduser()
        if path.is_dir():
            yield from _walk_directory(path, recursive)
        else:
            yield path


def _walk_directory(path: Path, recursive: bool) -> Iterator[Path]:
    if recursive:
        for file_path in path.rglob("*"):
            if file_path.is_file():
                yield file_path
    else:
        for file_path in path.iterdir():
            if file_path.is_file():
                yield file_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest documents into spindle")
    parser.add_argument("inputs", nargs="+", help="Files or directories to ingest")
    parser.add_argument(
        "--templates",
        nargs="*",
        default=[],
        help="Directories containing ingestion templates",
    )
    parser.add_argument(
        "--catalog-url",
        default="sqlite:///spindle_ingestion.db",
        help="SQLAlchemy database URL for the catalog",
    )
    parser.add_argument(
        "--vector-store",
        default=".spindle/vector_store",
        help="Directory for the Chroma vector store",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into directories when ingesting",
    )
    parser.add_argument(
        "--no-vector-store",
        action="store_true",
        help="Disable vector store persistence",
    )
    parser.add_argument(
        "--no-catalog",
        action="store_true",
        help="Disable SQLite catalog persistence",
    )
    parser.add_argument(
        "--event-log",
        default=None,
        help="SQLAlchemy database URL for persisting service events",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    vector_store = None if args.no_vector_store else args.vector_store
    catalog_url = None if args.no_catalog else args.catalog_url
    template_paths = [Path(p).expanduser() for p in args.templates]

    config = build_config(
        template_paths=template_paths,
        catalog_url=catalog_url,
        vector_store_uri=vector_store,
    )
    detach_observer = None
    if args.event_log:
        store = EventLogStore(args.event_log)
        detach_observer = attach_persistent_observer(get_event_recorder(), store)
    RECORDER.record(
        name="run.start",
        payload={
            "input_count": len(args.inputs),
            "recursive": args.recursive,
            "catalog_url": catalog_url,
            "vector_store": vector_store,
        },
    )
    try:
        result = run_ingestion(_expand_paths(args.inputs, args.recursive), config)
    except Exception as exc:
        RECORDER.record(
            name="run.error",
            payload={
                "input_count": len(args.inputs),
                "error": str(exc),
            },
        )
        raise
    finally:
        if detach_observer:
            detach_observer()
    RECORDER.record(
        name="run.complete",
        payload={
            "processed_documents": result.metrics.processed_documents,
            "processed_chunks": result.metrics.processed_chunks,
            "error_count": len(result.metrics.errors),
        },
    )

    print(
        f"Ingested {result.metrics.processed_documents} documents "
        f"into {catalog_url or 'memory'}; chunks={result.metrics.processed_chunks}"
    )
    if result.metrics.errors:
        print("Errors:")
        for error in result.metrics.errors:
            print(f"  - {error}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

