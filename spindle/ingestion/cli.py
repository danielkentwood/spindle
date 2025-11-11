"""Command line interface for the spindle ingestion pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator

from spindle.configuration import (
    ConfigurationError,
    DEFAULT_STORAGE_ROOT_NAME,
    SpindleConfig,
    default_config,
    load_config_from_file,
    render_default_config,
)
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
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a spindle config.py that defines SPINDLE_CONFIG",
    )
    parser.add_argument("inputs", nargs="+", help="Files or directories to ingest")
    parser.add_argument(
        "--templates",
        nargs="*",
        default=None,
        help="Directories containing ingestion templates",
    )
    parser.add_argument(
        "--catalog-url",
        default=None,
        help="SQLAlchemy database URL for the catalog",
    )
    parser.add_argument(
        "--vector-store",
        default=None,
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


def _load_spindle_config(config_path: str | None) -> SpindleConfig:
    if config_path:
        try:
            return load_config_from_file(config_path)
        except ConfigurationError as exc:
            raise SystemExit(str(exc)) from exc
    return default_config()


def _run_config_init(args: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a spindle config.py with default paths",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="config.py",
        help="Destination file path (default: config.py)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Root storage directory for generated config "
            f"(default: <cwd>/{DEFAULT_STORAGE_ROOT_NAME})"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists",
    )
    options = parser.parse_args(args)
    output_path = Path(options.output).expanduser()
    if output_path.exists() and not options.force:
        print(
            f"Error: {output_path} already exists. Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    storage_root = (
        Path(options.root).expanduser()
        if options.root
        else Path.cwd() / DEFAULT_STORAGE_ROOT_NAME
    )
    content = render_default_config(storage_root)
    output_path.write_text(content)
    default_config(storage_root).storage.ensure_directories()
    print(f"Wrote default configuration to {output_path}")
    return 0


def _run_ingest(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    vector_store = None if args.no_vector_store else args.vector_store
    catalog_url = None if args.no_catalog else args.catalog_url
    spindle_cfg = _load_spindle_config(args.config)
    template_paths = (
        [Path(p).expanduser() for p in args.templates] if args.templates else None
    )

    config = build_config(
        template_paths=template_paths,
        catalog_url=catalog_url,
        vector_store_uri=vector_store,
        spindle_config=spindle_cfg,
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


def main(argv: list[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if argv_list and argv_list[0] == "config":
        return _run_config_init(argv_list[1:])
    return _run_ingest(argv_list)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

