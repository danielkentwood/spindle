"""Command line interface for the Ontology Pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from spindle.configuration import (
    SpindleConfig,
    default_config,
    load_config_from_file,
)
from spindle.graph_store import GraphStore
from spindle.ingestion.storage import DocumentCatalog
from spindle.ingestion.storage.corpus import CorpusManager
from spindle.pipeline import (
    ExtractionStrategyType,
    PipelineOrchestrator,
    PipelineStage,
)


def _load_config(config_path: Optional[str]) -> SpindleConfig:
    """Load configuration from file or use defaults."""
    if config_path:
        return load_config_from_file(config_path)
    return default_config()


def _get_orchestrator(config: SpindleConfig) -> PipelineOrchestrator:
    """Create a PipelineOrchestrator from config."""
    config.storage.ensure_directories()

    catalog_path = config.storage.catalog_url or f"sqlite:///{config.storage.log_dir}/catalog.db"
    catalog = DocumentCatalog(catalog_path)
    corpus_manager = CorpusManager(catalog)

    graph_path = str(config.storage.graph_store_path)
    graph_store = GraphStore(graph_path)

    orchestrator = PipelineOrchestrator(corpus_manager, graph_store)
    orchestrator.register_default_stages()

    return orchestrator


def _get_strategy(strategy_str: str) -> ExtractionStrategyType:
    """Parse strategy string to enum."""
    mapping = {
        "sequential": ExtractionStrategyType.SEQUENTIAL,
        "batch": ExtractionStrategyType.BATCH_CONSOLIDATE,
        "batch_consolidate": ExtractionStrategyType.BATCH_CONSOLIDATE,
        "sample": ExtractionStrategyType.SAMPLE_BASED,
        "sample_based": ExtractionStrategyType.SAMPLE_BASED,
    }
    return mapping.get(strategy_str.lower(), ExtractionStrategyType.SEQUENTIAL)


# =============================================================================
# Corpus Commands
# =============================================================================


def cmd_corpus_create(args: argparse.Namespace) -> int:
    """Create a new corpus."""
    config = _load_config(args.config)
    orchestrator = _get_orchestrator(config)

    corpus = orchestrator.corpus_manager.create_corpus(
        name=args.name,
        description=args.description or "",
        corpus_id=args.id,
    )

    print(f"Created corpus: {corpus.corpus_id}")
    print(f"  Name: {corpus.name}")
    if corpus.description:
        print(f"  Description: {corpus.description}")

    return 0


def cmd_corpus_list(args: argparse.Namespace) -> int:
    """List all corpora."""
    config = _load_config(args.config)
    orchestrator = _get_orchestrator(config)

    corpora = orchestrator.corpus_manager.list_corpora()

    if not corpora:
        print("No corpora found.")
        return 0

    print(f"Found {len(corpora)} corpus(a):\n")
    for corpus in corpora:
        doc_count = orchestrator.corpus_manager.get_corpus_document_count(corpus.corpus_id)
        completed = len(corpus.pipeline_state.get("completed_stages", []))

        print(f"  {corpus.corpus_id}")
        print(f"    Name: {corpus.name}")
        print(f"    Documents: {doc_count}")
        print(f"    Pipeline stages completed: {completed}/6")
        print()

    return 0


def cmd_corpus_info(args: argparse.Namespace) -> int:
    """Show corpus details."""
    config = _load_config(args.config)
    orchestrator = _get_orchestrator(config)

    corpus = orchestrator.corpus_manager.get_corpus(args.corpus_id)
    if corpus is None:
        print(f"Error: Corpus not found: {args.corpus_id}", file=sys.stderr)
        return 1

    doc_count = orchestrator.corpus_manager.get_corpus_document_count(corpus.corpus_id)
    status = orchestrator.get_status(corpus)

    print(f"Corpus: {corpus.corpus_id}")
    print(f"  Name: {corpus.name}")
    print(f"  Description: {corpus.description or '(none)'}")
    print(f"  Created: {corpus.created_at}")
    print(f"  Updated: {corpus.updated_at}")
    print(f"  Documents: {doc_count}")
    print()
    print("Pipeline Status:")
    print(f"  Completed stages: {', '.join(status['completed_stages']) or '(none)'}")
    print(f"  Pending stages: {', '.join(status['pending_stages'])}")
    if status["current_stage"]:
        print(f"  Current stage: {status['current_stage']}")

    return 0


def cmd_corpus_add(args: argparse.Namespace) -> int:
    """Add documents to a corpus."""
    config = _load_config(args.config)
    orchestrator = _get_orchestrator(config)

    corpus = orchestrator.corpus_manager.get_corpus(args.corpus_id)
    if corpus is None:
        print(f"Error: Corpus not found: {args.corpus_id}", file=sys.stderr)
        return 1

    # If document IDs provided directly
    if args.document_ids:
        count = orchestrator.corpus_manager.add_documents(
            args.corpus_id,
            args.document_ids,
        )
        print(f"Added {count} document(s) to corpus {args.corpus_id}")
        return 0

    # Otherwise, list documents from catalog that match input paths
    print("Note: To add documents, first ingest them using spindle-ingest,")
    print("then provide their document IDs to this command.")
    return 1


def cmd_corpus_delete(args: argparse.Namespace) -> int:
    """Delete a corpus."""
    config = _load_config(args.config)
    orchestrator = _get_orchestrator(config)

    deleted = orchestrator.corpus_manager.delete_corpus(
        args.corpus_id,
        delete_documents=args.remove_documents,
    )

    if not deleted:
        print(f"Error: Corpus not found: {args.corpus_id}", file=sys.stderr)
        return 1

    print(f"Deleted corpus: {args.corpus_id}")
    return 0


# =============================================================================
# Pipeline Commands
# =============================================================================


def cmd_pipeline_run(args: argparse.Namespace) -> int:
    """Run pipeline stages."""
    config = _load_config(args.config)
    orchestrator = _get_orchestrator(config)

    corpus = orchestrator.corpus_manager.get_corpus(args.corpus_id)
    if corpus is None:
        print(f"Error: Corpus not found: {args.corpus_id}", file=sys.stderr)
        return 1

    strategy = _get_strategy(args.strategy)

    if args.stage == "all":
        print(f"Running all pipeline stages on corpus {args.corpus_id}...")
        print(f"Strategy: {strategy.value}")
        print()

        results = orchestrator.run_all(corpus, strategy, stop_on_error=not args.continue_on_error)

        for result in results:
            status_icon = "✓" if result.success else "✗"
            print(f"  {status_icon} {result.stage.value}: ", end="")
            if result.success:
                print(f"{result.artifact_count} artifacts")
            else:
                print(f"FAILED - {result.error_message}")

        success_count = sum(1 for r in results if r.success)
        print()
        print(f"Completed {success_count}/{len(results)} stages")

        return 0 if all(r.success for r in results) else 1

    else:
        # Run single stage
        try:
            stage = PipelineStage(args.stage)
        except ValueError:
            valid = [s.value for s in PipelineStage]
            print(f"Error: Invalid stage '{args.stage}'. Valid stages: {valid}", file=sys.stderr)
            return 1

        print(f"Running {stage.value} stage on corpus {args.corpus_id}...")

        result = orchestrator.run_stage(corpus, stage, strategy)

        if result.success:
            print(f"✓ Success: {result.artifact_count} artifacts extracted")
            duration = (result.finished_at - result.started_at).total_seconds()
            print(f"  Duration: {duration:.2f}s")
            return 0
        else:
            print(f"✗ Failed: {result.error_message}")
            return 1


def cmd_pipeline_status(args: argparse.Namespace) -> int:
    """Show pipeline status for a corpus."""
    config = _load_config(args.config)
    orchestrator = _get_orchestrator(config)

    corpus = orchestrator.corpus_manager.get_corpus(args.corpus_id)
    if corpus is None:
        print(f"Error: Corpus not found: {args.corpus_id}", file=sys.stderr)
        return 1

    status = orchestrator.get_status(corpus)

    print(f"Pipeline Status for: {status['corpus_name']} ({status['corpus_id']})")
    print()

    if status["started_at"]:
        print(f"Started: {status['started_at']}")
    if status["finished_at"]:
        print(f"Finished: {status['finished_at']}")
    print(f"Strategy: {status['strategy']}")
    print()

    print("Stages:")
    all_stages = list(PipelineStage)
    for stage in all_stages:
        stage_name = stage.value
        if stage_name in [s for s in status["completed_stages"]]:
            result = status["stage_results"].get(stage_name, {})
            artifacts = result.get("artifact_count", 0) if result else 0
            print(f"  ✓ {stage_name}: {artifacts} artifacts")
        elif stage_name == status.get("current_stage"):
            print(f"  → {stage_name}: IN PROGRESS")
        else:
            print(f"  ○ {stage_name}: pending")

    return 0


def cmd_pipeline_reset(args: argparse.Namespace) -> int:
    """Reset pipeline state for a corpus."""
    config = _load_config(args.config)
    orchestrator = _get_orchestrator(config)

    corpus = orchestrator.corpus_manager.get_corpus(args.corpus_id)
    if corpus is None:
        print(f"Error: Corpus not found: {args.corpus_id}", file=sys.stderr)
        return 1

    if args.from_stage:
        try:
            stage = PipelineStage(args.from_stage)
        except ValueError:
            valid = [s.value for s in PipelineStage]
            print(f"Error: Invalid stage '{args.from_stage}'. Valid: {valid}", file=sys.stderr)
            return 1

        orchestrator.reset_from_stage(corpus, stage)
        print(f"Reset pipeline from stage {args.from_stage} for corpus {args.corpus_id}")
    else:
        orchestrator.reset_pipeline(corpus)
        print(f"Reset all pipeline state for corpus {args.corpus_id}")

    return 0


def cmd_pipeline_export(args: argparse.Namespace) -> int:
    """Export pipeline artifacts."""
    config = _load_config(args.config)
    orchestrator = _get_orchestrator(config)

    corpus = orchestrator.corpus_manager.get_corpus(args.corpus_id)
    if corpus is None:
        print(f"Error: Corpus not found: {args.corpus_id}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    export_data = {
        "corpus_id": corpus.corpus_id,
        "corpus_name": corpus.name,
    }

    # Export vocabulary
    vocab_stage = orchestrator.get_stage(PipelineStage.VOCABULARY)
    if vocab_stage:
        terms = vocab_stage.load_artifacts(args.corpus_id)
        export_data["vocabulary"] = [
            {
                "term_id": t.term_id,
                "preferred_label": t.preferred_label,
                "definition": t.definition,
                "synonyms": t.synonyms,
                "domain": t.domain,
            }
            for t in terms
        ]

    # Export thesaurus
    thesaurus_stage = orchestrator.get_stage(PipelineStage.THESAURUS)
    if thesaurus_stage:
        entries = thesaurus_stage.load_artifacts(args.corpus_id)
        export_data["thesaurus"] = [
            {
                "preferred_label": e.preferred_label,
                "use_for": e.use_for,
                "broader_terms": e.broader_terms,
                "narrower_terms": e.narrower_terms,
                "related_terms": e.related_terms,
            }
            for e in entries
        ]

    # Export ontology
    ontology_stage = orchestrator.get_stage(PipelineStage.ONTOLOGY)
    if ontology_stage:
        ontologies = ontology_stage.load_artifacts(args.corpus_id)
        if ontologies:
            ont = ontologies[0]
            export_data["ontology"] = {
                "entity_types": [
                    {
                        "name": et.name,
                        "description": et.description,
                        "attributes": [
                            {"name": a.name, "type": a.type, "description": a.description}
                            for a in et.attributes
                        ],
                    }
                    for et in ont.entity_types
                ],
                "relation_types": [
                    {
                        "name": rt.name,
                        "description": rt.description,
                        "domain": rt.domain,
                        "range": rt.range,
                    }
                    for rt in ont.relation_types
                ],
            }

    output_file = output_dir / f"{args.corpus_id}_pipeline_export.json"
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"Exported pipeline artifacts to: {output_file}")
    return 0


# =============================================================================
# Argument Parsing
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="spindle-pipeline",
        description="Ontology Pipeline CLI for building semantic knowledge systems",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to spindle config.py",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # corpus create
    corpus_create = subparsers.add_parser("corpus-create", help="Create a new corpus")
    corpus_create.add_argument("name", help="Corpus name")
    corpus_create.add_argument("--description", "-d", help="Corpus description")
    corpus_create.add_argument("--id", help="Custom corpus ID")

    # corpus list
    subparsers.add_parser("corpus-list", help="List all corpora")

    # corpus info
    corpus_info = subparsers.add_parser("corpus-info", help="Show corpus details")
    corpus_info.add_argument("corpus_id", help="Corpus ID")

    # corpus add
    corpus_add = subparsers.add_parser("corpus-add", help="Add documents to corpus")
    corpus_add.add_argument("corpus_id", help="Corpus ID")
    corpus_add.add_argument("document_ids", nargs="*", help="Document IDs to add")

    # corpus delete
    corpus_delete = subparsers.add_parser("corpus-delete", help="Delete a corpus")
    corpus_delete.add_argument("corpus_id", help="Corpus ID")
    corpus_delete.add_argument(
        "--remove-documents",
        action="store_true",
        help="Also remove document associations",
    )

    # run
    run = subparsers.add_parser("run", help="Run pipeline stages")
    run.add_argument("corpus_id", help="Corpus ID")
    run.add_argument(
        "--stage",
        "-s",
        default="all",
        help="Stage to run (vocabulary, metadata, taxonomy, thesaurus, ontology, knowledge_graph, or 'all')",
    )
    run.add_argument(
        "--strategy",
        default="sequential",
        choices=["sequential", "batch", "sample"],
        help="Extraction strategy",
    )
    run.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running stages even if one fails",
    )

    # status
    status = subparsers.add_parser("status", help="Show pipeline status")
    status.add_argument("corpus_id", help="Corpus ID")

    # reset
    reset = subparsers.add_parser("reset", help="Reset pipeline state")
    reset.add_argument("corpus_id", help="Corpus ID")
    reset.add_argument(
        "--from-stage",
        help="Reset from this stage onward (inclusive)",
    )

    # export
    export = subparsers.add_parser("export", help="Export pipeline artifacts")
    export.add_argument("corpus_id", help="Corpus ID")
    export.add_argument(
        "--output",
        "-o",
        help="Output directory (default: current directory)",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "corpus-create": cmd_corpus_create,
        "corpus-list": cmd_corpus_list,
        "corpus-info": cmd_corpus_info,
        "corpus-add": cmd_corpus_add,
        "corpus-delete": cmd_corpus_delete,
        "run": cmd_pipeline_run,
        "status": cmd_pipeline_status,
        "reset": cmd_pipeline_reset,
        "export": cmd_pipeline_export,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            return handler(args)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

