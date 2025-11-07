from pathlib import Path

from spindle.ingestion.service import run_ingestion
from spindle.ingestion.types import IngestionConfig, TemplateSelector, TemplateSpec
from tests.fixtures.ingestion import (
    DummyLoader,
    DummySplitter,
    dummy_graph_hook,
    dummy_metadata_extractor,
)


def test_run_ingestion_persists_catalog(tmp_path: Path) -> None:
    document_path = tmp_path / "doc.txt"
    document_path.write_text("first line\nsecond line", encoding="utf-8")

    catalog_path = tmp_path / "catalog.db"

    spec = TemplateSpec(
        name="dummy",
        selector=TemplateSelector(file_extensions=(".txt",)),
        loader=DummyLoader,
        splitter=DummySplitter,
        metadata_extractors=(dummy_metadata_extractor,),
        graph_hooks=(dummy_graph_hook,),
    )

    config = IngestionConfig(
        template_specs=(spec,),
        template_search_paths=(),
        catalog_url=f"sqlite:///{catalog_path}",
        vector_store_uri=None,
    )

    result = run_ingestion([document_path], config)

    assert result.metrics.processed_documents == 1
    assert result.metrics.processed_chunks == 2
    assert catalog_path.exists()
    assert result.document_graph.nodes

