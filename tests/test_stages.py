"""Mock-based tests for all Stage wrappers."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest


class TestPreprocessingStage:
    def test_run_delegates_to_preprocessor(self):
        from spindle.stages.preprocessing import PreprocessingStage

        stage = PreprocessingStage(cfg={"chunk_size": 512}, tracker=MagicMock())
        mock_chunks = [MagicMock(), MagicMock()]

        with patch("spindle.preprocessing.preprocessor.SpindlePreprocessor") as MockPrep:
            MockPrep.return_value.return_value = mock_chunks
            result = stage.run(paths=["/tmp/doc.txt"])

        assert result == mock_chunks
        MockPrep.return_value.assert_called_once()

    def test_input_schema(self):
        from spindle.stages.preprocessing import PreprocessingStage
        schema = PreprocessingStage().input_schema()
        assert schema["required"] == ["paths"]

    def test_output_schema(self):
        from spindle.stages.preprocessing import PreprocessingStage
        schema = PreprocessingStage().output_schema()
        assert schema["type"] == "array"

    def test_callable(self):
        from spindle.stages.preprocessing import PreprocessingStage
        stage = PreprocessingStage()
        with patch("spindle.preprocessing.preprocessor.SpindlePreprocessor") as MockPrep:
            MockPrep.return_value.return_value = []
            result = stage(["/tmp/doc.txt"])
        assert result == []


class TestKOSExtractionStage:
    def test_run_delegates_to_pipeline(self):
        from spindle.stages.kos_extraction import KOSExtractionStage

        mock_kos = MagicMock()
        stage = KOSExtractionStage(kos_service=mock_kos)

        with patch("spindle.kos.extraction.KOSExtractionPipeline") as MockPipe:
            MockPipe.return_value.run.return_value = {"mode": "cold_start", "chunks_processed": 3}
            result = stage.run(chunks=[MagicMock()])

        assert result["mode"] == "cold_start"
        assert result["chunks_processed"] == 3

    def test_input_schema(self):
        from spindle.stages.kos_extraction import KOSExtractionStage
        schema = KOSExtractionStage(kos_service=MagicMock()).input_schema()
        assert schema["type"] == "array"

    def test_output_schema(self):
        from spindle.stages.kos_extraction import KOSExtractionStage
        schema = KOSExtractionStage(kos_service=MagicMock()).output_schema()
        assert "mode" in schema["properties"]


class TestOntologySynthesisStage:
    def test_run_delegates_to_synthesize(self):
        from spindle.stages.ontology_synthesis import OntologySynthesisStage

        mock_kos = MagicMock()
        mock_kos._kos_dir = Path("/tmp/kos")
        stage = OntologySynthesisStage(kos_service=mock_kos)

        with patch("spindle.kos.synthesis.synthesize_ontology") as mock_synth, \
             patch("spindle.kos.synthesis.generate_shacl") as mock_shacl:
            mock_synth.return_value = {"status": "ok", "classes": 5}
            mock_shacl.return_value = {"shapes": 3}
            result = stage.run()

        assert result["status"] == "ok"
        assert result["classes"] == 5
        assert result["shacl"] == {"shapes": 3}

    def test_run_skips_shacl_on_non_ok(self):
        from spindle.stages.ontology_synthesis import OntologySynthesisStage

        mock_kos = MagicMock()
        mock_kos._kos_dir = Path("/tmp/kos")
        stage = OntologySynthesisStage(kos_service=mock_kos)

        with patch("spindle.kos.synthesis.synthesize_ontology") as mock_synth:
            mock_synth.return_value = {"status": "error"}
            result = stage.run()

        assert "shacl" not in result

    def test_input_schema(self):
        from spindle.stages.ontology_synthesis import OntologySynthesisStage
        mock_kos = MagicMock()
        mock_kos._kos_dir = Path("/tmp/kos")
        schema = OntologySynthesisStage(kos_service=mock_kos).input_schema()
        assert schema["type"] == "null"


class TestRetrievalStage:
    def test_local_mode_uses_kos(self):
        from spindle.stages.retrieval import RetrievalStage

        mock_kos = MagicMock()
        mock_kos.search_ann.return_value = [{"term": "test"}]
        stage = RetrievalStage(kos_service=mock_kos, mode="local")
        result = stage.run("test query")

        assert "kos" in result
        assert result["kos"] == [{"term": "test"}]
        assert "vector" not in result

    def test_global_mode_uses_vector(self):
        from spindle.stages.retrieval import RetrievalStage

        mock_vector = MagicMock()
        mock_vector.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"key": "val"}]],
        }
        stage = RetrievalStage(vector_store=mock_vector, mode="global")
        result = stage.run("test query")

        assert "vector" in result
        assert len(result["vector"]) == 1
        assert "kos" not in result

    def test_hybrid_mode_uses_both(self):
        from spindle.stages.retrieval import RetrievalStage

        mock_kos = MagicMock()
        mock_kos.search_ann.return_value = [{"term": "a"}]
        mock_vector = MagicMock()
        mock_vector.query.return_value = {"documents": [["b"]], "metadatas": [[{}]]}
        stage = RetrievalStage(kos_service=mock_kos, vector_store=mock_vector, mode="hybrid")
        result = stage.run("q")

        assert "kos" in result
        assert "vector" in result

    def test_graceful_with_none_stores(self):
        from spindle.stages.retrieval import RetrievalStage

        stage = RetrievalStage(mode="hybrid")
        result = stage.run("q")

        assert result == {}

    def test_graph_store_included_when_provided(self):
        from spindle.stages.retrieval import RetrievalStage

        mock_graph = MagicMock()
        mock_graph.get_triples.return_value = [{"s": "a", "p": "b", "o": "c"}]
        stage = RetrievalStage(graph_store=mock_graph, mode="local")
        result = stage.run("q")

        assert "graph" in result

    def test_mode_override_in_run(self):
        from spindle.stages.retrieval import RetrievalStage

        mock_kos = MagicMock()
        mock_kos.search_ann.return_value = []
        stage = RetrievalStage(kos_service=mock_kos, mode="global")
        # Override to local at call time
        result = stage.run("q", mode="local")

        assert "kos" in result

    def test_callable(self):
        from spindle.stages.retrieval import RetrievalStage
        stage = RetrievalStage(mode="local")
        result = stage("q")
        assert isinstance(result, dict)


class TestGenerationStage:
    def test_ontology_threaded_to_extractor(self):
        from spindle.stages.generation import GenerationStage

        mock_ontology = MagicMock()
        stage = GenerationStage(ontology=mock_ontology)

        with patch("spindle.extraction.extractor.SpindleExtractor") as MockExt:
            mock_result = MagicMock()
            mock_result.triples = []
            mock_result.reasoning = "test"
            MockExt.return_value.extract.return_value = mock_result
            stage.run(text="Hello world")

        MockExt.assert_called_once_with(ontology=mock_ontology, tracker=None)

    def test_extractor_injection_skips_create(self):
        from spindle.stages.generation import GenerationStage

        mock_extractor = MagicMock()
        mock_result = MagicMock()
        mock_result.triples = []
        mock_result.reasoning = "ok"
        mock_extractor.extract.return_value = mock_result
        stage = GenerationStage(extractor=mock_extractor)
        stage.run(text="Hello")

        mock_extractor.extract.assert_called_once()

    def test_run_returns_expected_keys(self):
        from spindle.stages.generation import GenerationStage

        mock_extractor = MagicMock()
        mock_result = MagicMock()
        mock_result.triples = []
        mock_result.reasoning = "extracted"
        mock_extractor.extract.return_value = mock_result
        stage = GenerationStage(extractor=mock_extractor)
        result = stage.run(text="Hello")

        assert "triples" in result
        assert "reasoning" in result
        assert "triple_count" in result
        assert result["triple_count"] == 0

    def test_run_batch(self):
        from spindle.stages.generation import GenerationStage

        mock_extractor = MagicMock()
        mock_result = MagicMock()
        mock_result.triples = []
        mock_result.reasoning = "ok"
        mock_extractor.extract.return_value = mock_result
        stage = GenerationStage(extractor=mock_extractor)
        results = stage.run_batch(["text1", "text2"], source_names=["a", "b"])

        assert len(results) == 2
        assert mock_extractor.extract.call_count == 2

    def test_input_schema(self):
        from spindle.stages.generation import GenerationStage
        schema = GenerationStage().input_schema()
        assert "text" in schema["properties"]
        assert schema["required"] == ["text"]

    def test_output_schema(self):
        from spindle.stages.generation import GenerationStage
        schema = GenerationStage().output_schema()
        assert "triples" in schema["properties"]


class TestEntityResolutionStage:
    def test_skip_when_no_stores(self):
        from spindle.stages.entity_resolution import EntityResolutionStage

        stage = EntityResolutionStage()
        result = stage.run()

        assert result == {"status": "skipped"}

    def test_skip_when_no_vector_store(self):
        from spindle.stages.entity_resolution import EntityResolutionStage

        stage = EntityResolutionStage(graph_store=MagicMock())
        result = stage.run()

        assert result == {"status": "skipped"}

    def test_delegates_to_resolver(self):
        from spindle.stages.entity_resolution import EntityResolutionStage

        mock_resolver = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"total_nodes_processed": 10, "same_as_edges_created": 2}
        mock_resolver.resolve_entities.return_value = mock_result

        stage = EntityResolutionStage(
            resolver=mock_resolver,
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )
        result = stage.run(apply_to_nodes=True, apply_to_edges=False, context="test")

        mock_resolver.resolve_entities.assert_called_once()
        assert result["total_nodes_processed"] == 10

    def test_creates_resolver_lazily(self):
        from spindle.stages.entity_resolution import EntityResolutionStage

        stage = EntityResolutionStage(
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )

        with patch("spindle.entity_resolution.resolver.EntityResolver") as MockResolver:
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"status": "ok"}
            MockResolver.return_value.resolve_entities.return_value = mock_result
            result = stage.run()

        MockResolver.assert_called_once()
        assert result["status"] == "ok"

    def test_input_schema(self):
        from spindle.stages.entity_resolution import EntityResolutionStage
        schema = EntityResolutionStage().input_schema()
        assert "apply_to_nodes" in schema["properties"]

    def test_output_schema(self):
        from spindle.stages.entity_resolution import EntityResolutionStage
        schema = EntityResolutionStage().output_schema()
        assert "status" in schema["properties"]

    def test_callable(self):
        from spindle.stages.entity_resolution import EntityResolutionStage
        stage = EntityResolutionStage()
        result = stage()
        assert result == {"status": "skipped"}
