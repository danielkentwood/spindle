"""Tests for eval_bridge: adapters, get_pipeline_definition, discovery compat."""

from unittest.mock import MagicMock, patch
import pytest

from spindle.eval_bridge import (
    StageDef,
    StageResult,
    PreprocessingAdapter,
    KOSExtractionAdapter,
    OntologySynthesisAdapter,
    RetrievalAdapter,
    GenerationAdapter,
    _get,
    get_pipeline_definition,
)


# ---------------------------------------------------------------------------
# _get helper
# ---------------------------------------------------------------------------

class TestGet:
    def test_dict(self):
        assert _get({"a": 1}, "a", 0) == 1

    def test_dict_default(self):
        assert _get({"a": 1}, "b", 42) == 42

    def test_none_cfg(self):
        assert _get(None, "x", "default") == "default"

    def test_object_attr(self):
        class Cfg:
            mode = "hybrid"
        assert _get(Cfg(), "mode", "local") == "hybrid"

    def test_object_attr_default(self):
        class Cfg:
            pass
        assert _get(Cfg(), "mode", "local") == "local"


# ---------------------------------------------------------------------------
# StageDef / StageResult fallback types
# ---------------------------------------------------------------------------

class TestStageDef:
    def test_defaults(self):
        sd = StageDef(name="test", stage=MagicMock())
        assert sd.input_keys == {}
        assert sd.metrics == []
        assert sd.gate is None

    def test_with_input_keys(self):
        sd = StageDef(
            name="generation",
            stage=MagicMock(),
            input_keys={"contexts": "retrieval.contexts"},
        )
        assert sd.input_keys["contexts"] == "retrieval.contexts"


class TestStageResult:
    def test_defaults(self):
        sr = StageResult()
        assert sr.outputs == {}
        assert sr.metrics == {}

    def test_with_data(self):
        sr = StageResult(outputs={"chunks": [1, 2]}, metrics={"count": 2})
        assert sr.outputs["chunks"] == [1, 2]
        assert sr.metrics["count"] == 2


# ---------------------------------------------------------------------------
# Adapter tests
# ---------------------------------------------------------------------------

class TestPreprocessingAdapter:
    def test_run_calls_inner_with_paths(self):
        inner = MagicMock()
        inner.run.return_value = ["chunk1", "chunk2"]
        adapter = PreprocessingAdapter(inner)

        cfg = {"dataset_path": ["/tmp/a.txt", "/tmp/b.txt"]}
        result = adapter.run(inputs={}, cfg=cfg)

        inner.run.assert_called_once_with(paths=["/tmp/a.txt", "/tmp/b.txt"])
        assert result.outputs["chunks"] == ["chunk1", "chunk2"]
        assert result.metrics["chunk_count"] == 2

    def test_name(self):
        assert PreprocessingAdapter(MagicMock()).name == "preprocessing"

    def test_string_path_converted_to_list(self):
        inner = MagicMock()
        inner.run.return_value = []
        adapter = PreprocessingAdapter(inner)

        adapter.run(inputs={}, cfg={"dataset_path": "/tmp/single.txt"})
        inner.run.assert_called_once_with(paths=["/tmp/single.txt"])

    def test_missing_dataset_path_uses_empty(self):
        inner = MagicMock()
        inner.run.return_value = []
        adapter = PreprocessingAdapter(inner)

        adapter.run(inputs={}, cfg={})
        inner.run.assert_called_once_with(paths=[])


class TestKOSExtractionAdapter:
    def test_run_passes_chunks(self):
        inner = MagicMock()
        inner.run.return_value = {
            "kos_path": "/tmp/kos.ttl",
            "chunks_processed": 5,
            "mode": "cold_start",
        }
        adapter = KOSExtractionAdapter(inner)

        result = adapter.run(
            inputs={"chunks": ["c1", "c2"]},
            cfg={},
        )

        inner.run.assert_called_once_with(chunks=["c1", "c2"])
        assert result.outputs["kos_path"] == "/tmp/kos.ttl"
        assert result.metrics["chunks_processed"] == 5

    def test_name(self):
        assert KOSExtractionAdapter(MagicMock()).name == "kos_extraction"


class TestOntologySynthesisAdapter:
    def test_run_calls_inner(self):
        inner = MagicMock()
        inner.run.return_value = {
            "ontology_path": "/tmp/ont.ttl",
            "shapes_path": "/tmp/shapes.ttl",
            "graph": MagicMock(),
            "classes": 10,
        }
        adapter = OntologySynthesisAdapter(inner)

        result = adapter.run(inputs={}, cfg={})

        inner.run.assert_called_once_with()
        assert result.outputs["ontology_path"] == "/tmp/ont.ttl"
        assert result.outputs["shapes_path"] == "/tmp/shapes.ttl"
        assert result.metrics["classes"] == 10

    def test_name(self):
        assert OntologySynthesisAdapter(MagicMock()).name == "ontology_synthesis"


class TestRetrievalAdapter:
    def test_run_with_query_from_inputs(self):
        inner = MagicMock()
        inner.run.return_value = {"graph": [], "vector": []}
        adapter = RetrievalAdapter(inner)

        result = adapter.run(inputs={"query": "test query"}, cfg={})

        inner.run.assert_called_once_with(query="test query")
        assert result.outputs["contexts"] == {"graph": [], "vector": []}

    def test_run_with_query_from_cfg(self):
        inner = MagicMock()
        inner.run.return_value = {}
        adapter = RetrievalAdapter(inner)

        adapter.run(inputs={}, cfg={"query": "cfg query"})
        inner.run.assert_called_once_with(query="cfg query")

    def test_name(self):
        assert RetrievalAdapter(MagicMock()).name == "retrieval"


class TestGenerationAdapter:
    def test_run_extracts_from_dict_contexts(self):
        inner = MagicMock()
        inner.run.return_value = {"triples": ["t1"], "triple_count": 1}
        adapter = GenerationAdapter(inner)

        contexts = {"graph": [{"text": "hello"}], "vector": [{"text": "world"}]}
        result = adapter.run(inputs={"contexts": contexts}, cfg={})

        assert inner.run.call_count == 2
        assert result.outputs["triples"] == ["t1", "t1"]
        assert result.metrics["triple_count"] == 2

    def test_run_extracts_from_list_contexts(self):
        inner = MagicMock()
        inner.run.return_value = {"triples": ["t1"], "triple_count": 1}
        adapter = GenerationAdapter(inner)

        contexts = [{"text": "hello"}]
        result = adapter.run(inputs={"contexts": contexts}, cfg={})

        inner.run.assert_called_once_with(text="hello")
        assert result.outputs["triples"] == ["t1"]

    def test_run_with_string_contexts(self):
        inner = MagicMock()
        inner.run.return_value = {"triples": [], "triple_count": 0}
        adapter = GenerationAdapter(inner)

        contexts = ["plain text"]
        result = adapter.run(inputs={"contexts": contexts}, cfg={})

        inner.run.assert_called_once_with(text="plain text")

    def test_name(self):
        assert GenerationAdapter(MagicMock()).name == "generation"


# ---------------------------------------------------------------------------
# get_pipeline_definition
# ---------------------------------------------------------------------------

class TestGetPipelineDefinition:
    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_returns_list(self, _mock_kos):
        result = get_pipeline_definition(include_kos=False, include_generation=False)
        assert isinstance(result, list)

    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_minimal_pipeline_no_kos_no_generation(self, _mock_kos):
        stages = get_pipeline_definition(include_kos=False, include_generation=False)

        assert len(stages) == 1
        assert stages[0].name == "preprocessing"

    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_each_element_is_stagedef(self, _mock_kos):
        stages = get_pipeline_definition(include_kos=False, include_generation=True)

        for sd in stages:
            assert isinstance(sd, StageDef)
            assert hasattr(sd.stage, "name")
            assert hasattr(sd.stage, "run")

    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_adapter_run_returns_stage_result(self, _mock_kos):
        stages = get_pipeline_definition(include_kos=False, include_generation=False)

        adapter = stages[0].stage
        # Patch inner stage run
        adapter._inner.run = MagicMock(return_value=[])
        result = adapter.run(inputs={}, cfg={})

        assert isinstance(result, StageResult)
        assert "chunks" in result.outputs

    @patch("spindle.eval_bridge._build_kos_service")
    def test_full_pipeline_stage_names(self, mock_build_kos):
        mock_build_kos.return_value = MagicMock()

        stages = get_pipeline_definition(include_kos=True, include_generation=True)

        names = [sd.name for sd in stages]
        assert names == [
            "preprocessing",
            "kos_extraction",
            "ontology_synthesis",
            "retrieval",
            "generation",
        ]

    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_kos_unavailable_skips_kos_stages(self, _mock_kos):
        stages = get_pipeline_definition(include_kos=True, include_generation=True)

        names = [sd.name for sd in stages]
        assert "kos_extraction" not in names
        assert "ontology_synthesis" not in names
        assert "retrieval" not in names

    @patch("spindle.eval_bridge._build_kos_service")
    def test_kos_extraction_has_gate(self, mock_build_kos):
        mock_build_kos.return_value = MagicMock()

        stages = get_pipeline_definition(include_kos=True, include_generation=False)

        kos_sd = [sd for sd in stages if sd.name == "kos_extraction"][0]
        assert kos_sd.gate is not None
        # Gate passes when orphan_concept_ratio <= 0.3
        assert kos_sd.gate({"orphan_concept_ratio": 0.2}) is True
        assert kos_sd.gate({"orphan_concept_ratio": 0.5}) is False
        # Default (missing key) should fail gate
        assert kos_sd.gate({}) is False

    @patch("spindle.eval_bridge._build_kos_service")
    def test_stage_defs_have_input_keys(self, mock_build_kos):
        mock_build_kos.return_value = MagicMock()

        stages = get_pipeline_definition(include_kos=True, include_generation=True)

        kos_sd = [sd for sd in stages if sd.name == "kos_extraction"][0]
        assert kos_sd.input_keys == {"chunks": "preprocessing.chunks"}

        gen_sd = [sd for sd in stages if sd.name == "generation"][0]
        assert gen_sd.input_keys == {"contexts": "retrieval.contexts"}

    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_discovery_isinstance_list(self, _mock_kos):
        """Runner does isinstance(result, list) — must pass."""
        result = get_pipeline_definition()
        assert isinstance(result, list)
