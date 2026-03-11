"""Tests for eval_bridge: get_pipeline_definition, StageDef, PipelineDefinition."""

from unittest.mock import MagicMock, patch
import pytest

from spindle.eval_bridge import (
    PipelineDefinition,
    StageDef,
    _get,
    get_pipeline_definition,
)


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


class TestPipelineDefinition:
    def test_defaults(self):
        defn = PipelineDefinition()
        assert defn.stages == []
        assert defn.metadata == {}


class TestGetPipelineDefinition:
    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_minimal_pipeline_no_kos_no_generation(self, _mock_kos):
        defn = get_pipeline_definition(include_kos=False, include_generation=False)

        assert len(defn.stages) == 1
        assert defn.stages[0].name == "preprocessing"
        assert defn.metadata["stage_count"] == 1

    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_ontology_threads_to_generation(self, _mock_kos):
        mock_ontology = MagicMock()
        defn = get_pipeline_definition(
            include_kos=False,
            include_generation=True,
            ontology=mock_ontology,
        )

        gen_stage_def = [sd for sd in defn.stages if sd.name == "generation"][0]
        assert gen_stage_def.stage._ontology is mock_ontology

    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_tracker_threads_to_all_stages(self, _mock_kos):
        mock_tracker = MagicMock()
        defn = get_pipeline_definition(
            include_kos=False,
            include_generation=True,
            tracker=mock_tracker,
        )

        for sd in defn.stages:
            assert sd.stage._tracker is mock_tracker

    @patch("spindle.eval_bridge._build_kos_service")
    def test_graph_vector_store_thread_to_retrieval(self, mock_build_kos):
        mock_kos = MagicMock()
        mock_build_kos.return_value = mock_kos
        mock_graph = MagicMock()
        mock_vector = MagicMock()

        defn = get_pipeline_definition(
            include_kos=True,
            include_generation=False,
            graph_store=mock_graph,
            vector_store=mock_vector,
        )

        retrieval_sd = [sd for sd in defn.stages if sd.name == "retrieval"][0]
        assert retrieval_sd.stage._graph is mock_graph
        assert retrieval_sd.stage._vector is mock_vector

    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_metadata_stage_names(self, _mock_kos):
        defn = get_pipeline_definition(
            include_kos=False,
            include_generation=True,
        )

        assert defn.metadata["stage_names"] == ["preprocessing", "generation"]
        assert defn.metadata["stage_count"] == 2

    @patch("spindle.eval_bridge._build_kos_service")
    def test_full_kos_pipeline_stages(self, mock_build_kos):
        mock_kos = MagicMock()
        mock_build_kos.return_value = mock_kos

        defn = get_pipeline_definition(
            include_kos=True,
            include_generation=True,
        )

        names = [sd.name for sd in defn.stages]
        assert names == [
            "preprocessing",
            "kos_extraction",
            "ontology_synthesis",
            "retrieval",
            "generation",
        ]

    @patch("spindle.eval_bridge._build_kos_service")
    def test_kos_unavailable_skips_kos_stages(self, mock_build_kos):
        mock_build_kos.return_value = None

        defn = get_pipeline_definition(include_kos=True, include_generation=True)

        names = [sd.name for sd in defn.stages]
        assert "kos_extraction" not in names
        assert "ontology_synthesis" not in names
        assert "retrieval" not in names

    @patch("spindle.eval_bridge._build_kos_service")
    def test_stage_defs_have_input_keys(self, mock_build_kos):
        mock_kos = MagicMock()
        mock_build_kos.return_value = mock_kos

        defn = get_pipeline_definition(include_kos=True, include_generation=True)

        kos_sd = [sd for sd in defn.stages if sd.name == "kos_extraction"][0]
        assert kos_sd.input_keys == {"chunks": "preprocessing.chunks"}

        gen_sd = [sd for sd in defn.stages if sd.name == "generation"][0]
        assert gen_sd.input_keys == {"contexts": "retrieval.contexts"}

    @patch("spindle.eval_bridge._build_kos_service", return_value=None)
    def test_stage_def_metrics_and_gate_defaults(self, _mock_kos):
        defn = get_pipeline_definition(include_kos=False, include_generation=True)

        for sd in defn.stages:
            assert sd.metrics == []
            assert sd.gate is None
