"""Tests for process graph extraction helpers."""

from __future__ import annotations

from typing import Callable

import pytest

from spindle import extractor
from spindle.baml_client.types import (
    EvidenceSpan,
    ProcessDependency,
    ProcessExtractionIssue,
    ProcessExtractionResult,
    ProcessGraph,
    ProcessStep,
    ProcessStepType,
)


def _make_step(
    step_id: str,
    title: str,
    summary: str,
    step_type: ProcessStepType = ProcessStepType.ACTIVITY,
    actors: list[str] | None = None,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    prerequisites: list[str] | None = None,
) -> ProcessStep:
    return ProcessStep(
        step_id=step_id,
        title=title,
        summary=summary,
        step_type=step_type,
        actors=actors or [],
        inputs=inputs or [],
        outputs=outputs or [],
        duration=None,
        prerequisites=prerequisites or [],
        evidence=[EvidenceSpan(text=f"{title} evidence")],
    )


def _make_dependency(
    source: str,
    target: str,
    relation: str = "precedes",
    condition: str | None = None,
) -> ProcessDependency:
    return ProcessDependency(
        from_step=source,
        to_step=target,
        relation=relation,
        condition=condition,
        evidence=[EvidenceSpan(text=f"{source} to {target}")],
    )


def _fake_result(graph: ProcessGraph) -> ProcessExtractionResult:
    return ProcessExtractionResult(
        status="process_found",
        graph=graph,
        reasoning="1. Generated for testing.",
        issues=[],
    )


def _patch_extractor(monkeypatch: pytest.MonkeyPatch, factory: Callable[[], ProcessExtractionResult]) -> None:
    monkeypatch.setattr(
        extractor.b,
        "ExtractProcessGraph",
        lambda **_: factory(),
    )


def test_extract_process_graph_merges_existing_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    existing_graph = ProcessGraph(
        process_name="Release Pipeline",
        scope="Deploying application changes",
        primary_goal="Ship stable builds",
        start_step_ids=["plan"],
        end_step_ids=["deploy"],
        steps=[
            _make_step("plan", "Plan Work", "Plan upcoming tasks."),
            _make_step("develop", "Develop Feature", "Implement feature.", prerequisites=["plan"]),
        ],
        dependencies=[_make_dependency("plan", "develop")],
        notes=["Legacy notes"],
    )

    updated_graph = ProcessGraph(
        process_name="Release Pipeline",
        scope=None,
        primary_goal="Ship stable builds quickly",
        start_step_ids=[],
        end_step_ids=[],
        steps=[
            _make_step("develop", "Develop Feature", "Updated description.", actors=["engineer"]),
            _make_step("review", "Code Review", "Review changes.", prerequisites=["develop"]),
        ],
        dependencies=[
            _make_dependency("develop", "review"),
        ],
        notes=["Include code review outcomes"],
    )

    _patch_extractor(monkeypatch, lambda: _fake_result(updated_graph))

    result = extractor.extract_process_graph(
        text="Develop feature then request code review.",
        existing_graph=existing_graph,
    )

    assert result.graph is not None
    step_ids = {step.step_id for step in result.graph.steps}
    assert step_ids == {"plan", "develop", "review"}
    assert set(result.graph.start_step_ids) == {"plan"}
    assert set(result.graph.end_step_ids) == {"review"}
    develop_step = next(step for step in result.graph.steps if step.step_id == "develop")
    assert "engineer" in develop_step.actors
    assert "Legacy notes" in result.graph.notes
    assert result.issues == []


def test_extract_process_graph_flags_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    cyclic_graph = ProcessGraph(
        process_name="Circular Flow",
        scope=None,
        primary_goal="Illustrate cycle detection",
        start_step_ids=[],
        end_step_ids=[],
        steps=[
            _make_step("a", "Start", "Beginning."),
            _make_step("b", "Loop", "Loops back.", prerequisites=["a"]),
        ],
        dependencies=[
            _make_dependency("a", "b"),
            _make_dependency("b", "a"),
        ],
        notes=[],
    )

    existing_issue = ProcessExtractionIssue(
        code="llm_warning",
        message="Ambiguous phrasing detected.",
        related_step_ids=["a"],
    )

    def factory() -> ProcessExtractionResult:
        return ProcessExtractionResult(
            status="process_found",
            graph=cyclic_graph,
            reasoning="1. Cycle detected in text.",
            issues=[existing_issue],
        )

    _patch_extractor(monkeypatch, factory)

    result = extractor.extract_process_graph(
        text="Step A leads to Step B which loops back to Step A.",
    )

    codes = {issue.code for issue in result.issues}
    assert "llm_warning" in codes
    assert "cycle_detected" in codes
    cycle_issue = next(issue for issue in result.issues if issue.code == "cycle_detected")
    assert set(cycle_issue.related_step_ids) == {"a", "b"}

