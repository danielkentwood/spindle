"""Observers for ingestion performance metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

from spindle.ingestion.types import IngestionEvent


@dataclass
class PerformanceTracker:
    """Collect stage timing information from ingestion events."""

    stage_totals: Dict[str, float] = None
    stage_counts: Dict[str, int] = None

    def __post_init__(self) -> None:
        self.stage_totals = defaultdict(float)
        self.stage_counts = defaultdict(int)

    def __call__(self, event: IngestionEvent) -> None:
        if event.name == "stage_complete":
            stage = event.payload.get("stage")
            duration = float(event.payload.get("duration_ms", 0.0))
            self.stage_totals[stage] += duration
            self.stage_counts[stage] += 1

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            stage: {
                "duration_ms": total,
                "count": self.stage_counts.get(stage, 0),
                "avg_ms": total / self.stage_counts.get(stage, 1),
            }
            for stage, total in self.stage_totals.items()
        }

