"""Tracker protocol and NoOpTracker for standalone spindle usage.

When running under spindle-eval, the runner injects a real tracker
(MLflowTracker, FileTracker, etc.). When running standalone,
NoOpTracker routes all events to Python's logging module at DEBUG
level — silent by default, visible when opted in.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger("spindle.tracking")


class NoOpTracker:
    """Default tracker when not running under spindle-eval.

    Routes all events/metrics to Python logging at DEBUG level.
    Satisfies the spindle_eval Tracker protocol when spindle-eval is installed.
    """

    def log_metric(self, key: str, value: float) -> None:
        logger.debug("metric %s=%s", key, value)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        logger.debug("metrics %s", metrics)

    def log_param(self, key: str, value: Any) -> None:
        logger.debug("param %s=%s", key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        logger.debug("params %s", params)

    def log_event(
        self,
        service: str,
        name: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        logger.debug(
            "event %s/%s %s",
            service,
            name,
            json.dumps(payload, default=str) if payload else "{}",
        )

    def log_artifact(self, path: str) -> None:
        logger.debug("artifact %s", path)

    @contextmanager
    def start_stage(self, name: str) -> Generator[None, None, None]:
        logger.debug("stage_start %s", name)
        try:
            yield
        finally:
            logger.debug("stage_end %s", name)

    def end_run(self) -> None:
        logger.debug("run_end")
