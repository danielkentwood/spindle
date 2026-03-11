"""Smoke tests for NoOpTracker."""

from spindle.tracking import NoOpTracker


class TestNoOpTracker:
    def test_log_metric(self):
        tracker = NoOpTracker()
        tracker.log_metric("accuracy", 0.95)

    def test_log_metrics(self):
        tracker = NoOpTracker()
        tracker.log_metrics({"accuracy": 0.95, "f1": 0.88})

    def test_log_param(self):
        tracker = NoOpTracker()
        tracker.log_param("model", "claude-sonnet")

    def test_log_params(self):
        tracker = NoOpTracker()
        tracker.log_params({"model": "claude-sonnet", "temp": 0.7})

    def test_log_event(self):
        tracker = NoOpTracker()
        tracker.log_event("extraction", "start", {"doc_count": 5})

    def test_log_event_no_payload(self):
        tracker = NoOpTracker()
        tracker.log_event("extraction", "start")

    def test_log_artifact(self):
        tracker = NoOpTracker()
        tracker.log_artifact("/tmp/output.json")

    def test_start_stage_context_manager(self):
        tracker = NoOpTracker()
        with tracker.start_stage("preprocessing") as ctx:
            assert ctx is None  # yields None

    def test_end_run(self):
        tracker = NoOpTracker()
        tracker.end_run()
