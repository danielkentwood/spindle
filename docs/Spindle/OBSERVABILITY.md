## Observability

Spindle emits structured service events across core subsystems.

## What is tracked

- Extraction lifecycle and LLM usage metrics.
- Preprocessing stage timings and counts.
- Graph store and entity-resolution operational events.

## Components

- `spindle.observability.events`: event models.
- `spindle.observability.storage`: optional SQLite-backed recorder.

## Usage pattern

Most runtime components accept a tracker argument and default to a no-op tracker.
Pass a concrete tracker/recorder when you need persistent event logs for
debugging, audits, or performance analysis.

## Operational guidance

- Keep event logging on in non-trivial pipeline runs.
- Persist logs in CI or staging when validating regressions.
