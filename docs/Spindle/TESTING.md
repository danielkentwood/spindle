## Testing

All test commands should be run with `uv`.

## Test suites

- Unit tests: fast, mostly isolated from external services.
- Integration tests: marked `integration`, require live credentials/services.

## Common commands

```bash
# Full default run
uv run pytest tests/

# Unit-only
uv run pytest tests/ -m "not integration"

# Integration-only
uv run pytest tests/ -m integration

# Single file
uv run pytest tests/test_extractor.py -v

# Single test
uv run pytest tests/test_helpers.py::TestFindSpanIndices::test_exact_match -v
```

## Coverage

Default pytest config already enables coverage reporting for `spindle`.

```bash
uv run pytest tests/ --cov=spindle --cov-report=term-missing
```

## Notes

- If API-key-backed tests fail, verify `ANTHROPIC_API_KEY`.
- Prefer running targeted tests while iterating on one subsystem.
