## Environment Setup

Spindle reads model and provider credentials from environment variables.

## Required

- `ANTHROPIC_API_KEY`: required for extraction and synthesis flows that call Anthropic models.

## Optional

- `OPENAI_API_KEY`: OpenAI embeddings fallback.
- `HF_API_KEY`: HuggingFace API embeddings fallback.
- `GEMINI_API_KEY`: Google embeddings support.
- Langfuse variables (if you use Langfuse for tracing).

## Local setup

Create `.env` in repo root:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
HF_API_KEY=hf_...
GEMINI_API_KEY=AIza...
```

Then load it in your shell (or rely on your runner/IDE):

```bash
set -a; source .env; set +a
```

## Auth behavior notes

- `SpindleExtractor` can auto-detect auth from environment.
- `SpindleConfig.with_auto_detected_llm(...)` auto-detects LLM credentials at config construction time.
- If no compatible credentials are available, extraction calls fail at runtime.
