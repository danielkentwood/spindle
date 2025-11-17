# Environment Setup

This guide explains how to configure your Spindle development environment, install the required tooling, and provide the credentials for both direct API usage and Google Vertex AI integrations.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Install uv](#install-uv)
4. [Create and Use the Virtual Environment](#create-and-use-the-virtual-environment)
5. [Manage Dependencies](#manage-dependencies)
6. [Configure Environment Variables](#configure-environment-variables)
7. [Vertex AI Setup](#vertex-ai-setup)
8. [Troubleshooting](#troubleshooting)
9. [Additional Resources](#additional-resources)
10. [Support](#support)

## Overview

- ⚙️ Spindle uses [uv](https://github.com/astral-sh/uv) for Python environment and package management.
- 🔐 Credentials are supplied via environment variables or a `.env` file (auto-loaded by `python-dotenv` in demos).
- ☁️ Vertex AI integration is supported for accessing Anthropic Claude and Google Gemini models through Google Cloud Platform (GCP).
- 🔄 Spindle automatically prefers Vertex AI when available and falls back to direct API keys when not.

## Prerequisites

1. macOS, Linux, or Windows shell (instructions assume Unix-like shell).
2. Python 3.9+ (managed via `.python-version` if present).
3. `uv` installed and available on your `PATH`.
4. (Optional) Access to GCP with Vertex AI enabled if you plan to use managed models.

## Install uv

UV is a fast, reliable drop-in replacement for pip, written in Rust. Benefits include 10–100x faster installs, better dependency resolution, and simplified workflows for AI agents.

### Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew
brew install uv
```

After installation, restart your terminal or run:

```bash
source "$HOME/.cargo/env"
```

Verify the installation:

```bash
uv --version
```

## Create and Use the Virtual Environment

### 1. Create the Environment

```bash
# Navigate to the project root
cd /Users/thalamus/Repos/spindle

# Create the virtual environment (uses Python from .python-version)
uv venv
```

This creates `.venv/` with the environment using Python 3.9+ (or the version specified in `.python-version` if present).

### 2. Install Project Dependencies

```bash
# Install everything (production + dev) in editable mode
uv pip install -e ".[dev]"

# Production-only install
uv pip install -e .

# Install from requirements files
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## Manage Dependencies

### Running Commands

```bash
# Always prefix Python commands with 'uv run'
uv run python demos/example.py

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=spindle --cov-report=html
```

### Adding New Dependencies

```bash
# Install a new package
uv pip install some-package

# Add to pyproject.toml or requirements.txt to keep the dependency pinned
```

### Development Dependencies

```bash
# Example: install a dev-only dependency
uv pip install pytest-asyncio

# Add to pyproject.toml under [project.optional-dependencies.dev]
```

### Updating Dependencies

```bash
# Update a specific package
uv pip install --upgrade some-package

# Update from a requirements file (use with caution)
uv pip install --upgrade -r requirements.txt
```

### Inspecting the Environment

```bash
uv pip list
uv pip show spindle-kg
```

## Configure Environment Variables

Spindle expects API keys and cloud credentials to be provided via environment variables. The simplest approach is to create a `.env` file in the project root (auto-loaded by `python-dotenv` in demos and examples).

### Create a `.env` File

```bash
# docs/.env example
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Getting an Anthropic API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in.
3. Navigate to API Keys.
4. Create a new API key.
5. Copy the key into your `.env` file.

### Optional Keys

Add keys for the providers you plan to use:

```
# OpenAI embeddings (VectorStore helpers)
OPENAI_API_KEY=your_openai_api_key_here

# Hugging Face Inference API
HF_API_KEY=your_hugging_face_key_here  # alias: HUGGINGFACE_API_KEY

# Gemini embeddings
GEMINI_API_KEY=your_gemini_key_here
```

### Security Note

Never commit your `.env` file to version control. The `.env` file is already listed in `.gitignore`.

## Vertex AI Setup

This section consolidates the Vertex AI setup guide for using Anthropic Claude and Google Gemini models through GCP.

### Overview

- Vertex AI is the recommended path when available (leverages GCP billing, IAM, and VPC security).
- Access both Anthropic Claude models and Gemini models through a single interface.
- Spindle auto-detects Vertex AI credentials; falls back to direct API keys.

### Benefits

- ✅ Unified GCP billing and credits.
- ✅ Enterprise-grade security via IAM and networking controls.
- ✅ Access to Claude and Gemini models with regional control.
- ✅ Automatic fallback to direct API keys if Vertex AI is unavailable.

### Prerequisites

1. A GCP project with Vertex AI enabled.
2. Enable the following APIs:
   - Vertex AI (`aiplatform.googleapis.com`)
   - Cloud services (`cloudservices.googleapis.com`)
3. (Optional) Request access to specific models in the Vertex AI Model Garden.

### Request Model Access

#### Anthropic Claude Models

1. Go to the [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden).
2. Search for “Claude”.
3. Request access to the models you need, e.g.:
   - Claude 2.1
   - Claude 3 Opus
   - Claude 3.5 Haiku

#### Google Gemini Models

Most Gemini models are generally available once Vertex AI is enabled.

### Check Regional Availability

Model availability varies by region. As of this writing:

- 🟡 Claude Sonnet: `us-east5`, `us-east1`, `europe-west1`, and others
- 🟢 Gemini models: `us-central1`, `us-east1`, `europe-west1`, and others

Consult the [Vertex AI locations documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/locations) for current details.

### Authentication Setup

#### Method 1: `gcloud auth application-default login` (Local Development)

```bash
# Authenticate with your Google account
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

#### Method 2: Service Account (Recommended for Production)

```bash
# Create a service account
gcloud iam service-accounts create spindle-vertex-ai \
    --display-name "Spindle Vertex AI Service Account"
```

```bash
# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:spindle-vertex-ai@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

```bash
# Download a key
gcloud iam service-accounts keys create "$HOME/spindle-vertex-key.json" \
    --iam-account=spindle-vertex-ai@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

```bash
# Point GOOGLE_APPLICATION_CREDENTIALS at the key file
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/spindle-vertex-key.json"
```

#### Method 3: Workload Identity (GCP-hosted workloads)

Use [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) for GKE and Cloud Run so service accounts are attached automatically without keys.

### Configuration

Set environment variables directly or via `.env`:

```bash
# GCP Project configuration
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-east5"  # choose a supported region

# Authentication (choose one)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

# Optional fallbacks
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

Example `.env` snippet:

```
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-east5
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Optional fallbacks
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Usage Examples

**Auto-Detection**

```python
from spindle import SpindleExtractor, create_ontology

ontology = create_ontology(
    entity_types=[
        {"name": "Person", "description": "A human being"},
        {"name": "Organization", "description": "A company"},
    ],
    relation_types=[
        {
            "name": "works_at",
            "description": "Employment",
            "domain": "Person",
            "range": "Organization",
        },
    ],
)

extractor = SpindleExtractor(ontology)
result = extractor.extract("Alice works at TechCorp.", source_name="example")
for triple in result.triples:
    print(f"{triple.subject.name} - {triple.predicate} - {triple.object.name}")
```

**Explicit Vertex AI Configuration**

```python
from spindle import SpindleExtractor
from spindle.llm_config import LLMConfig, AuthMethod

config = LLMConfig(
    gcp_project_id="my-project-123",
    gcp_region="us-east5",
    preferred_auth_method=AuthMethod.VERTEX_AI,
)

extractor = SpindleExtractor(
    ontology=ontology,
    llm_config=config,
    auto_detect_auth=False,
)

result = extractor.extract(text, source_name="example")
```

**Anthropic Vertex AI SDK**

```python
from spindle.llm_config import LLMConfig, get_anthropic_vertex_client

config = LLMConfig(
    gcp_project_id="my-project-123",
    gcp_region="us-east5",
)

client = get_anthropic_vertex_client(config)
message = client.messages.create(
    model="claude-sonnet-4020250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, Claude!"}],
)

print(message.content[0].text)
```

**Model-as-a-Service (MAAS) API**

```python
import requests
from spindle.llm_config import LLMConfig, create_vertex_maas_request

config = LLMConfig(
    gcp_project_id="my-project-123",
    gcp_region="us-east5",
)

request_config = create_vertex_maas_request(
    config=config,
    model_name="claude-sonnet-4020250514",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)

response = requests.post(
    request_config["url"],
    headers=request_config["headers"],
    json=request_config["payload"],
)

print(response.json())
```

**Detect Available Authentication**

```python
from spindle.llm_config import detect_available_auth, Provider

config = detect_available_auth()
print("Available authentication methods:")
for method in config.available_auth_methods:
    print(f"- {method.value}")

print("\nPreferred method:", config.preferred_auth_method.value)

print("\nProvider routing:")
for provider in [Provider.ANTHROPIC, Provider.OPENAI, Provider.GOOGLE]:
    auth = config.get_auth_method_for_provider(provider)
    print(f"{provider.value}: {auth.value if auth else 'Not available'}")
```

### Best Practices

- Use auto-detection unless you must override the configuration.
- Keep credentials in environment variables, not hard-coded in code.
- Prefer service accounts in production; use Workload Identity on GCP infrastructure.
- Configure API key fallbacks so Spindle continues operating if Vertex AI becomes unavailable.
- Choose GCP regions that match your latency and compliance requirements.
- Monitor Vertex AI quotas, budgets, and usage in Cloud Monitoring.

## Troubleshooting

### UV Environment Issues

```bash
# Recreate the virtual environment
rm -rf .venv
uv venv
uv pip install -e ".[dev]"
```

```bash
# Verify Python version
uv run python --version  # Should show 3.9 or higher
```

```bash
# Clear uv cache and reinstall if dependencies behave unexpectedly
uv cache clean
rm -rf .venv
uv venv
uv pip install -e ".[dev]"
```

### Vertex AI Common Issues

**Permission denied**

```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="user:your-email@example.com" \
    --role="roles/aiplatform.user"
```

**Model not found**

- Ensure you requested access via Model Garden.
- Confirm the model is available in your region.
- Wait 10–15 minutes after requesting access.

**Invalid authentication credentials**

- Verify `GOOGLE_APPLICATION_CREDENTIALS` points to a valid key.
- Re-run `gcloud auth application-default login` if using user credentials.
- Ensure service account keys have not expired.

**Region not supported**

- Check [Vertex AI locations](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/locations).
- Update `GCP_REGION` to a supported region (e.g., `us-east5` for Claude Sonnet 4).

**Enable debug logging**

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spindle.llm_config")
logger.setLevel(logging.DEBUG)

from spindle.llm_config import detect_available_auth

config = detect_available_auth()
```

**Test authentication programmatically**

```python
from spindle.llm_config import LLMConfig, get_anthropic_vertex_client

try:
    config = LLMConfig(
        gcp_project_id="your-project-id",
        gcp_region="us-east5",
    )

    client = get_anthropic_vertex_client(config)
    response = client.messages.create(
        model="claude-sonnet-4020250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hi!"}],
    )

    print("Vertex AI authentication successful!")
    print("Response:", response.content[0].text)
except Exception as exc:
    print("Authentication failed:", exc)
```

## Additional Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Anthropic on Vertex AI](https://docs.anthropic.com/en/api/cloud-on-vertex-ai)
- [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/docs/start/explore-models)
- [GCP IAM Best Practices](https://cloud.google.com/iam/docs/best-practices)
- [Spindle Example Notebooks](https://spindle.ai/notebooks)

## Support

- Review this guide and the included troubleshooting steps.
- Check the example notebook at `spindle/notebooks/example_auth.ipynb`.
- File an issue on GitHub with the error message, steps to reproduce, and credential context (service account or user role).

