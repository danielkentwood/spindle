# Vertex AI Setup Guide
You, 14 hours ago · Enhance authentication and configuration for LLM access
This guide covers how to set up and use Google Cloud Platform's Vertex AI with Spindle for accessing Anthropic Claude models and Google Gemini models.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Authentication Setup](#authentication-setup)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

## Overview

Spindle supports multiple ways to access LLM models:

- **Vertex AI (Recommended when available)**: Access Anthropic Claude and Google Gemini models through GCP Vertex AI.
- **Direct API**: Use API keys directly from Anthropic, OpenAI, etc.

### Benefits of Using Vertex AI

- ✅ **GCP Integration**: Uses existing GCP credits and unified billing
- ✅ **Enterprise Security**: Leverage GCP IAM, VPC, and compliance features
- ✅ **Model Access**: Access both Claude (Anthropic) and Gemini (Google) models
- ✅ **Automatic Preference**: Spindle automatically prefers Vertex AI when credentials are available
- ✅ **Fallback Support**: Automatically falls back to direct API keys if Vertex AI is unavailable

## Prerequisites

1. Create a GCP project (or use an existing one)
2. Enable Vertex AI for the project
3. Enable the following APIs:
   - Vertex AI
   - Cloud services enable aipatform.googleapis.com

## Step-by-Step Setup

### 1. Request Model Access

#### For Anthropic Claude Models

1. Go to the Vertex AI Model Garden(https://console.cloud.google.com/vertex-ai/model-garden)
2. Search for "Claude"
3. Request access to the models you need:
   - Claude 2.1
   - Claude 3 Opus
   - Claude 3.5 Haiku

#### For Google Gemini Models

Gemini models are generally available by default in Vertex AI-enabled projects.

### 3. Check Regional Availability

Different models are available in different regions. As of now:

- 🟡 **Claude Sonnet**: `us-east5`, `us-east1`, `europe-west1`, and others
- 🟢 **Gemini models**: `us-central1`, `us-east1`, `europe-west1`, and others

Check the [Vertex AI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/locations) for the latest availability.

## Authentication Setup

Spindle supports multiple authentication methods. Choose the one that fits your setup:

### Method 1: gcloud auth Application Default Credentials (Recommended for Local Development)

```bash
# Authenticate with your Google account
gcloud auth application-default login
```

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### Method 2: Service Account (Recommended for Production)

```bash
# Create a service account
gcloud iam service-accounts create spindle-vertex-ai \
    --display-name "Spindle Vertex AI Service Account"
```

```bash
# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:spindle-vertex-ai@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

```bash
# Download the key file:
gcloud iam service-accounts keys create ~/spindle-vertex-key.json \
    --iam-account=spindle-vertex-ai@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

```bash
# Set the environment variable:
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/spindle-vertex-key.json"
```

### Method 3: Workload Identity (for GKE/Cloud Run)

If running on GCP infrastructure, use [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity) for automatic credential management.

## Configuration

### Environment Variables

Spindle uses the following environment variables for Vertex AI configuration:

```bash
# GCP Project configuration
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-east5"  # or your preferred region

# Authentication (choose one)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
# (service account key file)
```

```bash
# Optional: Fallback direct API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### Using a `.env` File

Create a `.env` file in your project root:

```bash
# .env
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-east5
GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Optional fallbacks
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

Then load it in your Python code:

```python
from dotenv import load_dotenv
load_dotenv()

from spindle import SpindleExtractor
# Auto-detection will pick up the environment variables
extractor = SpindleExtractor(ontology)
```

## Usage Examples

### Auto-Detection (Simplest)

Spindle automatically detects and uses Vertex AI when credentials are available:

```python
from spindle import SpindleExtractor, create_ontology

# Define your ontology
entity_types = [
    {"name": "Person", "description": "A human being"},
    {"name": "Organization", "description": "A company"},
]

relation_types = [
    {"name": "works_at", "description": "Employment",
     "domain": "Person", "range": "Organization"},
]

ontology = create_ontology(entity_types, relation_types)

# Create extractor – automatically detects Vertex AI credentials
extractor = SpindleExtractor(ontology)

# Extract triples – uses Vertex AI if available, falls back to direct API
result = extractor.extract("Alice works at TechCorp.", source_name="example")

for triple in result.triples:
    print(f"{triple.subject.name} - {triple.predicate} - {triple.object.name}")
```

### Explicit Vertex AI Configuration

```python
from spindle import SpindleExtractor
from spindle.llm_config import LLMConfig, AuthMethod

# Explicitly configure Vertex AI
config = LLMConfig(
    gcp_project_id="my-project-123",
    gcp_region="us-east5",
    preferred_auth_method=AuthMethod.VERTEX_AI,
)

# Create extractor with explicit config
extractor = SpindleExtractor(
    ontology=ontology,
    llm_config=config,
    auto_detect_auth=False  # Don't auto-detect, use provided config
)

# Extract triples using Vertex AI
result = extractor.extract(text, source_name="example")
```

### Using Anthropic Vertex SDK Directly

For more control, use the Anthropic Vertex AI SDK directly:

```python
from spindle.llm_config import LLMConfig, get_anthropic_vertex_client

config = LLMConfig(
    gcp_project_id="my-project-123",
    gcp_region="us-east5",
)

# Get Anthropic Vertex client
client = get_anthropic_vertex_client(config)

# Make requests
message = client.messages.create(
    model="claude-sonnet-4020250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude!"},
    ],
)

print(message.content[0].text)
```

### Using MAAS API

Alternative approach using the Model-as-a-Service API:

```python
import requests
from spindle.llm_config import LLMConfig, create_vertex_maas_request

config = LLMConfig(
    gcp_project_id="my-project-123",
    gcp_region="us-east5",
)

# Create API request configuration
request_config = create_vertex_maas_request(
    config=config,
    model_name="claude-sonnet-4020250514",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100,
)

# Make the request
response = requests.post(
    request_config["url"],
    headers=request_config["headers"],
    json=request_config["payload"],
)

result = response.json()
print(result)
```

### Checking Available Authentication

Inspect what authentication methods are available:

```python
from spindle.llm_config import detect_available_auth, Provider

# Detect available credentials
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

## Troubleshooting

### Common Issues

#### 1. "Permission denied" errors

**Solution**: Ensure your service account or user has the `roles/aiplatform.user` role:

```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="user:your-email@example.com" \
    --role="roles/aiplatform.user"
```

#### 2. "Model not found" errors

**Solution**:
- Check that you've requested access to the model in Vertex AI Model Garden
- Verify the model is available in your chosen region
- Wait 10–15 minutes after requesting access for it to propagate

#### 3. "Invalid authentication credentials"

**Solution**:
- Verify `GOOGLE_APPLICATION_CREDENTIALS` points to a valid key file
- Run `gcloud auth application-default login` for user credentials
- Check that credentials haven't expired

#### 4. "Region not supported for model"

**Solution**:
- Check model availability: [Vertex AI Locations](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/locations)
- Update `GCP_REGION` to a supported region
- For Claude Sonnet 4, use `us-east5` or `europe-west1`

### Debug Mode

Enable detailed logging to diagnose issues:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spindle.llm_config")
logger.setLevel(logging.DEBUG)

from spindle.llm_config import detect_available_auth

config = detect_available_auth()
# This will print detailed detection information
```

### Testing Authentication

Quick test to verify Vertex AI access:

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

except Exception as e:
    print("Authentication failed:", e)
```

## Best Practices

### 1. Use Auto-Detection

Let Spindle automatically detect and choose the best authentication method:

```python
extractor = SpindleExtractor(ontology)
# Good: Auto-detection handles everything
```

```python
# Also good: Explicit only when needed
extractor = SpindleExtractor(ontology, llm_config=my_specific_config)
```

### 2. Set Environment Variables Properly

Use environment variables for configuration, not hardcoded values:

```python
# Good: Uses environment variables
import os
config = LLMConfig(
    gcp_project_id=os.getenv("GCP_PROJECT_ID"),
    gcp_region=os.getenv("GCP_REGION"),
)
```

```python
# Bad: Hardcoded credentials
config = LLMConfig(
    gcp_project_id="hardcoded-project-123",  # Don't do this
)
```

### 3. Use Service Accounts in Production

**Development**: User credentials via `gcloud auth application-default login`
**Production**: Service accounts with minimal required permissions
**GCP Infrastructure**: Workload Identity for automatic credential management

### 4. Handle Fallbacks Gracefully

Spindle automatically falls back to direct API keys when Vertex AI is unavailable. Ensure you have fallback credentials configured:

```bash
# Primary: Vertex AI
export GCP_PROJECT_ID="..."
export GCP_REGION="..."

# Fallback: Direct API
export ANTHROPIC_API_KEY="..."
```

### 5. Choose the Right Region

- Use regions close to your users for lower latency
- Check model availability in your chosen region
- Consider data residency requirements for your use case

### 6. Monitor Costs and Quotas

- Set up [billing alerts](https://cloud.google.com/billing/docs/how-to/budgets) in GCP
- Monitor [Vertex AI quotas](https://cloud.google.com/vertex-ai/docs/quota)
- Use [Cloud Monitoring](https://cloud.google.com/monitoring) to track API usage

## Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Anthropic on Vertex AI](https://docs.anthropic.com/en/api/cloud-on-vertex-ai)
- [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/docs/start/explore-models)
- [GCP IAM Best Practices](https://cloud.google.com/iam/docs/best-practices)
- [Spindle Example Notebooks](https://spindle.ai/notebooks)

## Support

- Check this troubleshooting guide
- Review the `example notebook` in `spindle/notebooks/example_auth.ipynb`
- File an issue on GitHub with:
  - Error message
  - Steps to reproduce
  - Service account or user role name

