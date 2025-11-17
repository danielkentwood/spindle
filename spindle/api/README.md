# Spindle REST API

A FastAPI-based REST API for Spindle's knowledge graph extraction and management services.

## Features

- **Document Ingestion**: Process and ingest documents with streaming progress
- **Triple Extraction**: Extract knowledge graph triples from text with batch and streaming support
- **Ontology Management**: Automatic ontology recommendation and extension
- **Entity Resolution**: Deduplicate entities and relationships in knowledge graphs
- **Process Extraction**: Extract process graphs and workflows from text
- **Session Management**: Stateful operations with session-based context

## Installation

Install the required dependencies:

```bash
uv pip install fastapi uvicorn python-multipart
```

Or if using pip:

```bash
pip install fastapi uvicorn[standard] python-multipart
```

## Running the API

### Using the CLI command:

```bash
spindle-api
```

### Using Python:

```python
from spindle.api.main import main

if __name__ == "__main__":
    main()
```

### Using uvicorn directly:

```bash
uvicorn spindle.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Documentation

Once the server is running, visit:

- **Interactive API docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API docs (ReDoc)**: http://localhost:8000/redoc

## Quick Start

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Create a Session (Optional for stateful operations)

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{"name": "my-session"}'
```

### 3. Extract Triples (Stateless)

```bash
curl -X POST http://localhost:8000/api/extraction/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice works at TechCorp as an engineer.",
    "source_name": "example-doc",
    "ontology_scope": "balanced"
  }'
```

### 4. Recommend an Ontology

```bash
curl -X POST http://localhost:8000/api/ontology/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alice works at TechCorp in San Francisco.",
    "scope": "balanced"
  }'
```

## API Endpoints

### Health & Status

- `GET /health` - Health check

### Sessions

- `POST /api/sessions` - Create a new session
- `GET /api/sessions` - List all sessions
- `GET /api/sessions/{session_id}` - Get session info
- `DELETE /api/sessions/{session_id}` - Delete session
- `PUT /api/sessions/{session_id}/ontology` - Update session ontology
- `PUT /api/sessions/{session_id}/config` - Update session config

### Ingestion

- `POST /api/ingestion/ingest` - Ingest documents (stateless)
- `POST /api/ingestion/ingest/stream` - Ingest with streaming progress (SSE)
- `POST /api/ingestion/session/{session_id}/ingest` - Ingest into session

### Extraction

- `POST /api/extraction/extract` - Extract triples from text (stateless)
- `POST /api/extraction/extract/batch` - Batch extraction
- `POST /api/extraction/extract/stream` - Streaming batch extraction (SSE)
- `POST /api/extraction/session/{session_id}/extract` - Extract with session context

### Ontology

- `POST /api/ontology/recommend` - Recommend ontology from text
- `POST /api/ontology/extend/analyze` - Analyze extension needs
- `POST /api/ontology/extend/apply` - Apply ontology extension
- `POST /api/ontology/recommend-and-extract` - Combined operation

### Resolution

- `POST /api/resolution/resolve` - Run entity resolution (stateless)
- `POST /api/resolution/session/{session_id}/resolve` - Resolve within session

### Process

- `POST /api/process/extract` - Extract process graph from text

## Usage Modes

### Stateless Mode

Provide all configuration in each request. No session management required.

**Example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/extraction/extract",
    json={
        "text": "Alice works at TechCorp.",
        "source_name": "doc1",
        "ontology": {
            "entity_types": [...],
            "relation_types": [...]
        }
    }
)
```

### Stateful Mode (Session-based)

Create a session and reuse configuration across multiple requests.

**Example:**

```python
import requests

# Create session
session_response = requests.post(
    "http://localhost:8000/api/sessions",
    json={"name": "my-project"}
)
session_id = session_response.json()["session_id"]

# Set ontology
requests.put(
    f"http://localhost:8000/api/sessions/{session_id}/ontology",
    json={"ontology": {...}}
)

# Extract using session
response = requests.post(
    f"http://localhost:8000/api/extraction/session/{session_id}/extract",
    json={
        "text": "Alice works at TechCorp.",
        "source_name": "doc1"
    }
)
```

### Streaming Mode (Server-Sent Events)

For long-running operations, use streaming endpoints to receive progress updates.

**Example:**

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/api/extraction/extract/stream",
    json={
        "texts": [
            {"text": "Alice works at TechCorp.", "source_name": "doc1"},
            {"text": "Bob manages Alice.", "source_name": "doc2"}
        ]
    },
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        print(f"Extracted {data['triple_count']} triples from {data['source_name']}")
```

## Python Client Example

```python
import requests

class SpindleAPIClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
    
    def create_session(self, name=None):
        """Create a new session."""
        response = requests.post(
            f"{self.base_url}/api/sessions",
            json={"name": name}
        )
        response.raise_for_status()
        self.session_id = response.json()["session_id"]
        return self.session_id
    
    def extract_triples(self, text, source_name, use_session=False):
        """Extract triples from text."""
        if use_session and self.session_id:
            url = f"{self.base_url}/api/extraction/session/{self.session_id}/extract"
        else:
            url = f"{self.base_url}/api/extraction/extract"
        
        response = requests.post(
            url,
            json={
                "text": text,
                "source_name": source_name
            }
        )
        response.raise_for_status()
        return response.json()
    
    def recommend_ontology(self, text, scope="balanced"):
        """Recommend an ontology."""
        response = requests.post(
            f"{self.base_url}/api/ontology/recommend",
            json={"text": text, "scope": scope}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = SpindleAPIClient()
client.create_session("my-project")
result = client.extract_triples(
    "Alice works at TechCorp.",
    "example-doc",
    use_session=True
)
print(f"Extracted {result['triple_count']} triples")
```

## Error Handling

The API returns structured error responses:

```json
{
  "code": "HTTP_404",
  "message": "Session not found: abc123",
  "details": {...}
}
```

Common status codes:
- `200` - Success
- `201` - Created
- `204` - No Content (successful deletion)
- `400` - Bad Request (invalid input)
- `404` - Not Found
- `422` - Validation Error
- `500` - Internal Server Error

## Configuration

### Environment Variables

You can configure the API using environment variables:

- `SPINDLE_API_HOST` - Host to bind to (default: 0.0.0.0)
- `SPINDLE_API_PORT` - Port to listen on (default: 8000)
- LLM authentication variables (see Spindle documentation)

### CORS

By default, CORS is enabled for all origins. In production, configure specific origins in `spindle/api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Testing

Run the API tests:

```bash
pytest tests/test_api.py -v
```

## Architecture

The API follows a modular structure:

- `main.py` - FastAPI app, session management, health checks
- `models.py` - Pydantic request/response models
- `session.py` - Session state management
- `dependencies.py` - Dependency injection helpers
- `utils.py` - Utility functions
- `routers/` - Service-specific route handlers
  - `ingestion.py` - Document ingestion
  - `extraction.py` - Triple extraction
  - `ontology.py` - Ontology operations
  - `resolution.py` - Entity resolution
  - `process.py` - Process extraction

## Performance Considerations

- **Batch operations**: Use batch endpoints for multiple documents
- **Streaming**: Use streaming endpoints for real-time progress
- **Sessions**: Reuse sessions for multiple related operations
- **Async operations**: The API uses async endpoints where beneficial

## Development

### Adding New Endpoints

1. Define request/response models in `models.py`
2. Create router in `routers/` directory
3. Register router in `main.py`
4. Add tests in `tests/test_api.py`

### Running in Development

```bash
uvicorn spindle.api.main:app --reload --host 0.0.0.0 --port 8000
```

## License

MIT License - See LICENSE file for details

