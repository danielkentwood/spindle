"""Integration tests for embedding APIs.

Tests embeddings from:
- OpenAI (via API)
- Google Gemini (via API)
- HuggingFace (via API)

Uses API keys from .env file. Marked as integration tests.
"""

import os
import time
import pytest
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test texts
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a high-level programming language.",
    "Machine learning models can generate embeddings from text.",
    "OpenAI creates advanced language models.",
]


# ========== Fixtures ==========

@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment."""
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture
def gemini_api_key():
    """Get Google Gemini API key from environment."""
    return os.getenv("GEMINI_API_KEY")


@pytest.fixture
def huggingface_api_key():
    """Get HuggingFace API key from environment."""
    return os.getenv("HF_API_KEY")


@pytest.fixture
def skip_if_no_openai_key(openai_api_key):
    """Skip test if OpenAI API key is not available."""
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping test")


@pytest.fixture
def skip_if_no_gemini_key(gemini_api_key):
    """Skip test if Gemini API key is not available."""
    if not gemini_api_key:
        pytest.skip("GEMINI_API_KEY not set - skipping test")


@pytest.fixture
def skip_if_no_huggingface_key(huggingface_api_key):
    """Skip test if HuggingFace API key is not available."""
    if not huggingface_api_key:
        pytest.skip("HF_API_KEY not set - skipping test")


# ========== Helper Functions ==========

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ========== OpenAI Tests ==========

@pytest.mark.integration
class TestOpenAIEmbeddings:
    """Tests for OpenAI embeddings API."""

    def test_openai_embeddings_basic(
        self, skip_if_no_openai_key, openai_api_key
    ):
        """Test basic OpenAI embeddings generation."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai module not installed")

        client = openai.OpenAI(api_key=openai_api_key)
        model = "text-embedding-3-small"

        # Test single embedding
        response = client.embeddings.create(
            model=model,
            input=TEST_TEXTS[0]
        )
        embedding = response.data[0].embedding

        assert embedding is not None
        assert len(embedding) > 0
        assert isinstance(embedding, list)
        assert all(isinstance(x, (int, float)) for x in embedding)

    def test_openai_embeddings_multiple_texts(
        self, skip_if_no_openai_key, openai_api_key
    ):
        """Test OpenAI embeddings for multiple texts."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai module not installed")

        client = openai.OpenAI(api_key=openai_api_key)
        model = "text-embedding-3-small"

        results = []
        for text in TEST_TEXTS:
            response = client.embeddings.create(
                model=model,
                input=text
            )
            embedding = response.data[0].embedding
            dim = len(embedding)
            results.append({
                "text": text,
                "dimension": dim,
                "embedding": embedding
            })

        # Verify all embeddings have same dimension
        dimensions = [r["dimension"] for r in results]
        assert len(set(dimensions)) == 1, "All embeddings should have same dimension"
        assert dimensions[0] > 0

    def test_openai_embeddings_similarity(
        self, skip_if_no_openai_key, openai_api_key
    ):
        """Test similarity calculation between embeddings."""
        try:
            import openai
        except ImportError:
            pytest.skip("openai module not installed")

        client = openai.OpenAI(api_key=openai_api_key)
        model = "text-embedding-3-small"

        # Get embeddings for two texts
        resp1 = client.embeddings.create(model=model, input=TEST_TEXTS[0])
        resp2 = client.embeddings.create(model=model, input=TEST_TEXTS[1])
        vec1 = np.array(resp1.data[0].embedding)
        vec2 = np.array(resp2.data[0].embedding)

        similarity = calculate_cosine_similarity(vec1, vec2)

        # Similarity should be between -1 and 1
        assert -1 <= similarity <= 1

        # Similarity should be a number
        assert not np.isnan(similarity)
        assert not np.isinf(similarity)


# ========== Gemini Tests ==========

@pytest.mark.integration
class TestGeminiEmbeddings:
    """Tests for Google Gemini embeddings API."""

    def test_gemini_embeddings_basic(
        self, skip_if_no_gemini_key, gemini_api_key
    ):
        """Test basic Gemini embeddings generation."""
        try:
            import google.generativeai as genai
        except ImportError:
            pytest.skip("google.generativeai module not installed")

        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"

        embedding = genai.embed_content(
            model=model,
            content=TEST_TEXTS[0],
            task_type="retrieval_document"
        )
        embedding_vector = embedding['embedding']

        assert embedding_vector is not None
        assert len(embedding_vector) > 0
        assert isinstance(embedding_vector, list)
        assert all(isinstance(x, (int, float)) for x in embedding_vector)

    def test_gemini_embeddings_multiple_texts(
        self, skip_if_no_gemini_key, gemini_api_key
    ):
        """Test Gemini embeddings for multiple texts."""
        try:
            import google.generativeai as genai
        except ImportError:
            pytest.skip("google.generativeai module not installed")

        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"

        results = []
        for text in TEST_TEXTS:
            embedding = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            embedding_vector = embedding['embedding']
            dim = len(embedding_vector)
            results.append({
                "text": text,
                "dimension": dim,
                "embedding": embedding_vector
            })

        # Verify all embeddings have same dimension
        dimensions = [r["dimension"] for r in results]
        assert len(set(dimensions)) == 1, "All embeddings should have same dimension"
        assert dimensions[0] > 0

    def test_gemini_embeddings_similarity(
        self, skip_if_no_gemini_key, gemini_api_key
    ):
        """Test similarity calculation between Gemini embeddings."""
        try:
            import google.generativeai as genai
        except ImportError:
            pytest.skip("google.generativeai module not installed")

        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"

        emb1 = genai.embed_content(
            model=model,
            content=TEST_TEXTS[0],
            task_type="retrieval_document"
        )
        emb2 = genai.embed_content(
            model=model,
            content=TEST_TEXTS[1],
            task_type="retrieval_document"
        )

        vec1 = np.array(emb1['embedding'])
        vec2 = np.array(emb2['embedding'])
        similarity = calculate_cosine_similarity(vec1, vec2)

        # Similarity should be between -1 and 1
        assert -1 <= similarity <= 1
        assert not np.isnan(similarity)
        assert not np.isinf(similarity)


# ========== HuggingFace Tests ==========

@pytest.mark.integration
class TestHuggingFaceEmbeddings:
    """Tests for HuggingFace embeddings API."""

    def test_huggingface_inference_client(
        self, skip_if_no_huggingface_key, huggingface_api_key
    ):
        """Test HuggingFace embeddings using InferenceClient."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            pytest.skip("huggingface_hub not installed")

        client = InferenceClient(token=huggingface_api_key)
        model = "sentence-transformers/all-MiniLM-L6-v2"

        # Test with first text
        result = client.feature_extraction(TEST_TEXTS[0], model=model)

        # Handle response format
        if isinstance(result, list):
            embedding = result[0] if len(result) > 0 else result
        elif hasattr(result, 'tolist'):
            embedding = result.tolist()
        else:
            embedding = result

        dim = len(embedding) if isinstance(embedding, (list, tuple)) else 0
        assert dim > 0, "Embedding should have positive dimension"

    def test_huggingface_inference_client_multiple_texts(
        self, skip_if_no_huggingface_key, huggingface_api_key
    ):
        """Test HuggingFace embeddings for multiple texts using InferenceClient."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            pytest.skip("huggingface_hub not installed")

        client = InferenceClient(token=huggingface_api_key)
        model = "sentence-transformers/all-MiniLM-L6-v2"

        results = []
        for text in TEST_TEXTS:
            result = client.feature_extraction(text, model=model)

            if isinstance(result, list):
                embedding = result[0] if len(result) > 0 else result
            elif hasattr(result, 'tolist'):
                embedding = result.tolist()
            else:
                embedding = result

            dim = len(embedding) if isinstance(embedding, (list, tuple)) else 0
            results.append({
                "text": text,
                "dimension": dim,
                "embedding": embedding
            })

        # Verify all embeddings have same dimension
        dimensions = [r["dimension"] for r in results if r["dimension"] > 0]
        assert len(dimensions) > 0, "At least one embedding should succeed"
        if len(dimensions) > 1:
            assert len(set(dimensions)) == 1, "All embeddings should have same dimension"

    def test_huggingface_direct_api(
        self, skip_if_no_huggingface_key, huggingface_api_key
    ):
        """Test HuggingFace embeddings using direct API endpoint."""
        import requests

        model = "sentence-transformers/all-MiniLM-L6-v2"
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {huggingface_api_key}"}

        # Test with first text
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": TEST_TEXTS[0]},
            timeout=60
        )

        # Handle model loading
        if response.status_code == 503:
            time.sleep(10)
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": TEST_TEXTS[0]},
                timeout=60
            )

        if response.status_code == 410:
            pytest.skip("Endpoint no longer available (410 Gone)")

        response.raise_for_status()
        result = response.json()

        # Handle different response formats
        if isinstance(result, list):
            embedding = result[0] if len(result) > 0 else result
        elif isinstance(result, dict) and "embeddings" in result:
            embedding = result["embeddings"][0] if result["embeddings"] else []
        else:
            embedding = result

        dim = len(embedding) if isinstance(embedding, (list, tuple)) else 0
        assert dim > 0, "Embedding should have positive dimension"

    def test_huggingface_direct_api_multiple_texts(
        self, skip_if_no_huggingface_key, huggingface_api_key
    ):
        """Test HuggingFace embeddings for multiple texts using direct API."""
        import requests

        model = "sentence-transformers/all-MiniLM-L6-v2"
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {huggingface_api_key}"}

        results = []
        for text in TEST_TEXTS:
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": text},
                timeout=60
            )

            if response.status_code == 503:
                time.sleep(5)
                response = requests.post(
                    api_url,
                    headers=headers,
                    json={"inputs": text},
                    timeout=60
                )

            if response.status_code == 410:
                pytest.skip("Endpoint no longer available (410 Gone)")

            response.raise_for_status()
            result = response.json()

            if isinstance(result, list):
                embedding = result[0] if len(result) > 0 else result
            elif isinstance(result, dict) and "embeddings" in result:
                embedding = result["embeddings"][0] if result["embeddings"] else []
            else:
                embedding = result

            dim = len(embedding) if isinstance(embedding, (list, tuple)) else 0
            results.append({
                "text": text,
                "dimension": dim,
                "embedding": embedding
            })

        # Verify at least some embeddings succeeded
        successful_results = [r for r in results if r["dimension"] > 0]
        assert len(successful_results) > 0, "At least one embedding should succeed"

        # If multiple succeeded, verify dimensions match
        if len(successful_results) > 1:
            dimensions = [r["dimension"] for r in successful_results]
            assert len(set(dimensions)) == 1, "All embeddings should have same dimension"

    def test_huggingface_similarity_calculation(
        self, skip_if_no_huggingface_key, huggingface_api_key
    ):
        """Test similarity calculation with HuggingFace embeddings."""
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            pytest.skip("huggingface_hub not installed")

        client = InferenceClient(token=huggingface_api_key)
        model = "sentence-transformers/all-MiniLM-L6-v2"

        emb1 = client.feature_extraction(TEST_TEXTS[0], model=model)
        emb2 = client.feature_extraction(TEST_TEXTS[1], model=model)

        # Convert to numpy arrays
        if isinstance(emb1, list):
            vec1 = np.array(emb1[0] if len(emb1) > 0 else emb1)
        elif hasattr(emb1, 'tolist'):
            vec1 = np.array(emb1)
        else:
            vec1 = np.array(emb1)

        if isinstance(emb2, list):
            vec2 = np.array(emb2[0] if len(emb2) > 0 else emb2)
        elif hasattr(emb2, 'tolist'):
            vec2 = np.array(emb2)
        else:
            vec2 = np.array(emb2)

        if len(vec1) == len(vec2) and len(vec1) > 0:
            similarity = calculate_cosine_similarity(vec1, vec2)
            assert -1 <= similarity <= 1
            assert not np.isnan(similarity)
            assert not np.isinf(similarity)

