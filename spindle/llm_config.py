"""LLM Configuration and Authentication Management

This module provides unified configuration for LLM access across different providers
and authentication methods, with automatic credential detection that prefers GCP Vertex AI
when available, falling back to direct API keys.

Supports:
- Anthropic models via direct API or Vertex AI
- OpenAI models via direct API
- Gemini models via Vertex AI
- Automatic credential detection and preference
- Both Anthropic Vertex SDK and MAAS API approaches
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Available authentication methods for LLM access."""

    VERTEX_AI = "vertex_ai"  # GCP Vertex AI (for both Anthropic and Gemini)
    DIRECT_API = "direct_api"  # Direct API keys (OpenAI, Anthropic)
    VERTEX_MAAS = "vertex_maas"  # Vertex AI Model-as-a-Service API


class Provider(Enum):
    """LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"  # For Gemini via Vertex AI


@dataclass
class LLMConfig:
    """
    Configuration for LLM access with support for multiple authentication methods.

    Attributes:
        gcp_project_id: GCP project ID for Vertex AI
        gcp_region: GCP region for Vertex AI (e.g., 'us-east5')
        anthropic_api_key: Direct Anthropic API key
        openai_api_key: Direct OpenAI API key
        google_api_key: Direct Google API key
        preferred_auth_method: Preferred authentication method
        available_auth_methods: List of detected available authentication methods
        gcp_credentials_path: Path to GCP service account JSON (optional)
    """

    gcp_project_id: Optional[str] = None
    gcp_region: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    preferred_auth_method: Optional[AuthMethod] = None
    available_auth_methods: List[AuthMethod] = field(default_factory=list)
    gcp_credentials_path: Optional[str] = None

    def has_vertex_ai_auth(self) -> bool:
        """Check if Vertex AI authentication is available."""

        return (
            self.gcp_project_id is not None
            and self.gcp_region is not None
            and (self.gcp_credentials_path is not None or self._has_default_gcp_credentials())
        )

    def has_direct_anthropic(self) -> bool:
        """Check if direct Anthropic API authentication is available."""

        return self.anthropic_api_key is not None

    def has_direct_openai(self) -> bool:
        """Check if direct OpenAI authentication is available."""

        return self.openai_api_key is not None

    def _has_default_gcp_credentials(self) -> bool:
        """Check if default GCP credentials are available."""

        try:
            import google.auth

            credentials, project = google.auth.default()
            return credentials is not None
        except Exception:
            return False

    def get_auth_method_for_provider(self, provider: Provider) -> Optional[AuthMethod]:
        """Determine the best authentication method for a given provider."""

        if provider == Provider.ANTHROPIC:
            if self.has_vertex_ai_auth() and AuthMethod.VERTEX_AI in self.available_auth_methods:
                return AuthMethod.VERTEX_AI
            if self.has_direct_anthropic() and AuthMethod.DIRECT_API in self.available_auth_methods:
                return AuthMethod.DIRECT_API
        elif provider == Provider.GOOGLE:
            if self.has_vertex_ai_auth() and AuthMethod.VERTEX_AI in self.available_auth_methods:
                return AuthMethod.VERTEX_AI
        elif provider == Provider.OPENAI:
            if self.has_direct_openai() and AuthMethod.DIRECT_API in self.available_auth_methods:
                return AuthMethod.DIRECT_API

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary representation."""

        return {
            "gcp_project_id": self.gcp_project_id,
            "gcp_region": self.gcp_region,
            "google_api_key": self.google_api_key,
            "anthropic_api_key": self.anthropic_api_key,
            "openai_api_key": self.openai_api_key,
            "preferred_auth_method": self.preferred_auth_method.value if self.preferred_auth_method else None,
            "available_auth_methods": [m.value for m in self.available_auth_methods],
            "gcp_credentials_path": self.gcp_credentials_path,
            "has_vertex_ai_auth": self.has_vertex_ai_auth(),
        }


def detect_available_auth() -> LLMConfig:
    """
    Auto-detect available authentication methods from environment variables.

    Priority (per requirement 3b):
        1. Check for GCP Vertex AI credentials (GOOGLE_APPLICATION_CREDENTIALS, default credentials)
        2. Check for GCP project ID and GCP region
        3. Check for direct API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)
        4. Prefer Vertex AI if both GCP and API keys are available

    Returns:
        LLMConfig with detected credentials and preferred authentication method
    """

    config = LLMConfig()
    available_methods = []

    config.gcp_project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    config.gcp_region = os.getenv("GCP_REGION") or os.getenv("CLOUD_ML_REGION", "us-east5")
    config.gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    config.openai_api_key = os.getenv("OPENAI_API_KEY")
    config.google_api_key = os.getenv("GOOGLE_API_KEY")

    if config.has_vertex_ai_auth():
        available_methods.append(AuthMethod.VERTEX_AI)
        logger.info("Detected Vertex AI credentials")

    if config.has_direct_anthropic() or config.has_direct_openai() or config.google_api_key:
        available_methods.append(AuthMethod.DIRECT_API)
        logger.info("Detected direct API key credentials")

    config.available_auth_methods = available_methods

    if AuthMethod.VERTEX_AI in available_methods:
        config.preferred_auth_method = AuthMethod.VERTEX_AI
        logger.info("Preferred auth method: Vertex AI")
    elif AuthMethod.DIRECT_API in available_methods:
        config.preferred_auth_method = AuthMethod.DIRECT_API
        logger.info("Preferred auth method: Direct API")
    else:
        logger.warning("No authentication credentials detected")

    return config


def create_baml_env_overrides(config: LLMConfig) -> Dict[str, str]:
    """
    Generate environment variable overrides for BAML based on detected credentials.

    These can be passed to BAML's with_options(env=...) method to dynamically
    configure authentication at runtime.

    Args:
        config: LLMConfig with detected or explicitly set credentials

    Returns:
        Dictionary of environment variables to override
    """

    env_overrides: Dict[str, str] = {}

    if config.has_vertex_ai_auth():
        env_overrides["GCP_PROJECT_ID"] = config.gcp_project_id
        env_overrides["GCP_REGION"] = config.gcp_region
        if config.gcp_credentials_path:
            env_overrides["GOOGLE_APPLICATION_CREDENTIALS"] = config.gcp_credentials_path

    if config.has_direct_anthropic():
        env_overrides["ANTHROPIC_API_KEY"] = config.anthropic_api_key

    if config.has_direct_openai():
        env_overrides["OPENAI_API_KEY"] = config.openai_api_key

    if config.google_api_key:
        env_overrides["GOOGLE_API_KEY"] = config.google_api_key

    return env_overrides


def get_anthropic_vertex_client(config: LLMConfig):
    """
    Create an Anthropic Vertex AI client using the Anthropic SDK.

    Args:
        config: LLMConfig with GCP credentials

    Returns:
        AnthropicVertex client instance

    Raises:
        ImportError: If anthropic[vertex] is not installed
        ValueError: If required GCP credentials are missing
    """

    if not config.has_vertex_ai_auth():
        raise ValueError(
            "GCP Vertex AI authentication not available. "
            "Set GCP_PROJECT_ID, GCP_REGION, and ensure GCP credentials are configured."
        )

    try:
        from anthropic import AnthropicVertex
    except ImportError as exc:
        raise ImportError(
            "anthropic[vertex] is required for Vertex AI access. "
            "Install with: pip install 'anthropic[vertex]>=0.68.0'"
        ) from exc

    return AnthropicVertex(
        project_id=config.gcp_project_id,
        region=config.gcp_region,
    )


def get_vertex_maas_api_token(config: LLMConfig) -> str:
    """
    Get OAuth2 access token for Vertex AI MAAS API.

    Args:
        config: LLMConfig with GCP credentials

    Returns:
        OAuth2 access token string

    Raises:
        ImportError: If google-auth is not installed
        ValueError: If GCP credentials are not available
    """

    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as exc:
        raise ImportError(
            "google-auth is required for MAAS API access. "
            "Install with: pip install 'google-auth>=2.0.0'"
        ) from exc

    creds, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    return creds.token


def create_vertex_maas_request(
    config: LLMConfig,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create a request payload for Vertex AI MAAS API.

    Args:
        config: LLMConfig with GCP credentials
        model_name: Model name (e.g., 'claude-sonnet-4@20250514')
        messages: List of message dicts with 'role' and 'content'
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters for the API

    Returns:
        Dictionary containing request configuration
    """

    if not config.has_vertex_ai_auth():
        raise ValueError("GCP Vertex AI authentication not available")

    access_token = get_vertex_maas_api_token(config)

    url = (
        f"https://{config.gcp_region}-aiplatform.googleapis.com/v1/"
        f"projects/{config.gcp_project_id}/locations/{config.gcp_region}/"
        f"publishers/anthropic/models/{model_name}:streamRawPredict"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    payload = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": messages,
        "max_tokens": max_tokens,
        **kwargs,
    }

    return {
        "url": url,
        "headers": headers,
        "payload": payload,
    }


def setup_llm_config(
    gcp_project_id: Optional[str] = None,
    gcp_region: Optional[str] = None,
    prefer_vertex_ai: bool = True,
    auto_detect: bool = True,
) -> LLMConfig:
    """
    Setup LLM configuration with optional overrides.

    Args:
        gcp_project_id: Override GCP project ID (None = use env var)
        gcp_region: Override GCP region (None = use env var)
        prefer_vertex_ai: Whether to prefer Vertex AI over direct API keys
        auto_detect: Whether to auto-detect credentials from environment

    Returns:
        Configured LLMConfig instance
    """

    if auto_detect:
        config = detect_available_auth()
    else:
        config = LLMConfig()

    if gcp_project_id is not None:
        config.gcp_project_id = gcp_project_id
    if gcp_region is not None:
        config.gcp_region = gcp_region

    if prefer_vertex_ai and config.has_vertex_ai_auth():
        config.preferred_auth_method = AuthMethod.VERTEX_AI
    elif not prefer_vertex_ai and config.has_direct_anthropic():
        config.preferred_auth_method = AuthMethod.DIRECT_API

    return config


