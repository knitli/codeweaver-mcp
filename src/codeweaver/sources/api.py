# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
API data source implementation for CodeWeaver.

Provides content discovery from REST and GraphQL APIs, treating
API documentation, schemas, and responses as indexable content.
"""

import logging

from collections.abc import Callable
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from codeweaver._types import ContentItem, SourceCapabilities
from codeweaver.sources.base import AbstractDataSource, SourceWatcher
from codeweaver.utils.decorators import not_implemented


logger = logging.getLogger(__name__)


@not_implemented(
    message=None,
    suggestions=[
        "We haven't implemented API source yet",
        "We'd love your help implementing it!",
        "Consider contributing to the project",
        "Or [open an issue](https://github.com/knitli/codeweaver-mcp/issues/) to discuss your use case",
        "You can also implement your own source using the SourceData protocol",
    ]
    )
class APISourceConfig(BaseModel):
    """Configuration specific to API data sources."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Inherited from BaseSourceConfig
    enabled: Annotated[bool, Field(True, description="Whether source is enabled")]
    priority: Annotated[int, Field(1, ge=1, le=100, description="Source priority")]
    source_id: Annotated[str | None, Field(None, description="Unique source identifier")]
    include_patterns: Annotated[
        list[str], Field(default_factory=list, description="File patterns to include")
    ]
    exclude_patterns: Annotated[
        list[str], Field(default_factory=list, description="File patterns to exclude")
    ]
    max_file_size_mb: Annotated[int, Field(1, ge=1, le=1000, description="Maximum file size in MB")]
    batch_size: Annotated[int, Field(8, ge=1, le=1000, description="Batch size for processing")]
    max_concurrent_requests: Annotated[
        int, Field(10, ge=1, le=100, description="Maximum concurrent requests")
    ]
    request_timeout_seconds: Annotated[
        int, Field(30, ge=1, le=300, description="Request timeout in seconds")
    ]
    enable_change_watching: Annotated[bool, Field(False, description="Enable change watching")]
    change_check_interval_seconds: Annotated[
        int, Field(60, ge=1, le=3600, description="Change check interval in seconds")
    ]
    enable_content_deduplication: Annotated[
        bool, Field(True, description="Enable content deduplication")
    ]
    enable_metadata_extraction: Annotated[
        bool, Field(False, description="Enable metadata extraction")
    ]
    supported_languages: Annotated[
        list[str], Field(default_factory=list, description="Supported programming languages")
    ]

    # API specific settings
    api_type: Annotated[str, Field(description="API type: 'rest', 'graphql', 'openapi', 'swagger'")]
    base_url: Annotated[str, Field(description="Base URL for the API (required)")]
    endpoints: Annotated[
        list[str], Field(default_factory=list, description="List of API endpoints to index")
    ]

    # Authentication
    auth_type: Annotated[
        str | None,
        Field(None, description="Authentication type: 'bearer', 'basic', 'api_key', 'oauth2'"),
    ]
    api_key: Annotated[str | None, Field(None, description="API key for authentication")]
    bearer_token: Annotated[str | None, Field(None, description="Bearer token for authentication")]
    username: Annotated[str | None, Field(None, description="Username for basic authentication")]
    password: Annotated[str | None, Field(None, description="Password for basic authentication")]

    # Request settings
    headers: Annotated[
        dict[str, str], Field(default_factory=dict, description="Custom HTTP headers")
    ]
    query_parameters: Annotated[
        dict[str, str], Field(default_factory=dict, description="Default query parameters")
    ]
    request_method: Annotated[
        str, Field("GET", description="HTTP request method: 'GET', 'POST', etc.")
    ]
    request_body: Annotated[
        dict[str, Any] | None, Field(None, description="Request body for POST/PUT requests")
    ]

    # Content extraction settings
    schema_discovery: Annotated[bool, Field(True, description="Discover OpenAPI/GraphQL schemas")]
    sample_responses: Annotated[bool, Field(False, description="Include sample API responses")]
    include_documentation: Annotated[bool, Field(True, description="Include API documentation")]
    max_response_size_kb: Annotated[
        int, Field(1024, ge=1, le=10240, description="Maximum response size in KB")
    ]


@not_implemented(
    message="API source requires HTTP client dependencies like httpx or requests",
    suggestions=[
        "Install httpx: pip install httpx",
        "Install requests: pip install requests",
        "Use filesystem source for static API documentation",
        "Consider implementing with aiohttp for async support",
        "Use web crawler source for API documentation pages"
    ]
)
class APISourceProvider(AbstractDataSource):
    """API data source implementation.

    Treats API content as indexable text by extracting:
    - OpenAPI/Swagger specifications
    - GraphQL schemas and queries
    - API documentation and descriptions
    - Sample API responses (configurable)
    - Endpoint definitions and parameters

    Note: This is a placeholder implementation. Full API integration
    would require additional dependencies like httpx or requests.
    """

    def __init__(self, source_id: str | None = None):
        """Initialize API data source.

        Args:
            source_id: Unique identifier for this source instance
        """
        super().__init__("api", source_id)

    @classmethod
    def check_availability(cls, capability: "SourceCapability") -> tuple[bool, str | None]:
        """Check if API source is available for the given capability."""
        from codeweaver._types.source_enums import SourceCapability

        # API source supports most capabilities but requires HTTP libraries
        supported_capabilities = {
            SourceCapability.CONTENT_DISCOVERY,
            SourceCapability.CONTENT_READING,
            SourceCapability.METADATA_EXTRACTION,
            SourceCapability.BATCH_PROCESSING,
            SourceCapability.AUTHENTICATION,
            SourceCapability.RATE_LIMITING,
            SourceCapability.PAGINATION,
        }

        if capability in supported_capabilities:
            # Check for HTTP client dependencies
            try:
                import httpx  # noqa: F401
                return True, None
            except ImportError:
                try:
                    import requests  # noqa: F401
                    return True, None
                except ImportError:
                    return False, "HTTP client not available (install with: uv add httpx or uv add requests)"

        return False, f"Capability {capability.value} not supported by API source"

    def get_capabilities(self) -> SourceCapabilities:
        """Get capabilities supported by API source."""
        return SourceCapabilities(
            supports_content_discovery=True,
            supports_content_reading=True,
            supports_change_watching=True,
            supports_metadata_extraction=True,
            supports_batch_processing=True,
            supports_authentication=True,
            supports_rate_limiting=True,
            supports_pagination=True,
        )

    async def discover_content(self, config: APISourceConfig) -> list[ContentItem]:
        """Discover content from an API.

        Args:
            config: API source configuration

        Returns:
            List of discovered API content items

        Raises:
            NotImplementedError: This is a placeholder implementation
        """
        if not config.get("enabled", True):
            return []

        api_type = config.get("api_type")
        if not api_type:
            raise ValueError("api_type is required for API source")

        if base_url := config.get("base_url"):  # noqa: F841
            # TODO: Implement API content discovery
            # This would involve:
            # 1. Discovering API schemas (OpenAPI, GraphQL)
            # 2. Extracting endpoint definitions
            # 3. Making sample requests to endpoints
            # 4. Parsing API documentation
            # 5. Creating ContentItems for each API artifact

            raise NotImplementedError(
                f"API source for {api_type} is not yet implemented. "
                "Future implementation will require HTTP client dependencies."
            )
        raise ValueError("base_url is required for API source")

    async def read_content(self, item: ContentItem) -> str:
        """Read content from an API resource.

        Args:
            item: Content item representing an API resource

        Returns:
            Text representation of the API content

        Raises:
            NotImplementedError: This is a placeholder implementation
        """
        if item.content_type != "api":
            raise ValueError(f"Unsupported content type for API source: {item.content_type}")

        # TODO: Implement API content reading
        # This would involve:
        # 1. Making HTTP requests to API endpoints
        # 2. Handling authentication
        # 3. Parsing responses (JSON, XML, etc.)
        # 4. Formatting as readable text
        # 5. Handling rate limits and errors

        raise NotImplementedError("API content reading not yet implemented")

    async def watch_changes(
        self, config: APISourceConfig, callback: Callable[[list[ContentItem]], None]
    ) -> SourceWatcher:
        """Set up API change watching.

        Args:
            config: API source configuration
            callback: Function to call when changes are detected

        Returns:
            Watcher for API changes

        Raises:
            NotImplementedError: This is a placeholder implementation
        """
        if not config.get("enable_change_watching", False):
            raise NotImplementedError("Change watching is disabled in configuration")

        # TODO: Implement API change watching
        # This would involve:
        # 1. Periodic polling of API endpoints
        # 2. Checking for schema changes
        # 3. Detecting new endpoints or documentation
        # 4. Comparing response structures
        # 5. Handling webhooks for real-time updates

        raise NotImplementedError("API change watching not yet implemented")

    async def validate_source(self, config: APISourceConfig) -> bool:
        """Validate API source configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Call parent validation first
            if not await super().validate_source(config):
                return False

            # Check required fields
            api_type = config.get("api_type")
            if not api_type:
                logger.warning("Missing api_type in API source configuration")
                return False

            base_url = config.get("base_url")
            if not base_url:
                logger.warning("Missing base_url in API source configuration")
                return False

            # TODO: Test API connectivity
            # This would involve making a test request to verify
            # the API is accessible and authentication works

            logger.warning("API connectivity validation not fully implemented")

        except Exception:
            logger.exception("Error validating API source configuration")
            return False

        else:
            return True

    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """Get detailed metadata for API content.

        Args:
            item: Content item representing API content

        Returns:
            Dictionary with detailed API metadata
        """
        metadata = await super().get_content_metadata(item)

        # TODO: Add API-specific metadata
        # This would include:
        # - HTTP methods and status codes
        # - Request/response schemas
        # - Authentication requirements
        # - Rate limiting information
        # - API version and documentation

        metadata.update({
            "api_metadata_available": False,
            "implementation_note": "API metadata extraction not yet implemented",
        })

        return metadata
