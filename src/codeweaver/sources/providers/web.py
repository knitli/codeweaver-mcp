# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Web crawler data source implementation for CodeWeaver.

Provides content discovery from websites and documentation sites,
with politeness policies and content extraction capabilities.
"""

import logging

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from codeweaver.sources.base import AbstractDataSource, SourceWatcher
from codeweaver.utils.decorators import not_implemented


if TYPE_CHECKING:
    from codeweaver.cw_types import ContentItem, SourceCapabilities, SourceCapability
logger = logging.getLogger(__name__)


@not_implemented
class WebCrawlerSourceConfig(BaseModel):
    """Configuration specific to web crawler data sources."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)
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
    start_urls: Annotated[list[str], Field(description="Starting URLs for crawling (required)")]
    allowed_domains: Annotated[
        list[str], Field(default_factory=list, description="Domains allowed for crawling")
    ]
    max_depth: Annotated[int, Field(3, ge=1, le=10, description="Maximum crawling depth")]
    max_pages: Annotated[
        int, Field(1000, ge=1, le=100000, description="Maximum number of pages to crawl")
    ]
    delay_between_requests: Annotated[
        float, Field(1.0, ge=0.1, le=10.0, description="Delay between requests in seconds")
    ]
    respect_robots_txt: Annotated[bool, Field(True, description="Respect robots.txt rules")]
    user_agent: Annotated[
        str, Field("CodeWeaver/1.0", min_length=1, description="User agent string for requests")
    ]
    max_requests_per_second: Annotated[
        float, Field(1.0, gt=0, le=10.0, description="Maximum requests per second")
    ]
    extract_text_only: Annotated[bool, Field(True, description="Extract text content only")]
    include_code_blocks: Annotated[
        bool, Field(True, description="Include code blocks in extraction")
    ]
    include_links: Annotated[bool, Field(False, description="Include links in extracted content")]
    include_images: Annotated[bool, Field(False, description="Include image information")]
    min_content_length: Annotated[
        int, Field(100, ge=1, le=10000, description="Minimum content length to index")
    ]
    allowed_content_types: Annotated[
        list[str],
        Field(
            default_factory=lambda: ["text/html", "text/plain", "text/markdown"],
            description="Allowed MIME content types",
        ),
    ]
    exclude_file_extensions: Annotated[
        list[str],
        Field(
            default_factory=lambda: [".pdf", ".doc", ".docx", ".xls", ".xlsx"],
            description="File extensions to exclude",
        ),
    ]
    css_selectors: Annotated[
        list[str],
        Field(default_factory=list, description="CSS selectors for specific content extraction"),
    ]
    xpath_expressions: Annotated[
        list[str],
        Field(default_factory=list, description="XPath expressions for content extraction"),
    ]


class WebCrawlerSourceProvider(AbstractDataSource):
    """Web crawler data source implementation.

    Crawls websites to extract indexable content with support for:
    - Documentation sites (GitBook, ReadTheDocs, etc.)
    - Code repositories with web interfaces
    - API documentation and technical blogs
    - Markdown and HTML content extraction

    Implements politeness policies to respect website resources
    and follows robots.txt conventions.

    Note: This is a placeholder implementation. Full web crawling
    would require additional dependencies like Scrapy or BeautifulSoup.
    """

    def __init__(self, source_id: str | None = None):
        """Initialize web crawler data source.

        Args:
            source_id: Unique identifier for this source instance
        """
        super().__init__("web", source_id)

    @classmethod
    def check_availability(cls, capability: "SourceCapability") -> tuple[bool, str | None]:
        """Check if web crawler source is available for the given capability."""
        from codeweaver.cw_types import SourceCapability

        supported_capabilities = {
            SourceCapability.CONTENT_DISCOVERY,
            SourceCapability.CONTENT_READING,
            SourceCapability.METADATA_EXTRACTION,
            SourceCapability.BATCH_PROCESSING,
            SourceCapability.RATE_LIMITING,
            SourceCapability.AUTHENTICATION,
        }
        if capability in supported_capabilities:
            try:
                import httpx
            except ImportError:
                try:
                    import requests
                except ImportError:
                    return (
                        False,
                        "HTTP client not available (install with: uv add httpx or uv add requests)",
                    )
            try:
                import bs4
            except ImportError:
                return (False, "HTML parser not available (install with: uv add beautifulsoup4)")
            else:
                return (True, None)
        if capability == SourceCapability.CHANGE_WATCHING:
            return (False, "Change watching for web content requires specialized monitoring setup")
        return (False, f"Capability {capability.value} not supported by WebCrawler source")

    def get_capabilities(self) -> "SourceCapabilities":
        """Get capabilities supported by web crawler source."""
        from codeweaver.cw_types import SourceCapabilities

        return SourceCapabilities(
            supports_content_discovery=True,
            supports_content_reading=True,
            supports_change_watching=True,
            supports_metadata_extraction=True,
            supports_batch_processing=True,
            supports_rate_limiting=True,
            supports_content_deduplication=True,
        )

    async def discover_content(self, config: WebCrawlerSourceConfig) -> "list[ContentItem]":
        """Discover content by crawling websites.

        Args:
            config: Web crawler source configuration

        Returns:
            List of discovered web content items

        Raises:
            NotImplementedError: This is a placeholder implementation
        """
        if not config.get("enabled", True):
            return []
        if start_urls := config.get("start_urls", []):
            raise NotImplementedError(
                "Web crawler source is not yet implemented. Future implementation will require web scraping dependencies."
            )
        raise ValueError("start_urls is required for web crawler source")

    async def read_content(self, item: "ContentItem") -> str:
        """Read content from a web page.

        Args:
            item: Content item representing a web page

        Returns:
            Extracted text content of the web page

        Raises:
            NotImplementedError: This is a placeholder implementation
        """
        if item.content_type != "url":
            raise ValueError(
                f"Unsupported content type for web crawler source: {item.content_type}"
            )
        raise NotImplementedError("Web content reading not yet implemented")

    async def watch_changes(
        self, config: WebCrawlerSourceConfig, callback: "Callable[[list[ContentItem]], None]"
    ) -> SourceWatcher:
        """Set up web content change watching.

        Args:
            config: Web crawler source configuration
            callback: Function to call when changes are detected

        Returns:
            Watcher for web content changes

        Raises:
            NotImplementedError: This is a placeholder implementation
        """
        if not config.get("enable_change_watching", False):
            raise NotImplementedError("Change watching is disabled in configuration")
        raise NotImplementedError("Web change watching not yet implemented")

    async def validate_source(self, config: WebCrawlerSourceConfig) -> bool:
        """Validate web crawler source configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if not await super().validate_source(config):
                return False
            start_urls = config.get("start_urls", [])
            if not start_urls:
                logger.warning("Missing start_urls in web crawler source configuration")
                return False
            for url in start_urls:
                if not url.startswith(("http://", "https://")):
                    logger.warning("Invalid URL format: %s", url)
                    return False
            logger.warning("Web URL accessibility validation not fully implemented")
        except Exception:
            logger.exception("Error validating web crawler source configuration")
            return False
        return True

    async def get_content_metadata(self, item: "ContentItem") -> dict[str, Any]:
        """Get detailed metadata for web content.

        Args:
            item: Content item representing web content

        Returns:
            Dictionary with detailed web metadata
        """
        metadata = await super().get_content_metadata(item)
        metadata.update({
            "web_metadata_available": False,
            "implementation_note": "Web metadata extraction not yet implemented",
        })
        return metadata

    async def health_check(self) -> bool:
        """Check web data source health by testing connectivity.

        Returns:
            True if source is healthy and operational, False otherwise
        """
        try:
            if not hasattr(self, "source_id") or not self.source_id:
                logger.warning("Web source missing source_id")
                return False
            logger.debug("Web source health check - stub implementation")
            try:
                import urllib.request

                with urllib.request.urlopen(
                    "https://httpbin.org/status/200", timeout=5
                ) as response:
                    if response.getcode() == 200:
                        logger.debug("Web source health check passed - connectivity OK")
                        return True
            except Exception as e:
                logger.warning("Web source connectivity test failed: %s", e)
                return False
            logger.warning("Web source health check failed - no connectivity")
        except Exception:
            logger.exception("Web source health check failed")
            return False
        else:
            return False
