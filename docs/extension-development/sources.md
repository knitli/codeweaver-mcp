# Building Custom Data Sources

This guide covers building custom data source connectors for CodeWeaver. Data sources provide content discovery and retrieval from various repositories and systems.

## 🎯 Overview

Data sources connect CodeWeaver to different types of content repositories and systems. CodeWeaver supports various source types:

- **Filesystem Sources**: Local and network file systems
- **Version Control Sources**: Git repositories, SVN, etc.
- **API Sources**: REST APIs, GraphQL endpoints, custom protocols
- **Database Sources**: SQL databases, NoSQL databases, document stores
- **Web Sources**: Web scraping, RSS feeds, web crawlers
- **Custom Sources**: Specialized integrations and proprietary systems

## 🏗️ Source Architecture

### Core DataSource Protocol

```python
from typing import Protocol, runtime_checkable, Callable, Any
from codeweaver.cw_types import (
    SourceCapabilities, SourceConfig, ContentItem, SourceWatcher
)

@runtime_checkable
class DataSource(Protocol):
    """Protocol for data source implementations."""
    
    # Capability Discovery
    def get_capabilities(self) -> SourceCapabilities:
        """Get source capabilities and supported operations."""
        ...
    
    # Content Operations
    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        """Discover content items from source."""
        ...
    
    async def read_content(self, item: ContentItem) -> str:
        """Read content from a content item."""
        ...
    
    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """Get metadata for content item."""
        ...
    
    # Change Watching
    async def watch_changes(
        self, 
        config: SourceConfig, 
        callback: Callable[[list[ContentItem]], None]
    ) -> SourceWatcher:
        """Watch for content changes."""
        ...
    
    # Validation
    async def validate_source(self, config: SourceConfig) -> bool:
        """Validate source configuration and connectivity."""
        ...
    
    async def health_check(self) -> bool:
        """Check source health and connectivity."""
        ...
```

### Base Source Classes

CodeWeaver provides abstract base classes to simplify source development:

#### AbstractDataSource
```python
from codeweaver.sources.base import AbstractDataSource
from codeweaver.cw_types import SourceProvider, SourceCapabilities, SourceCapability

class AbstractDataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, source_type: SourceProvider, source_id: str | None = None):
        self.source_type = source_type
        self.source_id = source_id or f"{source_type.value}_{id(self)}"
        self._watchers: list[SourceWatcher] = []
    
    # Required implementations
    @abstractmethod
    def get_capabilities(self) -> SourceCapabilities: ...
    @abstractmethod 
    async def discover_content(self, config: SourceConfig) -> list[ContentItem]: ...
    @abstractmethod
    async def read_content(self, item: ContentItem) -> str: ...
    
    # Default implementations provided
    async def watch_changes(self, config: SourceConfig, callback: Callable[[list[ContentItem]], None]) -> SourceWatcher: ...
    async def validate_source(self, config: SourceConfig) -> bool: ...
    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]: ...
```

## 🚀 Implementation Guide

### Step 1: Define Configuration

Create a Pydantic configuration model for your data source:

```python
from pydantic import BaseModel, Field
from typing import Annotated
from pathlib import Path

class MySourceConfig(BaseModel):
    """Configuration for MyDataSource."""
    
    # Connection settings
    base_url: Annotated[str, Field(description="Base URL for the data source")]
    api_key: Annotated[str | None, Field(default=None, description="API key for authentication")]
    timeout: Annotated[int, Field(default=30, ge=1, le=300, description="Request timeout in seconds")]
    
    # Content filtering
    include_patterns: Annotated[list[str], Field(default_factory=list, description="Glob patterns to include")]
    exclude_patterns: Annotated[list[str], Field(default_factory=list, description="Glob patterns to exclude")]
    max_file_size: Annotated[int, Field(default=1024*1024, ge=0, description="Maximum file size in bytes")]
    
    # Discovery settings
    recursive: Annotated[bool, Field(default=True, description="Recursive content discovery")]
    follow_links: Annotated[bool, Field(default=False, description="Follow symbolic links")]
    max_depth: Annotated[int, Field(default=10, ge=1, le=50, description="Maximum directory depth")]
    
    # Authentication
    username: Annotated[str | None, Field(default=None, description="Username for authentication")]
    password: Annotated[str | None, Field(default=None, description="Password for authentication")]
    
    # Rate limiting
    requests_per_second: Annotated[float, Field(default=10.0, ge=0.1, le=100.0)]
    concurrent_requests: Annotated[int, Field(default=5, ge=1, le=20)]
```

### Step 2: Implement Data Source Class

```python
from codeweaver.sources.base import AbstractDataSource
from codeweaver.cw_types import (
    SourceProvider, SourceCapabilities, SourceCapability, 
    ContentItem, SourceConfig, SourceError
)
import aiohttp
import asyncio
from datetime import datetime
import mimetypes
from urllib.parse import urljoin, urlparse

class MyDataSource(AbstractDataSource):
    """Custom data source implementation."""
    
    def __init__(self):
        super().__init__(SourceProvider.CUSTOM, "my_custom_source")
        self.session: aiohttp.ClientSession | None = None
        self._rate_limiter: AsyncRateLimiter | None = None
    
    def get_capabilities(self) -> SourceCapabilities:
        """Define source capabilities."""
        return SourceCapabilities(
            capabilities=[
                SourceCapability.CONTENT_DISCOVERY,
                SourceCapability.CONTENT_READING,
                SourceCapability.METADATA_EXTRACTION,
                SourceCapability.CHANGE_WATCHING,  # Optional
                SourceCapability.AUTHENTICATION,   # Optional
                SourceCapability.FILTERING,        # Optional
            ]
        )
    
    async def _initialize_session(self, config: MySourceConfig) -> None:
        """Initialize HTTP session with configuration."""
        if self.session is None:
            # Setup rate limiter
            self._rate_limiter = AsyncRateLimiter(
                max_calls=int(config.requests_per_second),
                time_window=1.0
            )
            
            # Setup connection parameters
            timeout = aiohttp.ClientTimeout(total=config.timeout)
            connector = aiohttp.TCPConnector(
                limit=config.concurrent_requests,
                keepalive_timeout=30
            )
            
            # Setup authentication headers
            headers = {"User-Agent": "CodeWeaver/1.0"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"
            elif config.username and config.password:
                import base64
                credentials = base64.b64encode(
                    f"{config.username}:{config.password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=headers
            )
    
    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        """Discover content items from the source."""
        if not isinstance(config, MySourceConfig):
            raise ValueError("Expected MySourceConfig")
        
        await self._initialize_session(config)
        
        content_items = []
        
        try:
            # Start discovery from base URL
            discovered_items = await self._discover_recursive(
                config.base_url, 
                config, 
                depth=0
            )
            
            # Apply filtering
            filtered_items = self._apply_content_filters(discovered_items, config)
            content_items.extend(filtered_items)
            
        except Exception as e:
            raise SourceError(f"Content discovery failed: {e}") from e
        
        return content_items
    
    async def _discover_recursive(
        self, 
        url: str, 
        config: MySourceConfig, 
        depth: int
    ) -> list[ContentItem]:
        """Recursively discover content."""
        if depth > config.max_depth:
            return []
        
        # Rate limiting
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        
        items = []
        
        try:
            # Make API request to discover content
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                
                # Parse response based on your API format
                for item_data in data.get("items", []):
                    item = await self._parse_content_item(item_data, config)
                    if item:
                        items.append(item)
                        
                        # Recursive discovery if it's a directory/collection
                        if item.content_type == "directory" and config.recursive:
                            child_items = await self._discover_recursive(
                                item.metadata.get("child_url", ""), 
                                config, 
                                depth + 1
                            )
                            items.extend(child_items)
        
        except aiohttp.ClientError as e:
            # Log error but continue discovery
            print(f"Error discovering {url}: {e}")
        
        return items
    
    async def _parse_content_item(
        self, 
        item_data: dict[str, Any], 
        config: MySourceConfig
    ) -> ContentItem | None:
        """Parse API response into ContentItem."""
        try:
            # Extract basic information
            item_id = item_data.get("id") or item_data.get("path")
            if not item_id:
                return None
            
            path = item_data.get("path", item_id)
            content_type = self._determine_content_type(item_data)
            
            # Size filtering
            size = item_data.get("size", 0)
            if size > config.max_file_size:
                return None
            
            # Extract metadata
            metadata = {
                "url": item_data.get("download_url") or item_data.get("url"),
                "type": item_data.get("type"),
                "sha": item_data.get("sha"),
                "child_url": item_data.get("url"),  # For directories
                "encoding": item_data.get("encoding"),
                **item_data.get("metadata", {})
            }
            
            # Parse modified time
            modified_time = None
            if "updated_at" in item_data:
                try:
                    modified_time = datetime.fromisoformat(
                        item_data["updated_at"].replace("Z", "+00:00")
                    ).timestamp()
                except ValueError:
                    pass
            
            return ContentItem(
                id=item_id,
                path=path,
                content_type=content_type,
                size=size,
                modified_time=modified_time,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error parsing item {item_data}: {e}")
            return None
    
    def _determine_content_type(self, item_data: dict[str, Any]) -> str:
        """Determine content type from item data."""
        # Check explicit type field
        item_type = item_data.get("type")
        if item_type:
            return item_type
        
        # Guess from path
        path = item_data.get("path", "")
        mime_type, _ = mimetypes.guess_type(path)
        
        if mime_type:
            return mime_type
        
        # Default fallback
        return "application/octet-stream"
    
    def _apply_content_filters(
        self, 
        items: list[ContentItem], 
        config: MySourceConfig
    ) -> list[ContentItem]:
        """Apply include/exclude patterns to filter content."""
        if not config.include_patterns and not config.exclude_patterns:
            return items
        
        filtered_items = []
        
        for item in items:
            # Skip directories if not doing recursive discovery
            if item.content_type == "directory":
                filtered_items.append(item)
                continue
            
            # Apply include patterns
            if config.include_patterns:
                if not any(self._matches_pattern(item.path, pattern) 
                          for pattern in config.include_patterns):
                    continue
            
            # Apply exclude patterns
            if config.exclude_patterns:
                if any(self._matches_pattern(item.path, pattern) 
                      for pattern in config.exclude_patterns):
                    continue
            
            filtered_items.append(item)
        
        return filtered_items
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)
    
    async def read_content(self, item: ContentItem) -> str:
        """Read content from a content item."""
        if not self.session:
            raise SourceError("Session not initialized")
        
        # Get download URL from metadata
        download_url = item.metadata.get("url")
        if not download_url:
            raise SourceError(f"No download URL for item {item.id}")
        
        # Rate limiting
        if self._rate_limiter:
            await self._rate_limiter.acquire()
        
        try:
            async with self.session.get(download_url) as response:
                response.raise_for_status()
                
                # Handle different content encodings
                encoding = item.metadata.get("encoding")
                if encoding == "base64":
                    import base64
                    content_data = await response.json()
                    content_bytes = base64.b64decode(content_data.get("content", ""))
                    return content_bytes.decode("utf-8", errors="ignore")
                else:
                    # Plain text content
                    return await response.text()
                    
        except aiohttp.ClientError as e:
            raise SourceError(f"Failed to read content for {item.id}: {e}") from e
        except UnicodeDecodeError as e:
            raise SourceError(f"Failed to decode content for {item.id}: {e}") from e
    
    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]:
        """Get extended metadata for content item."""
        # Return existing metadata plus any additional info
        metadata = item.metadata.copy() if item.metadata else {}
        
        # Add computed metadata
        metadata.update({
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "discovered_at": datetime.now().isoformat(),
            "file_extension": Path(item.path).suffix.lower(),
            "language": self._detect_language(item.path),
        })
        
        return metadata
    
    def _detect_language(self, path: str) -> str | None:
        """Detect programming language from file path."""
        extension_map = {
            ".py": "python",
            ".js": "javascript", 
            ".ts": "typescript",
            ".jsx": "jsx",
            ".tsx": "tsx",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".rs": "rust",
            ".go": "go",
            ".php": "php",
            ".rb": "ruby",
            ".css": "css",
            ".html": "html",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".xml": "xml",
            ".sql": "sql",
        }
        
        extension = Path(path).suffix.lower()
        return extension_map.get(extension)
    
    async def watch_changes(
        self, 
        config: SourceConfig, 
        callback: Callable[[list[ContentItem]], None]
    ) -> SourceWatcher:
        """Watch for content changes (optional capability)."""
        if not isinstance(config, MySourceConfig):
            raise ValueError("Expected MySourceConfig")
        
        # Create and return a watcher
        watcher = MySourceWatcher(self, config, callback)
        self._watchers.append(watcher)
        return watcher
    
    async def validate_source(self, config: SourceConfig) -> bool:
        """Validate source configuration and connectivity."""
        if not isinstance(config, MySourceConfig):
            return False
        
        try:
            await self._initialize_session(config)
            
            # Test connectivity with a simple request
            async with self.session.get(config.base_url) as response:
                return response.status == 200
                
        except Exception:
            return False
    
    async def health_check(self) -> bool:
        """Check source health and connectivity."""
        if not self.session:
            return False
        
        try:
            # Simple health check - ping a known endpoint
            async with self.session.get("/health") as response:
                return response.status == 200
        except Exception:
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Stop all watchers
        for watcher in self._watchers:
            if watcher.is_active:
                await watcher.stop()
        
        # Close session
        if self.session:
            await self.session.close()
            self.session = None
```

### Step 3: Implement Source Watcher (Optional)

```python
from codeweaver.cw_types import SourceWatcher
import asyncio
from datetime import datetime

class MySourceWatcher:
    """Watcher for source changes."""
    
    def __init__(
        self, 
        source: MyDataSource, 
        config: MySourceConfig, 
        callback: Callable[[list[ContentItem]], None]
    ):
        self.source = source
        self.config = config
        self.callback = callback
        self._active = False
        self._task: asyncio.Task | None = None
        self._last_check: datetime | None = None
        self._known_items: dict[str, ContentItem] = {}
    
    async def start(self) -> None:
        """Start watching for changes."""
        if self._active:
            return
        
        self._active = True
        self._task = asyncio.create_task(self._watch_loop())
    
    async def stop(self) -> None:
        """Stop watching for changes."""
        self._active = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    @property
    def is_active(self) -> bool:
        """Check if watcher is currently active."""
        return self._active
    
    def get_stats(self) -> dict[str, Any]:
        """Get watcher statistics."""
        return {
            "active": self._active,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "known_items_count": len(self._known_items),
            "watcher_type": "polling"
        }
    
    async def _watch_loop(self) -> None:
        """Main watching loop."""
        # Initial discovery
        current_items = await self.source.discover_content(self.config)
        self._known_items = {item.id: item for item in current_items}
        
        while self._active:
            try:
                await asyncio.sleep(30)  # Poll every 30 seconds
                
                # Discover current state
                current_items = await self.source.discover_content(self.config)
                current_items_dict = {item.id: item for item in current_items}
                
                # Find changes
                changes = []
                
                # New or modified items
                for item_id, item in current_items_dict.items():
                    if item_id not in self._known_items:
                        # New item
                        changes.append(item)
                    elif (self._known_items[item_id].modified_time != item.modified_time):
                        # Modified item
                        changes.append(item)
                
                # Deleted items (create tombstone entries)
                for item_id in self._known_items:
                    if item_id not in current_items_dict:
                        # Create deletion marker
                        deleted_item = self._known_items[item_id]
                        deleted_item.metadata = deleted_item.metadata or {}
                        deleted_item.metadata["deleted"] = True
                        changes.append(deleted_item)
                
                # Update known items
                self._known_items = current_items_dict
                self._last_check = datetime.now()
                
                # Notify callback if there are changes
                if changes:
                    self.callback(changes)
                    
            except Exception as e:
                print(f"Error in watch loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
```

### Step 4: Create Plugin Interface

```python
from codeweaver.factories.plugin_protocols import SourcePlugin
from codeweaver.cw_types import (
    ComponentType, BaseCapabilities, BaseComponentInfo, 
    ValidationResult, SourceCapabilities
)

class MySourcePlugin(SourcePlugin):
    """Plugin interface for MyDataSource."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "my_custom_source"
    
    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.SOURCE
    
    @classmethod
    def get_capabilities(cls) -> BaseCapabilities:
        return SourceCapabilities(
            capabilities=[
                SourceCapability.CONTENT_DISCOVERY,
                SourceCapability.CONTENT_READING,
                SourceCapability.METADATA_EXTRACTION,
                SourceCapability.CHANGE_WATCHING,
                SourceCapability.AUTHENTICATION,
                SourceCapability.FILTERING,
            ],
            supported_schemes=["http", "https"],
            max_file_size=1024 * 1024 * 10,  # 10MB
            supports_pagination=True,
            supports_streaming=False
        )
    
    @classmethod
    def get_component_info(cls) -> BaseComponentInfo:
        return BaseComponentInfo(
            name="my_custom_source",
            display_name="My Custom Data Source",
            description="Custom data source for API-based content discovery",
            component_type=ComponentType.SOURCE,
            version="1.0.0",
            author="Your Name",
            homepage="https://github.com/yourname/my-source"
        )
    
    @classmethod
    def validate_config(cls, config: MySourceConfig) -> ValidationResult:
        """Validate source configuration."""
        errors = []
        warnings = []
        
        # Validate base URL
        if not config.base_url:
            errors.append("Base URL is required")
        else:
            from urllib.parse import urlparse
            parsed = urlparse(config.base_url)
            if not parsed.scheme or not parsed.netloc:
                errors.append("Invalid base URL format")
        
        # Validate authentication
        if config.username and not config.password:
            errors.append("Password required when username is provided")
        
        # Validate rate limiting
        if config.requests_per_second > 100:
            warnings.append("High request rate may cause API throttling")
        
        if config.concurrent_requests > 20:
            warnings.append("High concurrency may overwhelm the API")
        
        # Validate file size limits
        if config.max_file_size > 100 * 1024 * 1024:  # 100MB
            warnings.append("Large file size limit may impact performance")
        
        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def get_dependencies(cls) -> list[str]:
        """Get required dependencies."""
        return ["aiohttp", "pydantic"]
    
    @classmethod
    def get_source_class(cls) -> type[DataSource]:
        return MyDataSource
```

### Step 5: Register the Source

```python
from codeweaver.factories.codeweaver_factory import CodeWeaverFactory

# Create factory instance
factory = CodeWeaverFactory()

# Register source
factory.register_source(
    "my_custom_source",
    MyDataSource,
    MySourcePlugin.get_capabilities(),
    MySourcePlugin.get_component_info()
)
```

## 🔧 Advanced Features

### Streaming Source Implementation

For large datasets or real-time data:

```python
from typing import AsyncIterator

class StreamingDataSource(AbstractDataSource):
    """Source with streaming capabilities."""
    
    async def discover_content_stream(
        self, 
        config: SourceConfig
    ) -> AsyncIterator[list[ContentItem]]:
        """Stream content discovery in batches."""
        batch_size = 100
        offset = 0
        
        while True:
            batch = await self._discover_batch(config, offset, batch_size)
            if not batch:
                break
            
            yield batch
            offset += batch_size
    
    async def read_content_stream(
        self, 
        item: ContentItem
    ) -> AsyncIterator[str]:
        """Stream content reading for large files."""
        chunk_size = 8192
        
        async with self.session.get(item.metadata["url"]) as response:
            async for chunk in response.content.iter_chunked(chunk_size):
                yield chunk.decode("utf-8", errors="ignore")
```

### Database Source Implementation

For SQL/NoSQL databases:

```python
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine

class DatabaseSource(AbstractDataSource):
    """Database source implementation."""
    
    def __init__(self):
        super().__init__(SourceProvider.DATABASE, "database_source")
        self.engine = None
    
    async def _initialize_database(self, config: DatabaseConfig):
        """Initialize database connection."""
        if not self.engine:
            self.engine = create_async_engine(
                config.database_url,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow
            )
    
    async def discover_content(self, config: DatabaseConfig) -> list[ContentItem]:
        """Discover content from database tables."""
        await self._initialize_database(config)
        
        items = []
        
        async with self.engine.begin() as conn:
            # Query for discoverable content
            query = """
                SELECT 
                    id, 
                    title, 
                    content, 
                    updated_at,
                    metadata
                FROM content_table 
                WHERE status = 'published'
                ORDER BY updated_at DESC
            """
            
            result = await conn.execute(query)
            
            for row in result:
                item = ContentItem(
                    id=str(row.id),
                    path=f"db://{config.table_name}/{row.id}",
                    content_type="text/plain",
                    size=len(row.content or ""),
                    modified_time=row.updated_at.timestamp(),
                    metadata={
                        "title": row.title,
                        "table": config.table_name,
                        **row.metadata
                    }
                )
                items.append(item)
        
        return items
    
    async def read_content(self, item: ContentItem) -> str:
        """Read content from database."""
        item_id = item.id
        table_name = item.metadata["table"]
        
        async with self.engine.begin() as conn:
            query = f"SELECT content FROM {table_name} WHERE id = $1"
            result = await conn.execute(query, [item_id])
            row = result.fetchone()
            
            return row.content if row else ""
```

### Git Repository Source

For version control integration:

```python
import git
from pathlib import Path

class GitRepositorySource(AbstractDataSource):
    """Git repository source implementation."""
    
    def __init__(self):
        super().__init__(SourceProvider.GIT, "git_source")
        self.repo: git.Repo | None = None
    
    async def discover_content(self, config: GitConfig) -> list[ContentItem]:
        """Discover files in Git repository."""
        await self._clone_or_update_repo(config)
        
        items = []
        repo_path = Path(config.local_path)
        
        # Walk repository files
        for file_path in repo_path.rglob("*"):
            if file_path.is_file() and not self._is_git_file(file_path):
                relative_path = file_path.relative_to(repo_path)
                
                # Get file info from Git
                try:
                    commits = list(self.repo.iter_commits(paths=str(relative_path), max_count=1))
                    latest_commit = commits[0] if commits else None
                    
                    item = ContentItem(
                        id=str(relative_path),
                        path=str(relative_path),
                        content_type=self._get_mime_type(file_path),
                        size=file_path.stat().st_size,
                        modified_time=latest_commit.committed_date if latest_commit else None,
                        metadata={
                            "commit_sha": latest_commit.hexsha if latest_commit else None,
                            "author": str(latest_commit.author) if latest_commit else None,
                            "branch": self.repo.active_branch.name,
                            "repository_url": config.repository_url
                        }
                    )
                    items.append(item)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return items
    
    async def _clone_or_update_repo(self, config: GitConfig):
        """Clone or update Git repository."""
        repo_path = Path(config.local_path)
        
        if repo_path.exists():
            # Update existing repo
            self.repo = git.Repo(repo_path)
            self.repo.remotes.origin.pull()
        else:
            # Clone new repo
            self.repo = git.Repo.clone_from(
                config.repository_url, 
                repo_path,
                branch=config.branch
            )
    
    def _is_git_file(self, file_path: Path) -> bool:
        """Check if file is part of Git metadata."""
        return ".git" in file_path.parts
```

## 🧪 Testing Your Data Source

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, patch
from my_package.source import MyDataSource, MySourceConfig
from codeweaver.cw_types import ContentItem

@pytest.fixture
def source_config():
    return MySourceConfig(
        base_url="https://api.example.com",
        api_key="test-key",
        include_patterns=["*.py", "*.md"],
        max_file_size=1024*1024
    )

@pytest.fixture
def source():
    return MyDataSource()

@pytest.fixture
def mock_http_session():
    session = AsyncMock()
    response = AsyncMock()
    response.json.return_value = {
        "items": [
            {
                "id": "test1",
                "path": "test1.py",
                "type": "file",
                "size": 1024,
                "download_url": "https://api.example.com/download/test1",
                "updated_at": "2023-01-01T00:00:00Z"
            },
            {
                "id": "test2",
                "path": "test2.md",
                "type": "file", 
                "size": 2048,
                "download_url": "https://api.example.com/download/test2",
                "updated_at": "2023-01-02T00:00:00Z"
            }
        ]
    }
    session.get.return_value.__aenter__.return_value = response
    return session

class TestMyDataSource:
    """Test suite for MyDataSource."""
    
    async def test_discover_content(self, source, source_config, mock_http_session):
        """Test content discovery."""
        source.session = mock_http_session
        
        items = await source.discover_content(source_config)
        
        assert len(items) == 2
        assert items[0].id == "test1"
        assert items[0].path == "test1.py"
        assert items[0].size == 1024
        
        # Verify API was called
        mock_http_session.get.assert_called_once()
    
    async def test_read_content(self, source, mock_http_session):
        """Test content reading."""
        source.session = mock_http_session
        
        # Mock content response
        content_response = AsyncMock()
        content_response.text.return_value = "print('Hello World')"
        mock_http_session.get.return_value.__aenter__.return_value = content_response
        
        item = ContentItem(
            id="test1",
            path="test1.py",
            content_type="text/plain",
            metadata={"url": "https://api.example.com/download/test1"}
        )
        
        content = await source.read_content(item)
        
        assert content == "print('Hello World')"
    
    async def test_content_filtering(self, source, source_config):
        """Test content filtering with patterns."""
        items = [
            ContentItem(id="1", path="test.py", content_type="text/plain"),
            ContentItem(id="2", path="test.js", content_type="text/plain"),
            ContentItem(id="3", path="README.md", content_type="text/plain"),
            ContentItem(id="4", path="config.json", content_type="application/json")
        ]
        
        filtered = source._apply_content_filters(items, source_config)
        
        # Should only include .py and .md files
        assert len(filtered) == 2
        assert filtered[0].path == "test.py"
        assert filtered[1].path == "README.md"
    
    async def test_validate_source(self, source, source_config, mock_http_session):
        """Test source validation."""
        source.session = mock_http_session
        
        # Mock successful response
        mock_http_session.get.return_value.__aenter__.return_value.status = 200
        
        is_valid = await source.validate_source(source_config)
        
        assert is_valid is True
    
    async def test_health_check(self, source, mock_http_session):
        """Test health check."""
        source.session = mock_http_session
        
        # Mock health endpoint response
        mock_http_session.get.return_value.__aenter__.return_value.status = 200
        
        healthy = await source.health_check()
        
        assert healthy is True
    
    def test_capabilities(self, source):
        """Test source capabilities."""
        capabilities = source.get_capabilities()
        
        assert SourceCapability.CONTENT_DISCOVERY in capabilities.capabilities
        assert SourceCapability.CONTENT_READING in capabilities.capabilities
        assert SourceCapability.METADATA_EXTRACTION in capabilities.capabilities
```

### Integration Tests

```python
@pytest.mark.integration
class TestDataSourceIntegration:
    """Integration tests with real APIs."""
    
    @pytest.fixture
    def real_source_config(self):
        return MySourceConfig(
            base_url=os.getenv("TEST_API_URL"),
            api_key=os.getenv("TEST_API_KEY"),
            timeout=30
        )
    
    async def test_real_discovery(self, real_source_config):
        """Test with real API (requires API key)."""
        if not os.getenv("TEST_API_KEY"):
            pytest.skip("No API key provided for integration testing")
        
        source = MyDataSource()
        
        try:
            items = await source.discover_content(real_source_config)
            
            assert isinstance(items, list)
            if items:
                item = items[0]
                assert isinstance(item, ContentItem)
                assert item.id
                assert item.path
                
                # Test reading content
                content = await source.read_content(item)
                assert isinstance(content, str)
                
        finally:
            await source.cleanup()
    
    async def test_change_watching(self, real_source_config):
        """Test change watching functionality."""
        if not os.getenv("TEST_API_KEY"):
            pytest.skip("No API key provided for integration testing")
        
        source = MyDataSource()
        changes_detected = []
        
        def change_callback(changes: list[ContentItem]):
            changes_detected.extend(changes)
        
        try:
            watcher = await source.watch_changes(real_source_config, change_callback)
            await watcher.start()
            
            # Wait briefly for initial discovery
            await asyncio.sleep(5)
            
            assert watcher.is_active
            stats = watcher.get_stats()
            assert stats["active"] is True
            
            await watcher.stop()
            assert not watcher.is_active
            
        finally:
            await source.cleanup()
```

## 📊 Performance Guidelines

### Efficient Content Discovery
- Use pagination for large datasets
- Implement parallel discovery for multiple sources
- Cache discovery results when appropriate
- Use incremental discovery for change detection

### Memory Management
- Stream large files instead of loading into memory
- Implement content size limits
- Use lazy loading for metadata
- Clean up resources properly

### Rate Limiting
- Respect API rate limits
- Implement exponential backoff
- Use connection pooling
- Monitor API quota usage

### Error Handling
- Implement comprehensive error handling
- Provide meaningful error messages
- Support graceful degradation
- Log errors for debugging

## 🚀 Next Steps

- **[Service Development →](./services.md)**: Build middleware services
- **[Testing Framework →](./testing.md)**: Comprehensive testing strategies
- **[Performance Guidelines →](./performance.md)**: Optimization best practices
- **[Protocol Reference →](../reference/protocols.md)**: Complete protocol documentation