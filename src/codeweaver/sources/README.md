# CodeWeaver Data Source Abstraction System

This directory contains the data source abstraction system for CodeWeaver, designed to support multiple content sources beyond just file systems while maintaining full backward compatibility.

## 🏗️ Architecture Overview

The data source abstraction system follows a plugin-based architecture that allows CodeWeaver to index content from various sources:

- **File Systems** (fully implemented)
- **Git Repositories** (placeholder)
- **Databases** (placeholder)
- **REST/GraphQL APIs** (placeholder)
- **Web Crawlers** (placeholder)
- **Enterprise Sources** (future: SharePoint, Slack, Confluence)

## 📁 Directory Structure

```
src/codeweaver/sources/
├── __init__.py                 # Public API exports
├── README.md                   # This documentation
├── base.py                     # Core protocols and data structures
├── config.py                   # Configuration extensions
├── factory.py                  # Source factory and registry
├── integration.py              # Backward compatibility integration
├── filesystem.py               # File system source (fully implemented)
├── git.py                      # Git repository source (placeholder)
├── database.py                 # Database source (placeholder)
├── api.py                      # API source (placeholder)
└── web.py                      # Web crawler source (placeholder)
```

## 🔌 Core Components

### 1. DataSource Protocol (`base.py`)

The foundation of the system is the `DataSource` protocol that all sources must implement:

```python
@runtime_checkable
class DataSource(Protocol):
    def get_capabilities(self) -> set[SourceCapability]: ...
    async def discover_content(self, config: SourceConfig) -> list[ContentItem]: ...
    async def read_content(self, item: ContentItem) -> str: ...
    async def watch_changes(self, config: SourceConfig, callback: Callable) -> SourceWatcher: ...
    async def validate_source(self, config: SourceConfig) -> bool: ...
    async def get_content_metadata(self, item: ContentItem) -> dict[str, Any]: ...
```

### 2. ContentItem Data Structure

Universal representation of discoverable content:

```python
class ContentItem:
    path: str                    # Universal identifier
    content_type: str           # 'file', 'url', 'database', 'api', 'git'
    metadata: dict[str, Any]    # Source-specific metadata
    last_modified: datetime     # Last modification timestamp
    size: int                   # Content size in bytes
    language: str               # Detected programming language
    source_id: str              # Source identifier
    version: str                # Version/commit identifier
    checksum: str               # Content checksum
```

### 3. Source Capabilities

Sources declare their capabilities using the `SourceCapability` enum:

- `CONTENT_DISCOVERY` - Can discover content items
- `CONTENT_READING` - Can read content from items
- `CHANGE_WATCHING` - Can watch for content changes
- `INCREMENTAL_SYNC` - Supports incremental updates
- `VERSION_HISTORY` - Provides version/commit history
- `METADATA_EXTRACTION` - Rich metadata extraction
- `REAL_TIME_UPDATES` - Real-time change notifications
- `BATCH_PROCESSING` - Efficient batch operations
- `CONTENT_DEDUPLICATION` - Built-in deduplication
- `RATE_LIMITING` - Built-in rate limiting
- `AUTHENTICATION` - Supports authentication
- `PAGINATION` - Supports paginated discovery

## 🔧 Implementation Status

### ✅ Fully Implemented

#### File System Source (`filesystem.py`)
- **Capabilities**: Content discovery, reading, change watching, metadata extraction, batch processing, deduplication
- **Features**:
  - Gitignore support via existing FileFilter
  - Intelligent file filtering
  - Change watching with configurable intervals
  - Full backward compatibility with existing code
  - Size limits and pattern filtering
  - Language detection integration

### 🚧 Placeholder Implementations

The following sources have placeholder implementations with full interface compliance but require additional dependencies for full functionality:

#### Git Repository Source (`git.py`)
- **Purpose**: Index git repositories with branch/commit support
- **Future Dependencies**: GitPython or pygit2
- **Planned Features**:
  - Branch and commit-specific indexing
  - Version history tracking
  - Incremental sync with git fetch/pull
  - Multiple repository support

#### Database Source (`database.py`)
- **Purpose**: Index database schemas, procedures, and sample data
- **Future Dependencies**: Database-specific drivers (psycopg2, pymongo, etc.)
- **Planned Features**:
  - SQL and NoSQL database support
  - Schema extraction as indexable content
  - Sample data indexing (configurable)
  - Change detection via triggers/polling

#### API Source (`api.py`)
- **Purpose**: Index REST/GraphQL API documentation and schemas
- **Future Dependencies**: httpx, requests
- **Planned Features**:
  - OpenAPI/Swagger specification extraction
  - GraphQL schema discovery
  - API documentation indexing
  - Sample response inclusion

#### Web Crawler Source (`web.py`)
- **Purpose**: Crawl documentation sites and technical content
- **Future Dependencies**: Scrapy, BeautifulSoup4
- **Planned Features**:
  - Politeness policies and robots.txt respect
  - Content extraction from HTML/Markdown
  - Documentation site specialization
  - Rate limiting and caching

## ⚙️ Configuration

### Basic Data Sources Configuration

```toml
[data_sources]
enabled = true
default_source_type = "filesystem"
max_concurrent_sources = 5
enable_content_deduplication = true
enable_metadata_extraction = true

# File System Source
[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1
source_id = "main_codebase"

[data_sources.sources.config]
root_path = "."
use_gitignore = true
additional_ignore_patterns = ["node_modules", ".venv"]
max_file_size_mb = 1
enable_change_watching = false
patterns = ["**/*.py", "**/*.js", "**/*.ts"]
```

### Multiple Sources Configuration

```toml
# Multiple file system sources
[[data_sources.sources]]
type = "filesystem"
enabled = true
priority = 1
source_id = "main_repo"

[data_sources.sources.config]
root_path = "/path/to/main/repo"
use_gitignore = true

[[data_sources.sources]]
type = "filesystem"  
enabled = true
priority = 2
source_id = "shared_libs"

[data_sources.sources.config]
root_path = "/path/to/shared/libraries"
patterns = ["**/*.py", "**/*.js"]
```

## 🔄 Backward Compatibility

The system maintains **100% backward compatibility** with existing CodeWeaver deployments:

1. **Default Behavior**: Without data source configuration, the system automatically creates a file system source using existing configuration
2. **Same Interface**: The `index_codebase` method signature remains unchanged
3. **Migration Path**: Legacy configurations are automatically migrated to the new system
4. **Gradual Adoption**: Teams can migrate to the new system incrementally

### Automatic Migration

When no data sources are configured, the system automatically creates a file system source:

```python
# Legacy configuration is automatically converted to:
{
    "type": "filesystem",
    "enabled": True,
    "priority": 1,
    "source_id": "default_filesystem",
    "config": {
        "root_path": ".",
        "use_gitignore": True,
        "additional_ignore_patterns": [...],  # From existing config
        "max_file_size_mb": 1,               # From existing config
        # ... other settings from legacy config
    }
}
```

## 🚀 Usage Examples

### Creating Data Sources Programmatically

```python
from codeweaver.sources import SourceFactory, FileSystemSourceConfig

# Create source factory
factory = SourceFactory()

# Create file system source
config: FileSystemSourceConfig = {
    "enabled": True,
    "root_path": "/path/to/code",
    "use_gitignore": True,
    "patterns": ["**/*.py", "**/*.js"],
}

source = factory.create_source("filesystem", config)

# Discover and read content
content_items = await source.discover_content(config)
for item in content_items:
    content = await source.read_content(item)
    print(f"Content from {item.path}: {len(content)} characters")
```

### Integration with Existing Server

```python
from codeweaver.sources.integration import DataSourceManager
from codeweaver.sources.config import DataSourcesConfig

# Create data source manager
config = DataSourcesConfig()
config.add_source_config(
    source_type="filesystem",
    config={"root_path": ".", "use_gitignore": True},
    enabled=True,
    priority=1
)

manager = DataSourceManager(config)
await manager.initialize_sources()

# Discover content from all sources
content_items = await manager.discover_all_content()
```

## 🔧 Extending the System

### Creating a Custom Data Source

1. **Implement the DataSource Protocol**:

```python
from codeweaver.sources.base import AbstractDataSource, SourceCapability

class CustomSource(AbstractDataSource):
    def __init__(self, source_id: str | None = None):
        super().__init__("custom", source_id)
    
    def get_capabilities(self) -> set[SourceCapability]:
        return {
            SourceCapability.CONTENT_DISCOVERY,
            SourceCapability.CONTENT_READING,
        }
    
    async def discover_content(self, config: SourceConfig) -> list[ContentItem]:
        # Implementation here
        pass
    
    async def read_content(self, item: ContentItem) -> str:
        # Implementation here
        pass
```

2. **Register the Source**:

```python
from codeweaver.sources import get_source_registry

registry = get_source_registry()
registry.register("custom", CustomSource)
```

3. **Use in Configuration**:

```toml
[[data_sources.sources]]
type = "custom"
enabled = true
priority = 1

[data_sources.sources.config]
# Custom configuration here
```

## 🧪 Testing

The system includes comprehensive testing support:

```python
from codeweaver.sources.factory import get_source_factory

# Validate source configuration without creating
factory = get_source_factory()
is_valid = await factory.validate_source_config("filesystem", config)

# List available sources and capabilities
sources_info = factory.list_available_sources()
print(f"Available sources: {list(sources_info.keys())}")
```

## 🔍 Future Enhancements

### Planned Features

1. **Enterprise Sources**:
   - SharePoint integration for corporate documentation
   - Slack/Teams chat history indexing
   - Confluence and wiki system support

2. **Advanced Content Processing**:
   - Multi-language content translation
   - Content summarization and extraction
   - Semantic deduplication using embeddings

3. **Performance Optimizations**:
   - Distributed content discovery
   - Incremental indexing with change detection
   - Content caching and invalidation strategies

4. **Monitoring and Observability**:
   - Source health monitoring
   - Content discovery metrics
   - Performance profiling and optimization

### Integration Opportunities

1. **CI/CD Integration**: Automatic reindexing on repository changes
2. **Cloud Storage**: Support for S3, GCS, Azure Blob storage
3. **Documentation Platforms**: GitBook, Notion, Obsidian integration
4. **Development Tools**: IDE integration, code review system support

## 📚 Related Documentation

- [EXTENSIBILITY_DESIGN.md](../../EXTENSIBILITY_DESIGN.md) - Overall extensibility architecture
- [PROVIDER_SYSTEM_IMPLEMENTATION.md](../../PROVIDER_SYSTEM_IMPLEMENTATION.md) - Provider system design
- [Configuration Guide](../config.py) - Main configuration system
- [MCP Server Documentation](../server.py) - Main server implementation

This data source abstraction system enables CodeWeaver to evolve from a file-system-only indexing tool into a comprehensive code intelligence platform that can index and search across any content source while maintaining the simplicity and performance that make it valuable.