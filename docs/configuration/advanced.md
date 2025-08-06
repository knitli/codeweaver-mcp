<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Advanced Configuration

This guide covers advanced configuration patterns, custom factory setups, plugin development, and enterprise deployment scenarios for CodeWeaver.

## Factory Pattern Configuration

CodeWeaver uses a factory pattern system for extensible component creation and dependency injection.

### CodeWeaverFactory Configuration

```toml
[factory]
# Factory behavior
enable_caching = true
lazy_loading = true
strict_validation = true

# Component discovery
auto_discovery = true
scan_paths = ["src/", "plugins/", "extensions/"]
discovery_patterns = ["*_provider.py", "*_backend.py", "*_service.py"]

# Plugin system
plugin_directories = ["~/.codeweaver/plugins", "./plugins"]
enable_user_plugins = true
plugin_security_scan = true
```

### Component Registration

#### Programmatic Registration

```python
from codeweaver.factories import codeweaver_factory
from codeweaver.cw_types import ComponentType

# Register custom embedding provider
codeweaver_factory.register_component(
    ComponentType.EMBEDDING_PROVIDER,
    "my_embedder",
    MyCustomEmbedder,
    config_schema=MyEmbedderConfig
)

# Register with metadata
codeweaver_factory.register_component(
    ComponentType.VECTOR_BACKEND,
    "my_backend",
    MyVectorBackend,
    metadata={
        "version": "1.0.0",
        "author": "Your Name",
        "description": "Custom vector backend",
        "capabilities": ["hybrid_search", "filtering"]
    }
)
```

#### Configuration-Based Registration

```toml
[factory.components.embedding_providers]
my_embedder = "mypackage.providers:MyCustomEmbedder"

[factory.components.vector_backends]
my_backend = "mypackage.backends:MyVectorBackend"

[factory.components.services]
my_service = "mypackage.services:MyCustomService"
```

### Dependency Injection Configuration

```toml
[factory.dependency_injection]
# Enable automatic dependency resolution
auto_wire = true
circular_dependency_detection = true
lazy_initialization = true

# Singleton configuration
[factory.singletons]
config_manager = true
services_manager = true
telemetry_client = false

# Factory scopes
[factory.scopes]
default_scope = "singleton"  # singleton, transient, scoped
request_scope = "scoped"
```

## Plugin System Configuration

### Plugin Discovery and Loading

```toml
[plugins]
# Discovery settings
enable_discovery = true
discovery_paths = [
    "~/.codeweaver/plugins",
    "./plugins",
    "/usr/local/lib/codeweaver/plugins"
]

# Security settings
enable_plugin_validation = true
require_signatures = false
allowed_plugin_origins = ["official", "verified"]

# Loading behavior
lazy_loading = true
load_on_demand = true
plugin_timeout = 30
```

### Plugin Configuration Schema

```toml
[plugins.my_custom_plugin]
enabled = true
priority = 100
config_file = "plugins/my_plugin/config.toml"

# Plugin-specific settings
[plugins.my_custom_plugin.settings]
api_key = "plugin-api-key"
endpoint = "https://api.example.com"
batch_size = 50
```

### Entry Point Configuration

```toml
# pyproject.toml for plugin development
[project.entry-points."codeweaver.embedding_providers"]
my_provider = "mypackage.providers:MyProvider"

[project.entry-points."codeweaver.vector_backends"]
my_backend = "mypackage.backends:MyBackend"

[project.entry-points."codeweaver.data_sources"]
my_source = "mypackage.sources:MyDataSource"
```

## Multi-Environment Configuration

### Environment-Specific Profiles

```toml
# Base configuration
[profile]
name = "base"

[providers.voyage_ai]
model = "voyage-code-3"

# Development overrides
[profiles.development]
inherits = "base"

[profiles.development.backend]
provider = "docarray"
backend_type = "memory"

[profiles.development.services]
enable_caching = false
log_level = "DEBUG"

# Production overrides
[profiles.production]
inherits = "base"

[profiles.production.backend]
provider = "qdrant"
enable_hybrid_search = true
prefer_grpc = true

[profiles.production.services]
enable_health_monitoring = true
enable_metrics = true
```

### Configuration Inheritance

```toml
# base.toml
[chunking]
max_chunk_size = 1500
min_chunk_size = 50

[indexing]
batch_size = 8

# development.toml
[inherits]
base = "base.toml"

[chunking]
# Override specific settings
max_chunk_size = 1000  # Smaller chunks for testing

# production.toml
[inherits]
base = "base.toml"

[indexing]
# Override for production
batch_size = 16
concurrent_files = 20
```

### Dynamic Configuration Loading

```python
from codeweaver.config import ConfigManager

# Load environment-specific configuration
config_manager = ConfigManager()

# Development
dev_config = config_manager.load_profile("development")

# Production with environment overrides
prod_config = config_manager.load_profile(
    "production",
    env_overrides=True,
    validate_required=True
)

# Runtime configuration updates
config_manager.update_config({
    "chunking": {"max_chunk_size": 2000}
})
```

## Custom Data Sources

### Data Source Implementation

```python
from codeweaver.cw_types import DataSource, ContentItem
from typing import AsyncIterator

class GitDataSource(DataSource):
    def __init__(self, config: dict):
        self.repo_url = config["repo_url"]
        self.branch = config.get("branch", "main")
        self.auth_token = config.get("auth_token")

    async def discover_content(self, **kwargs) -> AsyncIterator[ContentItem]:
        """Discover content from Git repository"""
        repo = await self._clone_repo()

        for file_path in self._walk_files(repo):
            content = await self._read_file(repo, file_path)

            yield ContentItem(
                content=content,
                metadata={
                    "source_type": "git",
                    "repo_url": self.repo_url,
                    "file_path": file_path,
                    "branch": self.branch,
                    "commit_hash": await self._get_commit_hash(repo, file_path)
                }
            )

    async def health_check(self) -> bool:
        """Check if repository is accessible"""
        try:
            await self._test_connection()
            return True
        except Exception:
            return False
```

### Data Source Configuration

```toml
[data_sources.git]
provider = "git"
repo_url = "https://github.com/user/repo.git"
branch = "main"
auth_token = "${GIT_AUTH_TOKEN}"

# Filtering
include_patterns = ["*.py", "*.js", "*.md"]
exclude_patterns = ["test/**", "docs/**"]

# Performance
batch_size = 10
max_file_size = 1048576
concurrent_downloads = 5
```

### Multiple Data Sources

```toml
# Filesystem source
[data_sources.filesystem]
provider = "filesystem"
base_path = "/path/to/code"
use_gitignore = true

# Git source
[data_sources.git_remote]
provider = "git"
repo_url = "https://github.com/user/repo.git"

# Database source
[data_sources.database]
provider = "database"
connection_string = "postgresql://user:pass@host/db"
query = "SELECT content, metadata FROM code_files"

# API source
[data_sources.api]
provider = "api"
endpoint = "https://api.example.com/code"
auth_header = "Bearer ${API_TOKEN}"
```

## Enterprise Deployment Configuration

### High Availability Setup

```toml
[enterprise.high_availability]
# Load balancing
enable_load_balancing = true
backend_replicas = [
    "https://qdrant-1.internal:6333",
    "https://qdrant-2.internal:6333",
    "https://qdrant-3.internal:6333"
]

# Failover configuration
enable_failover = true
health_check_interval = 30
failover_timeout = 5

# Cluster coordination
cluster_mode = true
cluster_nodes = [
    "codeweaver-1.internal:8080",
    "codeweaver-2.internal:8080"
]
```

### Security Configuration

```toml
[enterprise.security]
# Authentication
auth_provider = "ldap"  # ldap, oauth2, saml
ldap_url = "ldap://directory.company.com"
ldap_base_dn = "ou=users,dc=company,dc=com"

# Authorization
enable_rbac = true
default_role = "user"
admin_roles = ["admin", "codeweaver-admin"]

# Encryption
encrypt_at_rest = true
encryption_key_file = "/etc/codeweaver/encryption.key"
tls_cert_file = "/etc/codeweaver/tls.crt"
tls_key_file = "/etc/codeweaver/tls.key"

# Audit logging
enable_audit_logging = true
audit_log_file = "/var/log/codeweaver/audit.log"
audit_log_format = "json"
```

### Monitoring and Observability

```toml
[enterprise.monitoring]
# Metrics collection
metrics_provider = "prometheus"
metrics_endpoint = "/metrics"
metrics_port = 9090

# Distributed tracing
enable_tracing = true
tracing_provider = "jaeger"
jaeger_endpoint = "http://jaeger:14268/api/traces"

# Health checks
health_check_endpoint = "/health"
readiness_endpoint = "/ready"
liveness_endpoint = "/live"

# Alerting
[enterprise.monitoring.alerts]
webhook_url = "https://alerts.company.com/webhook"
email_notifications = ["ops@company.com"]
slack_webhook = "https://hooks.slack.com/services/..."
```

### Performance Tuning

```toml
[enterprise.performance]
# Connection pooling
max_connections = 100
connection_timeout = 30
pool_recycle = 3600

# Caching
enable_distributed_cache = true
cache_provider = "redis"
redis_cluster = [
    "redis-1.internal:6379",
    "redis-2.internal:6379",
    "redis-3.internal:6379"
]

# Resource limits
max_memory_gb = 16
max_cpu_cores = 8
max_concurrent_requests = 1000

# Batch processing
batch_size = 100
max_batch_wait_ms = 100
batch_timeout_seconds = 30
```

## Configuration Validation and Schema

### Custom Validation Rules

```python
from pydantic import BaseModel, Field, validator
from typing import Optional

class CustomProviderConfig(BaseModel):
    api_key: str = Field(..., min_length=10)
    endpoint: str = Field(..., regex=r'^https?://')
    timeout: int = Field(30, ge=1, le=300)
    batch_size: int = Field(32, ge=1, le=1000)

    @validator('api_key')
    def validate_api_key(cls, v):
        if not v.startswith('cp_'):
            raise ValueError('API key must start with "cp_"')
        return v

    @validator('endpoint')
    def validate_endpoint(cls, v):
        if 'localhost' in v and not cls.is_development():
            raise ValueError('Localhost endpoints not allowed in production')
        return v
```

### Configuration Schema Registration

```python
from codeweaver.factories import codeweaver_factory

# Register configuration schema
codeweaver_factory.register_config_schema(
    "custom_provider",
    CustomProviderConfig
)
```

### Validation Configuration

```toml
[validation]
# Global validation settings
strict_mode = true
validate_on_startup = true
fail_on_validation_error = true

# Custom validation rules
[validation.rules]
api_keys_required = true
secure_urls_only = true
resource_limits_enforced = true

# Environment-specific validation
[validation.environments.production]
require_ssl = true
require_auth = true
forbid_localhost = true
```

## Configuration Management Patterns

### Configuration Builder Pattern

```python
from codeweaver.config import ConfigBuilder

# Build configuration programmatically
config = (ConfigBuilder()
    .with_profile("production")
    .with_provider("voyage_ai", {
        "api_key": "voyage-key",
        "model": "voyage-code-3"
    })
    .with_backend("qdrant", {
        "url": "https://qdrant.example.com",
        "collection": "production-embeddings"
    })
    .with_services({
        "chunking": {"provider": "fastmcp_chunking"},
        "filtering": {"provider": "fastmcp_filtering"}
    })
    .enable_monitoring()
    .enable_caching()
    .build()
)
```

### Configuration Templates

```toml
# template-minimal.toml
[profile]
name = "minimal"

[backend]
provider = "docarray"
backend_type = "memory"

# template-production.toml
[profile]
name = "production"

[backend]
provider = "qdrant"
url = "${QDRANT_URL}"
api_key = "${QDRANT_API_KEY}"
enable_hybrid_search = true

[providers.voyage_ai]
api_key = "${VOYAGE_API_KEY}"
model = "voyage-code-3"

[services]
enable_health_monitoring = true
enable_metrics = true
```

### Configuration Encryption

```python
from codeweaver.config import ConfigEncryption

# Encrypt sensitive configuration
encryption = ConfigEncryption("your-encryption-key")

# Encrypt API keys
encrypted_config = encryption.encrypt_values(
    config_dict,
    fields=["api_key", "password", "secret"]
)

# Save encrypted configuration
with open("config.encrypted.toml", "w") as f:
    toml.dump(encrypted_config, f)
```

## Debugging Configuration

### Configuration Inspection

```python
from codeweaver.config import CodeWeaverConfig

# Load and inspect configuration
config = CodeWeaverConfig()

# Print effective configuration
print("Effective Configuration:")
print(config.model_dump_json(indent=2))

# Check configuration sources
print("Configuration Sources:")
for source, values in config.get_source_info().items():
    print(f"  {source}: {list(values.keys())}")

# Validate configuration
try:
    config.validate()
    print("Configuration is valid")
except Exception as e:
    print(f"Configuration error: {e}")
```

### Debug Mode Configuration

```toml
[debug]
enabled = true
log_config_loading = true
trace_factory_creation = true
validate_schemas = true

# Debug specific components
[debug.components]
config_manager = true
factory_system = true
service_registry = true
plugin_loader = true
```

### Configuration Testing

```python
import pytest
from codeweaver.config import CodeWeaverConfig

def test_minimal_config():
    """Test minimal configuration works"""
    config = CodeWeaverConfig(profile={"name": "minimal"})
    assert config.backend.provider == "docarray"
    assert config.backend.backend_type == "memory"

def test_production_config():
    """Test production configuration validation"""
    config = CodeWeaverConfig(
        profile={"name": "production"},
        backend={
            "url": "https://qdrant.test.com",
            "api_key": "test-key"
        }
    )

    # Should require API key
    with pytest.raises(ValueError):
        CodeWeaverConfig(
            profile={"name": "production"}
            # Missing backend URL
        )
```

## Next Steps

- **Plugin development**: [Extension Development Guide](../extension-development/)
- **Performance optimization**: [Performance Guide](../user-guide/performance.md)
- **Enterprise deployment**: [Deployment Guide](../user-guide/deployment.md)
- **Monitoring and observability**: [Monitoring Guide](../user-guide/monitoring.md)
