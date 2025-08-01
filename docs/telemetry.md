<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Telemetry Configuration

CodeWeaver includes comprehensive usage tracking through PostHog with privacy-first design and multiple opt-out mechanisms.

## Privacy-First Design

- **Anonymous by default**: Uses hashed, session-specific user IDs
- **Data sanitization**: File paths, repository names, and queries are sanitized
- **Local-first**: Events are batched locally before transmission
- **No sensitive data**: Personal or proprietary information is never transmitted

## Opt-Out Mechanisms

### Environment Variables
```bash
# Disable telemetry completely
export CW_TELEMETRY_ENABLED=false

# Alternative opt-out method
export CW_NO_TELEMETRY=true
```

### Configuration File
```toml
[telemetry]
enabled = false
```

### Runtime API
```python
# Get telemetry service and disable it
telemetry_service = services_manager.get_telemetry_service()
if telemetry_service:
    telemetry_service.set_enabled(enabled=False)
```

## Configuration Options

All telemetry settings can be configured through environment variables (with `CW_` prefix):

```toml
[telemetry]
# Basic settings
enabled = true                    # CW_TELEMETRY_ENABLED
anonymous_tracking = true         # CW_TELEMETRY_ANONYMOUS_TRACKING
provider = "posthog_telemetry"    # CW_TELEMETRY_PROVIDER

# PostHog configuration
api_key = "your_posthog_key"      # CW_TELEMETRY_API_KEY
host = "https://app.posthog.com"  # CW_TELEMETRY_HOST

# Privacy settings
hash_file_paths = true            # CW_TELEMETRY_HASH_FILE_PATHS
hash_repository_names = true      # CW_TELEMETRY_HASH_REPOSITORY_NAMES
sanitize_queries = true           # CW_TELEMETRY_SANITIZE_QUERIES
collect_sensitive_data = false    # CW_TELEMETRY_COLLECT_SENSITIVE_DATA

# Event filtering
track_indexing = true             # CW_TELEMETRY_TRACK_INDEXING
track_search = true               # CW_TELEMETRY_TRACK_SEARCH
track_errors = true               # CW_TELEMETRY_TRACK_ERRORS
track_performance = true          # CW_TELEMETRY_TRACK_PERFORMANCE

# Performance settings
batch_size = 50                   # CW_TELEMETRY_BATCH_SIZE
flush_interval = 30.0             # CW_TELEMETRY_FLUSH_INTERVAL
max_queue_size = 1000             # CW_TELEMETRY_MAX_QUEUE_SIZE
```

## What Data is Collected

### Indexing Operations
- Repository hash (not actual name)
- File count and language distribution
- Processing time and success/failure status

### Search Operations
- Query type (semantic vs AST grep)
- Result count and search latency
- Query complexity assessment (simple/medium/complex)
- Filters used (anonymized)

### Performance Metrics
- Operation duration and resource usage
- Service health and availability
- Error rates and categories

### Privacy Measures
- File paths are hashed: `/secret/project/file.py` → `path_3_.py_a1b2c3d4`
- Repository names are hashed: `company/secret-repo` → `repo_e5f6g7h8`
- Search queries are sanitized to remove content while preserving patterns

## Getting Privacy Information

```python
# Check what data is being collected
privacy_info = telemetry_service.get_privacy_info()
print(privacy_info)
```

This returns detailed information about:
- Current telemetry status
- Data collection settings
- Privacy measures in effect
- Available opt-out methods

## Testing Mode

For development and testing, enable mock mode:

```toml
[telemetry]
mock_mode = true  # Events are simulated, not sent to PostHog
```

## Examples

### Basic Usage with Telemetry Enabled
```python
from codeweaver.types import ServicesConfig, TelemetryServiceConfig

# Configure telemetry
telemetry_config = TelemetryServiceConfig(
    enabled=True,
    api_key="your_posthog_key",
    anonymous_tracking=True,
    track_indexing=True,
    track_search=True,
)

services_config = ServicesConfig()
services_config.telemetry = telemetry_config

# Use with services manager
services_manager = ServicesManager(services_config)
await services_manager.initialize()
```

### Privacy-Focused Configuration
```python
# Maximum privacy settings
telemetry_config = TelemetryServiceConfig(
    enabled=True,
    anonymous_tracking=True,
    hash_file_paths=True,
    hash_repository_names=True,
    sanitize_queries=True,
    collect_sensitive_data=False,
)
```

### Complete Opt-Out
```python
# Disable telemetry entirely
telemetry_config = TelemetryServiceConfig(enabled=False)

# Or via environment variable
import os
os.environ["CW_TELEMETRY_ENABLED"] = "false"
```
