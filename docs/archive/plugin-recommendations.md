<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# MkDocs Plugin Recommendations

This document provides comprehensive recommendations for MkDocs plugins that enhance CodeWeaver's documentation experience, organized by functionality and use case.

## Essential Plugins (Required)

### 1. mkdocstrings - API Documentation
**Purpose**: Automatic API documentation generation from docstrings

```yaml
- mkdocstrings:
    handlers:
      python:
        options:
          # Documentation rendering
          show_source: true
          show_root_heading: true
          show_root_toc_entry: true
          show_object_full_path: false
          
          # Member handling
          show_bases: true
          show_inheritance_diagram: true
          show_if_no_docstring: true
          group_by_category: true
          
          # Docstring processing
          docstring_style: google
          docstring_options:
            ignore_init_summary: true
            trim_doctest_flags: true
          
          # Type annotations
          show_signature_annotations: true
          separate_signature: true
          
          # Filters
          filters:
            - "!^_[^_]"  # Hide private members except __special__
```

**Benefits**:
- ✅ Automatic API documentation from source code
- ✅ Google-style docstring support
- ✅ Type annotation display
- ✅ Cross-references with source code
- ✅ Inheritance diagrams

**CodeWeaver Use Cases**:
- Factory system documentation
- Protocol interface documentation
- Service provider APIs
- Plugin development guides

### 2. search - Enhanced Search
**Purpose**: Improved search functionality with better tokenization

```yaml
- search:
    separator: '[\s\u200b\-_,:!=\[\]()"`/]+|\.(?!\d)|&[lg]t;|(?!\b)(?=[A-Z][a-z])'
    lang:
      - en
    min_search_length: 2
    prebuild_index: true
```

**Benefits**:
- ✅ Better tokenization for code terms
- ✅ Camel case splitting
- ✅ Special character handling
- ✅ Pre-built search index for performance

### 3. minify - Performance Optimization
**Purpose**: Compress HTML, CSS, and JavaScript for faster loading

```yaml
- minify:
    minify_html: true
    minify_js: true
    minify_css: true
    htmlmin_opts:
      remove_comments: true
      remove_empty_space: true
      remove_redundant_attributes: true
    cache_safe: true
```

**Benefits**:
- ✅ Reduced page load times
- ✅ Bandwidth savings
- ✅ Better Core Web Vitals scores
- ✅ Improved SEO performance

## Content Enhancement Plugins

### 4. git-revision-date-localized - Version Tracking
**Purpose**: Add last modified dates and git information

```yaml
- git-revision-date-localized:
    enabled: !ENV [CI, false]
    type: timeago
    custom_format: "%d. %B %Y"
    timezone: UTC
    locale: en
    strict: false
    fallback_to_build_date: true
```

**Benefits**:
- ✅ Automatic last modified dates
- ✅ Git integration for versioning
- ✅ Timezone-aware timestamps
- ✅ Fallback to build date

### 5. tags - Content Organization
**Purpose**: Tag-based content organization and discovery

```yaml
- tags:
    tags_file: reference/tags.md
    tags_extra_files:
      compatibility.md:
        - compatibility
      performance.md:
        - performance
        - optimization
```

**Benefits**:
- ✅ Topic-based content discovery
- ✅ Tag-based navigation
- ✅ Automated tag pages
- ✅ Cross-reference system

**CodeWeaver Tag Strategy**:
- `beginner`, `intermediate`, `advanced` - Skill levels
- `configuration`, `api`, `plugin`, `tutorial` - Content types
- `python`, `typescript`, `docker` - Technologies
- `performance`, `security`, `troubleshooting` - Topics

### 6. awesome-pages - Navigation Management
**Purpose**: Simplified navigation structure management

```yaml
- awesome-pages:
    filename: .pages
    collapse_single_pages: true
    strict: false
```

**Benefits**:
- ✅ Simplified navigation configuration
- ✅ Automatic page ordering
- ✅ Flexible navigation structure
- ✅ Support for navigation metadata

## Social and SEO Plugins

### 7. social - Social Media Cards
**Purpose**: Automatic social media card generation

```yaml
- social:
    enabled: !ENV [CI, false]
    cards_dir: assets/images/social
    cards_layout_options:
      background_color: "#1976d2"
      background_image: assets/logo-background.png
      color: "#ffffff"
      font_family: Inter
    debug: false
```

**Benefits**:
- ✅ Rich social media previews
- ✅ Branded social cards
- ✅ Automatic card generation
- ✅ SEO improvement

### 8. rss - RSS Feed Generation
**Purpose**: RSS feeds for changelog and updates

```yaml
- rss:
    enabled: !ENV [CI, false]
    match_path: "reference/changelog.*"
    date_from_meta:
      as_creation: date
      as_update: git
      datetime_format: "%Y-%m-%d %H:%M"
    categories:
      - tags
      - categories
    image: https://docs.codeweaver.dev/assets/logo.png
```

**Benefits**:
- ✅ Automatic RSS feed generation
- ✅ Changelog distribution
- ✅ Update notifications
- ✅ SEO benefits

## Development and Quality Plugins

### 9. htmlproofer - Link Validation
**Purpose**: Validate internal and external links

```yaml
- htmlproofer:
    enabled: !ENV [CI, false]
    raise_error: true
    validate_external_urls: true
    validate_offline: false
    ignore_urls:
      - "https://example.com"  # Known problematic URLs
    ignore_files:
      - "404.html"
```

**Benefits**:
- ✅ Broken link detection
- ✅ External URL validation
- ✅ CI/CD integration
- ✅ Quality assurance

### 10. mike - Documentation Versioning
**Purpose**: Multi-version documentation management

```yaml
- mike:
    version_selector: true
    css_dir: css
    javascript_dir: js
    canonical_version: latest
```

**Benefits**:
- ✅ Version-specific documentation
- ✅ Version selector widget
- ✅ Canonical URL management
- ✅ Backward compatibility

## Advanced Feature Plugins

### 11. macros - Template System
**Purpose**: Jinja2 template macros and variables

```yaml
- macros:
    module_name: docs/macros
    include_dir: docs/includes
    variables:
      version: !ENV [CODEWEAVER_VERSION, "latest"]
      api_base: "https://api.codeweaver.dev"
```

**Benefits**:
- ✅ Dynamic content generation
- ✅ Reusable content blocks
- ✅ Environment variable injection
- ✅ Template-based documentation

**Example Macros**:
```python
# docs/macros.py
def define_env(env):
    @env.macro
    def code_example(language, file_path):
        return f"""
=== "{language.title()}"
    ```{language}
    --8<-- "{file_path}"
    ```plaintext
"""
```

### 12. include-markdown - Content Inclusion
**Purpose**: Include external markdown files and content

```yaml
- include-markdown:
    opening_tag: "{!"
    closing_tag: "!}"
    encoding: utf-8
    preserve_includer_indent: true
```

**Benefits**:
- ✅ Reusable content blocks
- ✅ Single-source documentation
- ✅ External file inclusion
- ✅ Consistent snippets

### 13. mermaid2 - Advanced Diagrams
**Purpose**: Enhanced Mermaid diagram support

```yaml
- mermaid2:
    arguments:
      theme: |
        %%{init: {'theme':'base', 'themeVariables': {
          'primaryColor': '#1976d2',
          'primaryTextColor': '#ffffff',
          'primaryBorderColor': '#1565c0',
          'lineColor': '#757575'
        }}}%%
```

**Benefits**:
- ✅ Advanced diagram theming
- ✅ CodeWeaver brand colors
- ✅ Interactive diagrams
- ✅ SVG output optimization

## Analytics and Monitoring Plugins

### 14. google-analytics - Usage Analytics
**Purpose**: Detailed usage analytics and insights

```yaml
- google-analytics:
    gtag: !ENV [GOOGLE_ANALYTICS_ID, ""]
    anonymize_ip: true
```

**Benefits**:
- ✅ Usage pattern analysis
- ✅ Popular content identification
- ✅ User behavior insights
- ✅ Performance metrics

### 15. privacy - GDPR Compliance
**Purpose**: Privacy controls and consent management

```yaml
- privacy:
    enabled: !ENV [CI, false]
    cache: false
    assets_exclude:
      - "*.woff2"
    external_assets_exclude:
      - "fonts.googleapis.com"
```

**Benefits**:
- ✅ GDPR compliance
- ✅ Asset optimization
- ✅ Privacy controls
- ✅ Consent management

## Plugin Configuration Best Practices

### Environment-Based Configuration
```yaml
# Enable expensive plugins only in CI/production
- plugin_name:
    enabled: !ENV [CI, false]
    
# Use environment variables for configuration
- plugin_name:
    api_key: !ENV [PLUGIN_API_KEY, ""]
```

### Performance Considerations
```yaml
# Order plugins by impact
plugins:
  - search           # Core functionality first
  - minify          # Performance optimization
  - social          # Resource-intensive last
```

### Development vs Production
```yaml
# mkdocs.yml (base configuration)
plugins:
  - search
  - mkdocstrings
  - minify

# mkdocs-prod.yml (production overrides)
plugins:
  - search
  - mkdocstrings
  - minify
  - social:
      enabled: true
  - google-analytics:
      gtag: "G-XXXXXXXXXX"
```

## Custom Plugin Development

### Plugin Template
```python
# plugins/codeweaver_examples.py
from mkdocs.config import config_options
from mkdocs.plugins import BasePlugin

class CodeWeaverExamplesPlugin(BasePlugin):
    config_scheme = (
        ('example_dir', config_options.Type(str, default='examples')),
        ('language', config_options.Type(str, default='python')),
    )
    
    def on_page_markdown(self, markdown, page, config, files):
        # Process markdown to add code examples
        return self.process_examples(markdown)
```

### Integration Points
- **on_config**: Modify configuration
- **on_files**: Process file collection
- **on_page_markdown**: Transform markdown
- **on_page_content**: Modify HTML content
- **on_post_build**: Post-processing

## Plugin Performance Optimization

### Caching Strategies
```yaml
- expensive_plugin:
    cache_dir: .cache/plugin_name
    cache_timeout: 3600  # 1 hour
```

### Conditional Loading
```yaml
- development_plugin:
    enabled: !ENV [MKDOCS_DEV, false]
    
- production_plugin:
    enabled: !ENV [MKDOCS_PROD, false]
```

### Resource Management
```yaml
- resource_intensive_plugin:
    max_workers: 4
    timeout: 30
    retry_count: 3
```

## Maintenance and Updates

### Plugin Update Strategy
1. **Monitor plugin versions** in requirements file
2. **Test compatibility** with Material theme updates
3. **Review breaking changes** in plugin changelogs
4. **Maintain fallback configurations** for critical plugins
5. **Document custom modifications** for future reference

### Health Monitoring
```yaml
# CI/CD pipeline checks
- name: Validate documentation build
  run: |
    mkdocs build --strict
    mkdocs serve --dev-addr 0.0.0.0:8000 &
    sleep 10
    curl -f http://localhost:8000/ || exit 1
```

This plugin configuration provides a comprehensive foundation for CodeWeaver's documentation while maintaining flexibility for future enhancements and customizations.