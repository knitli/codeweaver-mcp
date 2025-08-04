<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Material for MkDocs Theme Configuration

This document details the Material theme configuration recommendations for CodeWeaver's documentation, providing a comprehensive guide for maintaining and extending the theme.

## Theme Configuration Overview

The Material theme configuration is designed to provide:

- **Modern aesthetic** with CodeWeaver branding
- **Optimal performance** with lazy loading and caching
- **Accessibility compliance** with WCAG 2.1 AA standards
- **Responsive design** for all device sizes
- **Developer experience** optimizations

## Core Theme Features

### Navigation Features

```yaml
features:
  # Instant loading for SPA-like experience
  - navigation.instant      # Instant page loading
  - navigation.instant.prefetch  # Prefetch links on hover
  
  # Enhanced navigation
  - navigation.tracking     # URL fragment tracking
  - navigation.tabs         # Top-level tabs
  - navigation.tabs.sticky  # Sticky navigation
  - navigation.sections     # Section grouping
  - navigation.expand       # Auto-expand nav
  - navigation.path         # Breadcrumb navigation
  - navigation.indexes      # Section landing pages
  - navigation.top          # Back to top button
```

### Content Features

```yaml
features:
  # Code enhancements
  - content.code.copy      # Copy code buttons
  - content.code.select    # Select code sections
  - content.code.annotate  # Inline annotations
  
  # Content organization
  - content.tabs.link      # Linked content tabs
  - content.tooltips       # Rich tooltips
  - content.footnote.tooltips  # Footnote previews
  
  # Table of contents
  - toc.follow            # Follow scroll position
  - toc.integrate         # Integrate with nav
```

### Search Features

```yaml
features:
  # Enhanced search
  - search.highlight      # Highlight search terms
  - search.share         # Share search URLs
  - search.suggest       # Search suggestions
```

## Color Palette Strategy

### Dual Theme Support

```yaml
palette:
  # Light mode (default)
  - media: "(prefers-color-scheme: light)"
    scheme: default
    primary: blue          # CodeWeaver brand blue
    accent: amber          # Complementary accent
    toggle:
      icon: material/brightness-7
      name: Switch to dark mode
  
  # Dark mode
  - media: "(prefers-color-scheme: dark)"
    scheme: slate
    primary: blue          # Consistent branding
    accent: amber          # Maintains contrast
    toggle:
      icon: material/brightness-4
      name: Switch to light mode
```

### Brand Color System

```css
:root {
  /* Primary brand colors */
  --cw-primary: #1976d2;      /* CodeWeaver blue */
  --cw-primary-light: #42a5f5;
  --cw-primary-dark: #1565c0;
  
  /* Accent colors */
  --cw-accent: #ffc107;       /* Amber accent */
  --cw-accent-light: #ffecb3;
  --cw-accent-dark: #ff8f00;
  
  /* Semantic colors */
  --cw-success: #4caf50;      /* Green for success */
  --cw-warning: #ff9800;      /* Orange for warnings */
  --cw-error: #f44336;        /* Red for errors */
  --cw-info: #2196f3;         /* Blue for information */
}
```

## Typography System

### Font Stack

```yaml
font:
  text: Inter              # Modern, readable sans-serif
  code: JetBrains Mono     # Developer-optimized monospace
```

**Inter Font Benefits**:
- Excellent readability at all sizes
- Wide language support
- Optimized for digital interfaces
- Professional appearance

**JetBrains Mono Benefits**:
- Developer-designed for code
- Enhanced ligature support
- Clear character distinction
- Reduced eye strain

### Typography Scale

```css
/* Responsive typography scale */
h1 { font-size: clamp(2rem, 4vw, 3.5rem); }
h2 { font-size: clamp(1.5rem, 3vw, 2.5rem); }
h3 { font-size: clamp(1.25rem, 2.5vw, 2rem); }
h4 { font-size: clamp(1.125rem, 2vw, 1.5rem); }

/* Code typography */
code { 
  font-size: 0.9em;
  font-family: var(--md-code-font);
}

pre code {
  font-size: 0.85em;
  line-height: 1.5;
}
```

## Plugin Configuration Strategy

### Essential Plugins

#### 1. MkDocstrings (API Documentation)
```yaml
- mkdocstrings:
    handlers:
      python:
        options:
          # Source code display
          show_source: true
          show_root_heading: true
          
          # Member organization
          group_by_category: true
          show_bases: true
          
          # Documentation style
          docstring_style: google
          separate_signature: true
          
          # Type annotations
          show_signature_annotations: true
```

#### 2. Search Enhancement
```yaml
- search:
    separator: '[\s\u200b\-_,:!=\[\]()"`/]+|\.(?!\d)|&[lg]t;|(?!\b)(?=[A-Z][a-z])'
    # Improved tokenization for code terms
```

#### 3. Performance Optimization
```yaml
- minify:
    minify_html: true
    minify_js: true
    minify_css: true
    cache_safe: true
```

#### 4. Social Integration
```yaml
- social:
    cards_layout_options:
      background_color: "#1976d2"  # Brand color
      font_family: Inter
```

### Content Organization Plugins

#### 1. Git Integration
```yaml
- git-revision-date-localized:
    type: timeago
    custom_format: "%d. %B %Y"
    timezone: UTC
    locale: en
```

#### 2. Tags System
```yaml
- tags:
    tags_file: reference/tags.md
```

#### 3. RSS Feeds
```yaml
- rss:
    match_path: "reference/changelog.*"
    date_from_meta:
      as_creation: date
    categories:
      - tags
      - categories
```

## Markdown Extensions Configuration

### Code Enhancement Extensions

```yaml
markdown_extensions:
  # Syntax highlighting
  - pymdownx.highlight:
      anchor_linenums: true
      line_spans: __span
      pygments_lang_class: true
      use_pygments: true
      auto_title: true
      linenums: true
  
  # Inline code highlighting
  - pymdownx.inlinehilite
  
  # Code block features
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
```

### Content Organization Extensions

```yaml
markdown_extensions:
  # Content tabs
  - pymdownx.tabbed:
      alternate_style: true
      combine_header_slug: true
  
  # Admonitions
  - admonition
  - pymdownx.details
  
  # Lists and definitions
  - def_list
  - pymdownx.tasklist:
      custom_checkbox: true
```

### Interactive Features

```yaml
markdown_extensions:
  # Emoji support
  - pymdownx.emoji:
      emoji_index: !!python/name:material.extensions.emoji.twemoji
      emoji_generator: !!python/name:material.extensions.emoji.to_svg
  
  # Math notation
  - pymdownx.arithmatex:
      generic: true
  
  # Enhanced formatting
  - pymdownx.mark        # ==highlighting==
  - pymdownx.caret       # ^^insertions^^
  - pymdownx.tilde       # ~~deletions~~
  - pymdownx.keys        # ++ctrl+alt+del++
```

## Custom CSS Architecture

### CSS Organization

```plaintext
docs/assets/
├── extra.css           # Main custom styles
├── api-docs.css        # API documentation specific
├── components/         # Component-specific styles
│   ├── cards.css
│   ├── navigation.css
│   └── tables.css
└── utilities/          # Utility classes
    ├── spacing.css
    └── typography.css
```

### CSS Custom Properties Strategy

```css
/* Theme-aware custom properties */
:root {
  /* Spacing scale */
  --cw-spacing-xs: 0.25rem;
  --cw-spacing-sm: 0.5rem;
  --cw-spacing-md: 1rem;
  --cw-spacing-lg: 1.5rem;
  --cw-spacing-xl: 2rem;
  
  /* Shadow system */
  --cw-shadow-sm: 0 1px 3px rgba(0,0,0,0.12);
  --cw-shadow-md: 0 3px 6px rgba(0,0,0,0.16);
  --cw-shadow-lg: 0 10px 20px rgba(0,0,0,0.19);
  
  /* Border radius scale */
  --cw-radius-sm: 4px;
  --cw-radius-md: 8px;
  --cw-radius-lg: 12px;
}
```

## Performance Optimization

### Asset Optimization

```yaml
# CSS/JS minification
- minify:
    minify_html: true
    minify_js: true
    minify_css: true
    htmlmin_opts:
      remove_comments: true
    cache_safe: true
```

### Image Optimization

```yaml
# Social card generation
- social:
    enabled: !ENV [CI, false]  # Only in CI/production
    cards_layout_options:
      background_color: "#1976d2"
      font_family: Inter
```

### Loading Performance

```css
/* CSS containment for performance */
.md-content {
  contain: layout style;
}

/* Optimize font loading */
@font-face {
  font-family: 'Inter';
  font-display: swap;  /* Improve loading performance */
}
```

## Accessibility Features

### WCAG Compliance

```yaml
features:
  # Keyboard navigation
  - navigation.tracking
  - navigation.top
  
  # Screen reader support
  - content.tooltips
  - content.footnote.tooltips
```

### High Contrast Support

```css
@media (prefers-contrast: high) {
  .md-typeset .card {
    border-width: 2px;
    border-color: var(--md-default-fg-color);
  }
  
  .md-typeset .admonition {
    border-left-width: 6px;
  }
}
```

### Reduced Motion Support

```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

## Mobile Optimization

### Responsive Design

```css
/* Mobile-first responsive design */
@media screen and (max-width: 768px) {
  .grid.cards {
    grid-template-columns: 1fr;
  }
  
  .md-typeset .card {
    padding: var(--cw-spacing-md);
  }
  
  /* Optimize touch targets */
  .md-nav__link {
    min-height: 44px;
    display: flex;
    align-items: center;
  }
}
```

### Touch Optimization

```css
/* Improve touch interactions */
.md-button {
  min-height: 44px;  /* Minimum touch target size */
  padding: 12px 24px;
}

.md-nav__link {
  padding: 12px 16px;
  min-height: 44px;
}
```

## Customization Guidelines

### Adding New Components

1. **Create component CSS file** in `docs/assets/components/`
2. **Import in main CSS** using `@import`
3. **Use CSS custom properties** for theming
4. **Follow BEM methodology** for class naming
5. **Test across themes** (light/dark mode)

### Brand Customization

```css
/* Override brand colors */
:root {
  --md-primary-fg-color: #your-brand-color;
  --md-accent-fg-color: #your-accent-color;
}

/* Custom logo styling */
.md-header__button.md-logo {
  /* Logo customizations */
}
```

### Plugin Development

When adding new plugins:

1. **Check compatibility** with Material theme
2. **Configure in mkdocs.yml** with proper options
3. **Add custom CSS** if needed for styling
4. **Test performance impact**
5. **Document configuration** for maintainers

This configuration provides a solid foundation for CodeWeaver's documentation while maintaining flexibility for future enhancements and customizations.