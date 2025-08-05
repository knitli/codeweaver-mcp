<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Language Support Reference

**Comprehensive guide to programming language support in CodeWeaver**

CodeWeaver provides intelligent code understanding and chunking for 20+ programming languages through AST-aware parsing and language-specific optimizations.

## Overview

CodeWeaver supports languages through multiple layers:

1. **AST-Aware Chunking**: Using ast-grep for intelligent code segmentation
2. **Fallback Text Processing**: When AST parsing is unavailable
3. **Language-Specific Optimizations**: Custom patterns and rules per language
4. **File Type Detection**: Automatic language identification

## Supported Languages

### Tier 1: Full AST Support + Optimizations

These languages have complete AST-aware chunking with custom patterns for optimal code understanding:

#### **Python** 🐍
```yaml
Extensions: [".py", ".pyw", ".pyi"]
AST Patterns:
  - function_definition
  - class_definition
  - async_function_definition
  - method_definition
Chunking Strategy: "Function and class boundaries"
Quality: ⭐⭐⭐⭐⭐
```

**Features:**
- Intelligent class and method segmentation
- Docstring preservation
- Decorator awareness
- Import statement grouping
- Async/await pattern recognition

**Example Chunk Boundaries:**
```python
# Chunk 1: Imports and module docstring
"""Module for user authentication."""
import hashlib
from typing import Optional

# Chunk 2: Class definition
class UserAuth:
    """Handles user authentication logic."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    # Chunk 3: Method with implementation
    def hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salt = os.urandom(32)
        return hashlib.pbkdf2_hmac('sha256',
                                   password.encode('utf-8'),
                                   salt, 100000)
```

#### **JavaScript/TypeScript** 🌐
```yaml
JavaScript:
  Extensions: [".js", ".jsx", ".mjs", ".cjs"]
TypeScript:
  Extensions: [".ts", ".tsx", ".d.ts"]
AST Patterns:
  - function_declaration
  - arrow_function
  - class_declaration
  - method_definition
  - interface_declaration (TS)
  - type_alias_declaration (TS)
Chunking Strategy: "Component and function boundaries"
Quality: ⭐⭐⭐⭐⭐
```

**Features:**
- React component recognition
- Hook pattern detection
- Type definition preservation
- Module import/export analysis
- JSX structure awareness

**Example Chunk Boundaries:**
```typescript
// Chunk 1: Imports and types
import React, { useState, useEffect } from 'react';
import { User } from './types';

interface UserListProps {
  users: User[];
  onUserSelect: (user: User) => void;
}

// Chunk 2: Component definition
export const UserList: React.FC<UserListProps> = ({
  users,
  onUserSelect
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Chunk 3: Hook implementation
  useEffect(() => {
    if (users.length > 0 && !selectedId) {
      setSelectedId(users[0].id);
    }
  }, [users, selectedId]);

  // Component render logic continues...
};
```

#### **Rust** 🦀
```yaml
Extensions: [".rs"]
AST Patterns:
  - function_item
  - impl_item
  - struct_item
  - enum_item
  - trait_item
  - mod_item
Chunking Strategy: "Module and implementation boundaries"
Quality: ⭐⭐⭐⭐⭐
```

**Features:**
- Struct and enum definitions
- Trait and implementation blocks
- Module boundary respect
- Macro definition handling
- Lifetime annotation preservation

#### **Go** 🐹
```yaml
Extensions: [".go"]
AST Patterns:
  - function_declaration
  - method_declaration
  - type_declaration
  - interface_type
  - struct_type
Chunking Strategy: "Package and function boundaries"
Quality: ⭐⭐⭐⭐⭐
```

**Features:**
- Package-level organization
- Interface and struct definitions
- Method receiver awareness
- Goroutine pattern recognition

#### **Java** ☕
```yaml
Extensions: [".java"]
AST Patterns:
  - class_declaration
  - interface_declaration
  - method_declaration
  - constructor_declaration
  - enum_declaration
Chunking Strategy: "Class and method boundaries"
Quality: ⭐⭐⭐⭐
```

**Features:**
- Class hierarchy awareness
- Annotation preservation
- Package structure respect
- Inner class handling

### Tier 2: AST Support

Languages with AST parsing but limited custom optimizations:

#### **C/C++** ⚡
```yaml
C Extensions: [".c", ".h"]
C++ Extensions: [".cpp", ".hpp", ".cc", ".cxx", ".hxx"]
AST Patterns:
  - function_definition
  - struct_specifier
  - class_specifier
  - namespace_definition
Quality: ⭐⭐⭐⭐
```

#### **C#** 🔷
```yaml
Extensions: [".cs"]
AST Patterns:
  - class_declaration
  - interface_declaration
  - method_declaration
  - namespace_declaration
Quality: ⭐⭐⭐⭐
```

#### **Swift** 🐦
```yaml
Extensions: [".swift"]
AST Patterns:
  - class_declaration
  - struct_declaration
  - function_declaration
  - protocol_declaration
Quality: ⭐⭐⭐⭐
```

#### **Kotlin** 🎯
```yaml
Extensions: [".kt", ".kts"]
AST Patterns:
  - class_declaration
  - function_declaration
  - object_declaration
  - interface_declaration
Quality: ⭐⭐⭐⭐
```

#### **Scala** 🎭
```yaml
Extensions: [".scala", ".sc"]
AST Patterns:
  - class_definition
  - object_definition
  - trait_definition
  - function_definition
Quality: ⭐⭐⭐
```

### Tier 3: Basic AST Support

Languages with basic AST parsing and pattern recognition:

#### **Ruby** 💎
```yaml
Extensions: [".rb", ".rbw"]
AST Patterns:
  - class
  - module
  - method
Quality: ⭐⭐⭐
```

#### **PHP** 🐘
```yaml
Extensions: [".php", ".phtml"]
AST Patterns:
  - class_declaration
  - function_declaration
  - method_declaration
Quality: ⭐⭐⭐
```

#### **Dart** 🎯
```yaml
Extensions: [".dart"]
AST Patterns:
  - class_definition
  - function_signature
  - method_signature
Quality: ⭐⭐⭐
```

### Tier 4: Text-Based Processing

Languages processed with intelligent text chunking and heuristics:

#### **Shell Scripts** 🐚
```yaml
Extensions: [".sh", ".bash", ".zsh", ".fish"]
Processing: "Function and section boundaries"
Quality: ⭐⭐
```

#### **PowerShell** 💻
```yaml
Extensions: [".ps1", ".psm1", ".psd1"]
Processing: "Function and cmdlet boundaries"
Quality: ⭐⭐
```

#### **Lua** 🌙
```yaml
Extensions: [".lua"]
Processing: "Function boundaries"
Quality: ⭐⭐
```

#### **Perl** 🐪
```yaml
Extensions: [".pl", ".pm", ".t"]
Processing: "Subroutine boundaries"
Quality: ⭐⭐
```

## Markup and Data Languages

### Configuration Files

#### **YAML** 📄
```yaml
Extensions: [".yaml", ".yml"]
Processing: "Section and key boundaries"
Features:
  - Multi-document support
  - Nested structure preservation
  - Comment preservation
Quality: ⭐⭐⭐⭐
```

#### **JSON** 📊
```yaml
Extensions: [".json", ".jsonl"]
Processing: "Object and array boundaries"
Features:
  - Schema-aware chunking
  - Large object handling
  - Nested structure preservation
Quality: ⭐⭐⭐⭐
```

#### **TOML** ⚙️
```yaml
Extensions: [".toml"]
Processing: "Section boundaries"
Features:
  - Table structure preservation
  - Comment preservation
Quality: ⭐⭐⭐
```

#### **XML** 🏷️
```yaml
Extensions: [".xml", ".xsd", ".xsl"]
Processing: "Element boundaries"
Features:
  - Tag hierarchy preservation
  - Attribute handling
  - Namespace support
Quality: ⭐⭐⭐
```

### Documentation

#### **Markdown** 📝
```yaml
Extensions: [".md", ".markdown", ".mdown"]
Processing: "Heading and section boundaries"
Features:
  - Heading hierarchy preservation
  - Code block extraction
  - Link and reference handling
  - Table structure preservation
Quality: ⭐⭐⭐⭐⭐
```

#### **reStructuredText** 📖
```yaml
Extensions: [".rst", ".rest"]
Processing: "Section boundaries"
Features:
  - Directive handling
  - Cross-reference preservation
Quality: ⭐⭐⭐
```

#### **LaTeX** 📚
```yaml
Extensions: [".tex", ".latex"]
Processing: "Section and environment boundaries"
Features:
  - Command and environment preservation
  - Math formula handling
Quality: ⭐⭐⭐
```

### Web Technologies

#### **HTML** 🌐
```yaml
Extensions: [".html", ".htm", ".xhtml"]
Processing: "Tag and component boundaries"
Features:
  - Semantic tag recognition
  - Component structure preservation
  - Script and style extraction
Quality: ⭐⭐⭐⭐
```

#### **CSS/SCSS** 🎨
```yaml
CSS Extensions: [".css"]
SCSS Extensions: [".scss", ".sass"]
Processing: "Selector and rule boundaries"
Features:
  - Media query handling
  - Nested rule support (SCSS)
  - Variable and mixin extraction
Quality: ⭐⭐⭐
```

#### **Vue** 🖖
```yaml
Extensions: [".vue"]
Processing: "Component section boundaries"
Features:
  - Template, script, style separation
  - Component prop analysis
  - Composition API support
Quality: ⭐⭐⭐⭐
```

#### **Svelte** ⚡
```yaml
Extensions: [".svelte"]
Processing: "Component boundaries"
Features:
  - Reactive statement recognition
  - Component script analysis
Quality: ⭐⭐⭐
```

## Language-Specific Features

### Python Optimizations

#### Django Framework Support
```yaml
File Patterns:
  - models.py: "Model class boundaries"
  - views.py: "View function/class boundaries"
  - urls.py: "URL pattern boundaries"
  - admin.py: "Admin class boundaries"
  - serializers.py: "Serializer class boundaries"
Features:
  - Model field recognition
  - View decorator preservation
  - URL pattern extraction
```

#### FastAPI Support
```yaml
Features:
  - Route decorator recognition
  - Pydantic model extraction
  - Dependency injection patterns
  - Response model handling
```

### JavaScript/TypeScript Optimizations

#### React Framework Support
```yaml
Features:
  - Component lifecycle methods
  - Hook dependencies
  - PropTypes/interface definitions
  - Context provider patterns
  - Redux action/reducer patterns
```

#### Node.js Support
```yaml
Features:
  - Module export patterns
  - Express route definitions
  - Middleware function recognition
  - Package.json dependency analysis
```

### Rust Optimizations

#### Cargo Project Support
```yaml
Features:
  - Cargo.toml dependency extraction
  - Feature flag recognition
  - Test module identification
  - Benchmark function detection
```

## Configuration and Customization

### Language Detection Configuration

```bash
# Include specific languages only
export CW_INCLUDE_LANGUAGES="python,javascript,typescript,rust"

# Exclude certain languages
export CW_EXCLUDE_LANGUAGES="generated,minified"

# Custom file extensions
export CW_CUSTOM_EXTENSIONS='{"python": [".pyx", ".pxi"], "javascript": [".mjs"]}'
```

### Custom AST Patterns

Add custom patterns for specialized code:

```toml
[chunking.custom_patterns]
python = [
  "function_definition",
  "class_definition",
  "async_function_definition",
  "decorator_definition"
]

javascript = [
  "function_declaration",
  "arrow_function",
  "class_declaration",
  "method_definition"
]
```

### Language-Specific Chunk Sizes

```toml
[chunking.language_settings]
[chunking.language_settings.python]
max_size = 1500
min_size = 100
respect_boundaries = true

[chunking.language_settings.javascript]
max_size = 1200
min_size = 80
preserve_jsx = true

[chunking.language_settings.rust]
max_size = 2000
min_size = 150
include_tests = false
```

## File Type Detection

### Extension-Based Detection

```yaml
Primary Extensions:
  - ".py" :material-arrow-right-circle: Python
  - ".js" :material-arrow-right-circle: JavaScript
  - ".ts" :material-arrow-right-circle: TypeScript
  - ".rs" :material-arrow-right-circle: Rust
  - ".go" :material-arrow-right-circle: Go

Secondary Extensions:
  - ".pyi" :material-arrow-right-circle: Python (type stubs)
  - ".jsx" :material-arrow-right-circle: JavaScript (React)
  - ".tsx" :material-arrow-right-circle: TypeScript (React)
  - ".d.ts" :material-arrow-right-circle: TypeScript (declarations)
```

### Content-Based Detection

When extensions are ambiguous or missing:

```yaml
Shebang Recognition:
  - "#!/usr/bin/env python" :material-arrow-right-circle: Python
  - "#!/bin/bash" :material-arrow-right-circle: Shell
  - "#!/usr/bin/node" :material-arrow-right-circle: JavaScript

Content Patterns:
  - "package.json" :material-arrow-right-circle: Node.js project
  - "Cargo.toml" :material-arrow-right-circle: Rust project
  - "go.mod" :material-arrow-right-circle: Go module
  - "pyproject.toml" :material-arrow-right-circle: Python project
```

### Language Metadata

Each language includes metadata for optimization:

```yaml
Language Metadata:
  syntax_highlighting: "TreeSitter grammar available"
  ast_quality: "Quality rating (1-5 stars)"
  chunk_boundaries: "Preferred chunk boundaries"
  custom_patterns: "Language-specific patterns"
  framework_support: "Supported frameworks"
  community_patterns: "Community-contributed patterns"
```

## Performance Characteristics

### AST Parsing Performance

| Language | Parse Speed | Memory Usage | Accuracy |
|----------|-------------|--------------|----------|
| **Python** | Fast | Low | 95% |
| **JavaScript** | Fast | Low | 95% |
| **TypeScript** | Medium | Medium | 90% |
| **Rust** | Medium | Medium | 90% |
| **Go** | Fast | Low | 85% |
| **Java** | Medium | Medium | 85% |
| **C++** | Slow | High | 80% |

### Fallback Processing

When AST parsing fails:

1. **Heuristic Patterns**: Language-specific regex patterns
2. **Indentation Analysis**: Python-style indentation detection
3. **Bracket Matching**: Brace and parenthesis balancing
4. **Comment Preservation**: Keep documentation blocks
5. **String Literal Handling**: Preserve code examples in strings

## Language Support Roadmap

### Planned Additions

#### **Emerging Languages** (Next Release)
- **Zig**: Systems programming language
- **Nim**: Efficient, expressive, elegant
- **V**: Simple, fast, safe compiled language

#### **Specialized Languages** (Future)
- **R**: Statistical computing
- **Julia**: Scientific computing
- **Matlab**: Numerical computing
- **Fortran**: Scientific computing legacy

#### **Domain-Specific Languages**
- **GraphQL**: Schema and query language
- **Terraform**: Infrastructure as code
- **Kubernetes YAML**: Container orchestration
- **Docker**: Container definitions

### Community Contributions

We welcome community contributions for:

1. **New Language Support**: AST patterns and optimizations
2. **Framework Patterns**: Framework-specific chunking rules
3. **Custom Parsers**: Domain-specific language processors
4. **Quality Improvements**: Better boundary detection

#### Contributing Language Support

```python
# Example: Adding new language support
from codeweaver.languages import register_language

register_language(
    name="your_language",
    extensions=[".ext"],
    ast_patterns=["function_def", "class_def"],
    chunk_strategy="function_boundaries",
    custom_rules={
        "preserve_comments": True,
        "respect_indentation": True
    }
)
```

## Best Practices

### Language-Specific Tips

#### **Python Projects**
- Enable docstring preservation
- Include type annotations in chunks
- Respect package structure
- Handle async/await patterns

#### **JavaScript/TypeScript Projects**
- Preserve JSX component structure
- Include TypeScript definitions
- Handle module imports/exports
- Recognize React patterns

#### **Rust Projects**
- Respect module boundaries
- Include trait implementations
- Handle macro definitions
- Preserve cargo metadata

#### **Multi-Language Projects**
- Configure per-language chunk sizes
- Use language-specific patterns
- Handle build system files
- Preserve cross-language interfaces

### Performance Optimization

1. **Language Filtering**: Only index relevant languages
2. **File Size Limits**: Set appropriate limits per language
3. **AST Caching**: Cache parsed AST structures
4. **Parallel Processing**: Enable for large codebases

## Next Steps

- **[Provider Comparison :material-arrow-right-circle:](./provider-comparison.md)**: Choose optimal providers for your languages
- **[Configuration Reference :material-arrow-right-circle:](../getting-started/configuration.md)**: Configure language-specific settings
- **[Extension Development :material-arrow-right-circle:](../extension-development/)**: Add support for new languages
- **[Performance Optimization :material-arrow-right-circle:](../user-guide/performance.md)**: Optimize for your codebase
