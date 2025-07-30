# sourcery skip: avoid-single-character-names-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Cross-module enums for CodeWeaver.

This module centralizes enums that are used across multiple modules to avoid
circular dependencies and provide a single source of truth for shared enumeration
values. These enums support the factory pattern, error handling, component lifecycle,
and search system functionality.
"""

from codeweaver.types.base_enum import BaseEnum


class ComponentState(BaseEnum):
    """
    Component lifecycle states used throughout the factory system.

    This enum tracks the various states a component can be in during its lifecycle,
    from initial creation through destruction. Used by the initialization system,
    factory management, and component monitoring.

    States flow generally as: UNINITIALIZED → INITIALIZING → INITIALIZED →
    STARTING → RUNNING → STOPPING → STOPPED, with ERROR and DESTROYED as
    terminal states that can be reached from any other state.
    """

    UNINITIALIZED = "uninitialized"  # Component has been created but not yet initialized
    INITIALIZING = "initializing"  # Component is currently being initialized
    INITIALIZED = "initialized"  # Component has been successfully initialized
    STARTING = "starting"  # Component is in the process of starting
    RUNNING = "running"  # Component is actively running and operational
    STOPPING = "stopping"  # Component is in the process of stopping
    STOPPED = "stopped"  # Component has been stopped but can be restarted
    ERROR = "error"  # Component encountered an error and is non-functional
    DESTROYED = "destroyed"  # Component has been permanently destroyed


class SearchComplexity(BaseEnum):
    """
    Search complexity levels for task delegation decisions.

    Used by the task search system to categorize the complexity of search operations
    and determine whether they should be delegated to the Task tool for more
    comprehensive handling. Complexity assessment considers query scope, pattern
    matching requirements, and estimated resource usage.

    Progression from SIMPLE (direct execution) to UNCERTAIN (Task tool delegation
    strongly recommended) helps optimize search performance and accuracy.
    """

    SIMPLE = "simple"  # Straightforward search that can be handled directly
    MODERATE = "moderate"  # Moderately complex search that may benefit from delegation
    COMPLEX = "complex"  # Complex search requiring comprehensive analysis
    UNCERTAIN = "uncertain"  # Uncertain scope requiring Task tool delegation


class ErrorSeverity(BaseEnum):
    """
    Error severity levels for the error handling system.

    Used throughout the codebase to classify errors by their impact and urgency.
    Maps to standard logging levels and determines appropriate response actions,
    recovery strategies, and notification requirements.

    Levels range from TRACE (debug information) to FATAL (system shutdown required),
    with each level triggering different handling behaviors in the error management
    system.
    """

    TRACE = "trace"  # Debugging information for development
    DEBUG = "debug"  # Debug-level issues for troubleshooting
    INFO = "info"  # Informational messages about normal operation
    WARNING = "warning"  # Warnings that don't prevent operation but indicate issues
    ERROR = "error"  # Errors that prevent specific operations from completing
    CRITICAL = "critical"  # Critical errors requiring immediate attention
    FATAL = "fatal"  # Fatal errors causing system shutdown


class ErrorCategory(BaseEnum):
    """
    Error categories for classification and handling.

    Used by the error handling system to categorize errors by their source and nature,
    enabling targeted recovery strategies and appropriate handling logic. Each category
    has associated recovery suggestions and handling patterns.

    Categories cover the major error domains in the system: configuration issues,
    validation failures, component problems, plugin issues, network connectivity,
    resource availability, security violations, system-level errors, and user input.
    """

    CONFIGURATION = "configuration"  # Configuration-related errors (files, env vars, etc.)
    VALIDATION = "validation"  # Validation failures (input, schema, constraints)
    COMPONENT = "component"  # Component-specific errors (initialization, operation)
    PLUGIN = "plugin"  # Plugin-related errors (loading, compatibility, config)
    NETWORK = "network"  # Network connectivity errors (timeouts, unreachable)
    RESOURCE = "resource"  # Resource availability errors (memory, disk, quotas)
    SECURITY = "security"  # Security violations (auth, permissions, threats)
    SYSTEM = "system"  # System-level errors (OS, infrastructure, dependencies)
    USER = "user"  # User input errors (invalid data, malformed requests)


class ChunkingStrategy(BaseEnum):
    """
    Strategies for chunking content during processing.

    Used by the chunking system to determine how content should be split into
    manageable pieces for processing and embedding. Different strategies offer
    trade-offs between accuracy, performance, and structure preservation.
    """

    AUTO = "auto"  # Automatically detect best strategy based on content type
    AST = "ast"  # Use AST-based chunking for structured code analysis
    SIMPLE = "simple"  # Simple text-based chunking with fixed boundaries
    SEMANTIC = "semantic"  # Semantic-aware chunking preserving meaning boundaries


class PerformanceMode(BaseEnum):
    """
    Performance optimization modes for service operations.

    Used throughout the system to adjust processing strategies based on
    performance requirements and resource constraints. Modes range from
    minimal resource usage to maximum throughput optimization.
    """

    MINIMAL = "minimal"  # Minimal resource usage, slower processing
    BALANCED = "balanced"  # Balanced performance and resource usage
    OPTIMIZED = "optimized"  # Optimized for performance, higher resource usage
    MAXIMUM = "maximum"  # Maximum performance, highest resource usage


class Language(BaseEnum):
    """
    Programming languages supported by the system.

    Used by parsers, analyzers, and processors to handle language-specific
    features, syntax, and processing requirements. Each language has associated
    processing capabilities and configuration options.
    """

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    C_LANG = "c"
    RUST = "rust"
    GO = "go"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    TOML = "toml"
    MARKDOWN = "markdown"
    TEXT = "text"
    UNKNOWN = "unknown"

    @classmethod
    def ast_grep_languages(cls) -> tuple["Language"]:
        """Languages supported by Ast-Grep's builtin parsers."""
        return tuple(
            sorted(
                (lang
                for lang in cls.__members__.values()
                if lang not in (cls.UNKNOWN, cls.TEXT, cls.MARKDOWN, cls.XML)),
                key=lambda x: x.value
            )
        )

    @property
    def supports_ast_grep(self) -> bool:
        """Check if this language is supported by Ast-Grep."""
        return self in self.ast_grep_languages()
