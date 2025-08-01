# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Service-specific enums for CodeWeaver's services layer.

This module was created to resolve a circular dependency issue where services/data.py
was importing from the main cw_types package, which in turn imported from services modules.

By extracting enum classes to this separate module and having services/data.py import
directly from base modules (base_enum, config) instead of the main cw_types package,
the circular dependency is broken while preserving all typing improvements.
"""

from codeweaver.cw_types.base_enum import BaseEnum


class HealthStatus(BaseEnum):
    """Health status of a service or system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ValidationSeverity(BaseEnum):
    """Severity levels for validation errors and warnings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MemoryUsage(BaseEnum):
    """Memory usage categories for service providers."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PerformanceProfile(BaseEnum):
    """Performance profiles for service providers."""

    STANDARD = "standard"
    HIGH_PERFORMANCE = "high_performance"
    LOW_LATENCY = "low_latency"
    RESOURCE_EFFICIENT = "resource_efficient"


class ProviderStatus(BaseEnum):
    """Status of a service provider."""

    REGISTERED = "registered"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DEPRECATED = "deprecated"
