# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Comprehensive error handling patterns for CodeWeaver factories.

Implements the error classification, handling, recovery, and graceful
degradation patterns as specified in the architecture documents.
"""

import logging
import time
import traceback
import uuid

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass

from codeweaver.cw_types import (
    BaseComponentConfig,
    CodeWeaverFactoryError,
    ComponentType,
    ErrorCategory,
    ErrorSeverity,
)


logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Detailed error context information."""

    component_type: ComponentType | None = None
    component_name: str | None = None
    operation: str | None = None
    config_section: str | None = None
    plugin_name: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class FactoryError(CodeWeaverFactoryError):
    """Comprehensive error information."""

    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    context: ErrorContext
    exception: Exception | None = None
    timestamp: float = Field(default_factory=time.time)
    traceback: str | None = None
    error_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    recovery_suggestions: list[str] = Field(default_factory=list)
    """Unique identifier for the error instance."""

    def __post_init__(self):
        """Post-init processing."""
        if self.exception and not self.traceback:
            self.traceback = "".join(
                traceback.format_exception(
                    type(self.exception), self.exception, self.exception.__traceback__
                )
            )


class ErrorHandler:
    """Centralized error handling system."""

    def __init__(self):
        """Initialize the error handler."""
        self._error_handlers: dict[ErrorCategory, list[Callable]] = defaultdict(list)
        self._error_log: list[FactoryError] = []
        self._recovery_strategies: dict[ErrorCategory, list[Callable]] = defaultdict(list)

    def register_error_handler(
        self, category: ErrorCategory, handler: Callable[[FactoryError], None]
    ) -> None:
        """Register an error handler for a specific category."""
        self._error_handlers[category].append(handler)

    def register_recovery_strategy(
        self, category: ErrorCategory, strategy: Callable[[FactoryError], bool]
    ) -> None:
        """Register a recovery strategy for a specific error category."""
        self._recovery_strategies[category].append(strategy)

    def handle_error(self, error: FactoryError) -> bool:
        """Handle an error using registered handlers and recovery strategies."""
        # Log the error
        self._error_log.append(error)
        logger.log(
            self._severity_to_log_level(error.severity),
            "Factory error [%s/%s]: %s",
            error.category.value,
            error.severity.value,
            error.message,
            extra={"error_context": error.context.__dict__},
        )

        # Execute category-specific handlers
        for handler in self._error_handlers[error.category]:
            try:
                handler(error)
            except Exception:
                logger.exception("Error handler failed")

        # Attempt recovery
        recovery_successful = False
        for strategy in self._recovery_strategies[error.category]:
            try:
                if strategy(error):
                    recovery_successful = True
                    logger.info("Recovery successful for error %s", error.error_id)
                    break
            except Exception:
                logger.exception("Recovery strategy failed.")

        return recovery_successful

    def create_error(
        self,
        severity: ErrorSeverity,
        category: ErrorCategory,
        message: str,
        context: ErrorContext | None = None,
        exception: Exception | None = None,
    ) -> FactoryError:
        """Create a standardized error object."""
        return FactoryError(
            severity=severity,
            category=category,
            message=message,
            context=context or ErrorContext(),
            exception=exception,
            recovery_suggestions=self._generate_recovery_suggestions(category, exception),
        )

    def _generate_recovery_suggestions(
        self, category: ErrorCategory, exception: Exception | None
    ) -> list[str]:
        """Generate recovery suggestions based on error category."""
        suggestions_map = {
            ErrorCategory.CONFIGURATION: [
                "Check configuration file syntax and completeness",
                "Verify all required environment variables are set",
                "Ensure configuration values are within valid ranges",
            ],
            ErrorCategory.COMPONENT: [
                "Verify component dependencies are installed",
                "Check component configuration parameters",
                "Ensure required services are running",
            ],
            ErrorCategory.PLUGIN: [
                "Verify plugin compatibility with current version",
                "Check plugin installation and dependencies",
                "Validate plugin configuration",
            ],
            ErrorCategory.NETWORK: [
                "Check network connectivity",
                "Verify service endpoints are accessible",
                "Check firewall and proxy settings",
            ],
            ErrorCategory.RESOURCE: [
                "Check available system resources (memory, disk, CPU)",
                "Verify database/storage accessibility",
                "Check service quotas and limits",
            ],
        }

        return suggestions_map.get(category, ["Contact support for assistance"])

    def _severity_to_log_level(self, severity: ErrorSeverity) -> int:
        """Convert error severity to logging level."""
        severity_map = {
            ErrorSeverity.TRACE: logging.DEBUG,
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL,
        }
        return severity_map.get(severity, logging.ERROR)


@dataclass
class ComponentFallbackResult:
    """Result of component fallback attempt."""

    success: bool
    fallback_component: Any | None = None
    strategy_name: str | None = None
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class GracefulDegradationManager:
    """Manages graceful degradation when components fail."""

    def __init__(self):
        """Initialize the graceful degradation manager."""
        self._fallback_strategies: dict[ComponentType, list[FallbackStrategy]] = {}

    def register_fallback_strategy(
        self, component_type: ComponentType, strategy: "FallbackStrategy"
    ) -> None:
        """Register a fallback strategy for a component type."""
        if component_type not in self._fallback_strategies:
            self._fallback_strategies[component_type] = []
        self._fallback_strategies[component_type].append(strategy)

    async def handle_component_failure(
        self,
        component_type: ComponentType,
        original_config: BaseComponentConfig,
        error: FactoryError,
    ) -> ComponentFallbackResult:
        """Handle component failure with graceful degradation."""
        fallback_strategies = self._fallback_strategies.get(component_type, [])

        for strategy in fallback_strategies:
            try:
                if await strategy.can_handle(original_config, error):
                    fallback_component = await strategy.create_fallback(original_config)

                    return ComponentFallbackResult(
                        success=True,
                        fallback_component=fallback_component,
                        strategy_name=strategy.get_name(),
                        limitations=strategy.get_limitations(),
                        warnings=[
                            f"Using fallback strategy: {strategy.get_name()}",
                            f"Limitations: {', '.join(strategy.get_limitations())}",
                        ],
                    )
            except Exception as e:
                logger.warning("Fallback strategy '%s' failed: %s", strategy.get_name(), e)

        return ComponentFallbackResult(
            success=False, errors=[f"No suitable fallback strategy for {component_type.value}"]
        )


class FallbackStrategy:
    """Base class for fallback strategies."""

    async def can_handle(self, original_config: BaseComponentConfig, error: FactoryError) -> bool:
        """Check if this strategy can handle the failure."""
        raise NotImplementedError

    async def create_fallback(self, original_config: BaseComponentConfig) -> Any:
        """Create a fallback component."""
        raise NotImplementedError

    def get_name(self) -> str:
        """Get strategy name."""
        raise NotImplementedError

    def get_limitations(self) -> list[str]:
        """Get list of limitations for this fallback."""
        raise NotImplementedError
