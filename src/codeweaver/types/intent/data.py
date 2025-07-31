# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Intent data structures for CodeWeaver."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from codeweaver.types.intent.enums import Complexity, IntentType, Scope


@dataclass
class ParsedIntent:
    """
    Parsed intent structure without INDEX support.

    This represents a user's natural language intent parsed into
    structured components for strategy selection and execution.
    """

    intent_type: IntentType
    """The type of intent (SEARCH, UNDERSTAND, or ANALYZE)."""

    primary_target: str
    """Main focus of the intent (e.g., 'authentication', 'database')."""

    scope: Scope
    """Scope of the operation (FILE, MODULE, PROJECT, SYSTEM)."""

    complexity: Complexity
    """Assessed complexity level."""

    confidence: float
    """Confidence score of the parsing (0.0-1.0)."""

    filters: dict[str, Any]
    """Additional filtering constraints."""

    metadata: dict[str, Any]
    """Parser and processing metadata."""

    # Timestamp information
    parsed_at: datetime
    """When the intent was parsed."""

    def __post_init__(self):
        """Set parsed_at timestamp if not provided."""
        if not hasattr(self, "parsed_at") or self.parsed_at is None:
            self.parsed_at = datetime.now(UTC)


@dataclass
class IntentResult:
    """
    Result of intent processing.

    This represents the outcome of executing an intent through
    the strategy system, including success status, data, and metadata.
    """

    success: bool
    """Whether the intent processing succeeded."""

    data: Any
    """The result data (search results, analysis, etc.)."""

    metadata: dict[str, Any]
    """Execution metadata including strategy used, timing, etc."""

    error_message: str | None = None
    """Error message if processing failed."""

    suggestions: list[str] | None = None
    """Suggested next actions or alternative queries."""

    # Execution information
    executed_at: datetime
    """When the intent was executed."""

    execution_time: float
    """Time taken to execute the intent in seconds."""

    strategy_used: str | None = None
    """Name of the strategy that was used."""

    def __post_init__(self):
        """Set executed_at timestamp if not provided."""
        if not hasattr(self, "executed_at") or self.executed_at is None:
            self.executed_at = datetime.now(UTC)
