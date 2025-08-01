# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Base types and protocols for intent strategies."""

from typing import Any, Protocol, runtime_checkable

from codeweaver.cw_types.intent.data import IntentResult, ParsedIntent


@runtime_checkable
class IntentStrategy(Protocol):
    """
    Protocol for intent execution strategies.

    This defines the interface that all intent strategies must implement
    to be compatible with the strategy registry and selection system.
    """

    async def can_handle(self, parsed_intent: ParsedIntent) -> float:
        """
        Check if this strategy can handle the given intent.

        Args:
            parsed_intent: The parsed intent to evaluate

        Returns:
            Confidence score (0.0-1.0) indicating how well this strategy
            can handle the intent. 0.0 means cannot handle, 1.0 means
            perfect match.
        """
        ...

    async def execute(self, parsed_intent: ParsedIntent, context: dict[str, Any]) -> IntentResult:
        """
        Execute the intent using this strategy.

        Args:
            parsed_intent: The parsed intent to execute
            context: Execution context with services and metadata

        Returns:
            Result of the intent execution
        """
        ...
