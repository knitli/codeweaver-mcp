# sourcery skip: do-not-use-staticmethod
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Factory for creating intent parsers."""

import logging
import re

from typing import Any, ClassVar, Protocol, runtime_checkable

from codeweaver.types import IntentParsingError, ParsedIntent


def _raise_intent_error(message: str) -> None:
    """Raise an IntentParsingError with the given message."""
    raise IntentParsingError(message)


@runtime_checkable
class IntentParser(Protocol):
    """Protocol for intent parsers."""

    async def parse(self, intent_text: str) -> ParsedIntent:
        """Parse intent text into structured format."""
        ...


class IntentParserFactory:
    """
    Factory for creating intent parsers.

    This factory provides a unified interface for creating different
    types of intent parsers while allowing for future extensibility.

    Currently supported parsers:
    - "pattern": Pattern-based regex parser (default)

    Future parsers (Phase 2):
    - "nlp": NLP-enhanced parser with spaCy
    - "hybrid": Combination of pattern and NLP parsing
    """

    _logger = logging.getLogger(__name__)
    _patterns: ClassVar[set[re.compile | None]] = set()

    @staticmethod
    def create(config: dict[str, Any]) -> IntentParser:
        """
        Create parser based on configuration.

        Args:
            config: Parser configuration dictionary
                - type: Parser type ("pattern", "nlp", "hybrid")
                - Additional type-specific configuration

        Returns:
            Configured intent parser instance

        Raises:
            IntentParsingError: If parser type is unknown or configuration is invalid
        """
        parser_type = config.get("type", "pattern")

        try:
            match parser_type:
                case "pattern":
                    return IntentParserFactory._create_pattern_parser(config)
                case "nlp":
                    return IntentParserFactory._create_nlp_parser(config)
                case "hybrid":
                    return IntentParserFactory._create_hybrid_parser(config)
                case _:
                    _raise_intent_error(
                        f"Unknown parser type '{parser_type}'. Available: {IntentParserFactory.get_available_parsers()}"
                    )
        except Exception as e:
            IntentParserFactory._logger.exception("Failed to create parser: %s", parser_type)
            raise IntentParsingError(f"Failed to create parser '{parser_type}': {e}") from e

    @classmethod
    def patterns(cls) -> set[re.Pattern]:
        """
        Get compiled regex patterns used by the factory.

        Returns:
            Set of compiled regex patterns
        """
        return cls._patterns

    @staticmethod
    def _create_pattern_parser(config: dict[str, Any]) -> IntentParser:
        """Create pattern-based parser."""
        from codeweaver.intent.parsing.pattern_matcher import PatternBasedParser

        IntentParserFactory._logger.info("Creating pattern-based intent parser")
        parser = PatternBasedParser()

        # Apply any pattern-specific configuration
        if "custom_patterns" in config:
            parser.patterns.update(config["custom_patterns"])

        return parser

    @staticmethod
    def _create_nlp_parser(config: dict[str, Any]) -> IntentParser:
        """Create NLP-enhanced parser (Phase 2 implementation)."""
        # This will be implemented in Phase 2
        # For now, fall back to pattern parser with a warning
        IntentParserFactory._logger.warning(
            "NLP parser not yet implemented, falling back to pattern parser"
        )
        return IntentParserFactory._create_pattern_parser(config)

    @staticmethod
    def _create_hybrid_parser(config: dict[str, Any]) -> IntentParser:
        """Create hybrid parser combining pattern and NLP (Phase 2 implementation)."""
        # This will be implemented in Phase 2
        # For now, fall back to pattern parser with a warning
        IntentParserFactory._logger.warning(
            "Hybrid parser not yet implemented, falling back to pattern parser"
        )
        return IntentParserFactory._create_pattern_parser(config)

    @staticmethod
    def get_available_parsers() -> list[str]:
        """Get list of available parser types."""
        return ["pattern", "nlp", "hybrid"]

    @staticmethod
    def get_parser_info(parser_type: str) -> dict[str, dict[str, str]]:
        """Get information about a specific parser type."""
        parser_info = {
            "pattern": {
                "name": "Pattern-based Parser",
                "description": "Uses regex patterns to identify intents",
                "accuracy": "85-90%",
                "speed": "Very Fast",
                "memory": "Low",
                "dependencies": "None",
                "supported_languages": "All (language agnostic)",
                "phase": "1 (Available)",
            },
            "nlp": {
                "name": "NLP-enhanced Parser",
                "description": "Uses spaCy NLP models for advanced parsing",
                "accuracy": "92-95%",
                "speed": "Fast",
                "memory": "Medium",
                "dependencies": "spaCy, language models",
                "supported_languages": "English (primary)",
                "phase": "2 (Planned)",
            },
            "hybrid": {
                "name": "Hybrid Parser",
                "description": "Combines pattern matching with NLP analysis",
                "accuracy": "95-98%",
                "speed": "Medium",
                "memory": "Medium-High",
                "dependencies": "spaCy, language models",
                "supported_languages": "English (primary)",
                "phase": "2 (Planned)",
            },
        }

        return parser_info.get(parser_type, {})

    @staticmethod
    def validate_config(config: dict[str, Any]) -> tuple[bool, str]:
        """
        Validate parser configuration.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"

        parser_type = config.get("type")
        if not parser_type:
            return False, "Parser type is required"

        available_parsers = IntentParserFactory.get_available_parsers()
        if parser_type not in available_parsers:
            return False, f"Unknown parser type '{parser_type}'. Available: {available_parsers}"

        # Type-specific validation
        if parser_type == "pattern":
            return IntentParserFactory._validate_pattern_config(config)
        if parser_type == "nlp":
            return IntentParserFactory._validate_nlp_config(config)
        if parser_type == "hybrid":
            return IntentParserFactory._validate_hybrid_config(config)

        return True, ""

    @staticmethod
    def _validate_pattern_config(config: dict[str, Any]) -> tuple[bool, str]:
        """Validate pattern parser configuration."""
        # Custom patterns validation
        if "custom_patterns" in config:
            custom_patterns = config["custom_patterns"]
            if not isinstance(custom_patterns, dict):
                return False, "custom_patterns must be a dictionary"

            for pattern_type, patterns in custom_patterns.items():
                if not isinstance(patterns, list):
                    return False, f"Patterns for {pattern_type} must be a list"

                for pattern in patterns:
                    try:
                        compiled_pattern = re.compile(pattern)
                        IntentParserFactory._patterns.add(compiled_pattern)
                    except re.error as e:
                        return False, f"Invalid regex pattern '{pattern}': {e}"

        return True, ""

    @staticmethod
    def _validate_nlp_config(config: dict[str, Any]) -> tuple[bool, str]:
        """Validate NLP parser configuration (Phase 2)."""
        # Placeholder for Phase 2 implementation
        return True, ""

    @staticmethod
    def _validate_hybrid_config(config: dict[str, Any]) -> tuple[bool, str]:
        """Validate hybrid parser configuration (Phase 2)."""
        # Placeholder for Phase 2 implementation
        return True, ""
