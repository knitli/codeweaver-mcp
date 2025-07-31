"""Pattern-based intent parser without INDEX support."""

import logging
import re

from typing import Any

from codeweaver.types import Complexity, IntentParsingError, IntentType, ParsedIntent, Scope


class PatternBasedParser:
    """
    Pattern-based parser without INDEX intent support.

    This parser uses regex patterns to identify user intents and extract
    key information.

    Supported intents:
    - SEARCH: Find specific code elements, functions, or patterns
    - UNDERSTAND: Understand system architecture or code organization
    - ANALYZE: Analyze code for issues, patterns, or improvements

    The parser extracts:
    - Intent type (SEARCH, UNDERSTAND, ANALYZE)
    - Primary target (what the user is looking for)
    - Scope (FILE, MODULE, PROJECT, SYSTEM)
    - Complexity (SIMPLE, MODERATE, COMPLEX, ADAPTIVE)
    - Confidence score (0.0-1.0)
    """

    def __init__(self):
        """Initialize the pattern-based parser."""
        self.logger = logging.getLogger(__name__)
        self.patterns = self._load_intent_patterns_no_index()
        from codeweaver.intent.parsing.confidence_scorer import BasicConfidenceScorer

        self.confidence_scorer = BasicConfidenceScorer()

    def _load_intent_patterns_no_index(self) -> dict[str, list[str]]:
        """Load patterns excluding INDEX intents."""
        return {
            "search_patterns": [
                "find\\s+(.+)",
                "search\\s+for\\s+(.+)",
                "search\\s+(.+)",
                "locate\\s+(.+)",
                "show\\s+me\\s+(.+)",
                "get\\s+(.+)",
                "look\\s+for\\s+(.+)",
                "where\\s+is\\s+(.+)",
                "where\\s+are\\s+(.+)",
                "list\\s+(.+)",
            ],
            "understand_patterns": [
                "understand\\s+(.+)",
                "explain\\s+(.+)",
                "how\\s+does\\s+(.+)\\s+work",
                "what\\s+is\\s+(.+)",
                "what\\s+are\\s+(.+)",
                "describe\\s+(.+)",
                "tell\\s+me\\s+about\\s+(.+)",
                "help\\s+me\\s+understand\\s+(.+)",
                "walk\\s+me\\s+through\\s+(.+)",
                "overview\\s+of\\s+(.+)",
            ],
            "analyze_patterns": [
                "analyze\\s+(.+)",
                "review\\s+(.+)",
                "investigate\\s+(.+)",
                "check\\s+(.+)\\s+for",
                "examine\\s+(.+)",
                "assess\\s+(.+)",
                "audit\\s+(.+)",
                "evaluate\\s+(.+)",
                "identify\\s+(.+)\\s+in",
                "look\\s+for\\s+(.+)\\s+issues",
            ],
        }

    async def parse(self, intent_text: str) -> ParsedIntent:
        """
        Parse intent without INDEX support.

        Args:
            intent_text: Natural language intent from user

        Returns:
            ParsedIntent structure with extracted information

        Raises:
            IntentParsingError: If parsing fails
        """
        try:
            self.logger.debug("Parsing intent: %s", intent_text[:100])
            normalized_text = self._normalize_text(intent_text)
            intent_type = self._detect_intent_type_no_index(normalized_text)
            primary_target = self._extract_target(normalized_text, intent_type)
            scope = self._assess_scope(normalized_text, primary_target)
            complexity = self._assess_complexity(normalized_text, scope, intent_type)
            confidence = await self.confidence_scorer.score(
                intent_text, intent_type, primary_target
            )
            filters = self._extract_filters(normalized_text)
            metadata = {
                "parser": "pattern_based",
                "patterns_matched": self._get_matched_patterns(normalized_text, intent_type),
                "index_support": False,
                "normalized_text": normalized_text,
                "background_indexing_note": "Indexing handled automatically in background",
            }
            parsed_intent = ParsedIntent(
                intent_type=intent_type,
                primary_target=primary_target,
                scope=scope,
                complexity=complexity,
                confidence=confidence,
                filters=filters,
                metadata=metadata,
            )
            self.logger.debug(
                "Parsed intent: type=%s, target=%s, confidence=%.2f",
                intent_type.value,
                primary_target,
                confidence,
            )
        except Exception as e:
            self.logger.exception("Failed to parse intent: %s", intent_text)
            raise IntentParsingError(f"Failed to parse intent '{intent_text}': {e}") from e
        else:
            return parsed_intent

    def _normalize_text(self, text: str) -> str:
        """Normalize input text for better pattern matching."""
        normalized = text.lower().strip()
        normalized = re.sub("\\s+", " ", normalized)
        normalized = normalized.replace("can you", "")
        normalized = normalized.replace("could you", "")
        normalized = normalized.replace("please", "")
        normalized = normalized.replace("help me", "")
        return normalized.strip()

    def _detect_intent_type_no_index(self, text: str) -> IntentType:
        """Detect intent type without INDEX option."""
        index_keywords = ["index", "indexing", "build index", "create index", "reindex"]
        if any((keyword in text for keyword in index_keywords)):
            self.logger.info("INDEX intent detected - redirecting to background service note")
            return IntentType.SEARCH
        for pattern in self.patterns["search_patterns"]:
            if re.search(pattern, text):
                return IntentType.SEARCH
        for pattern in self.patterns["understand_patterns"]:
            if re.search(pattern, text):
                return IntentType.UNDERSTAND
        for pattern in self.patterns["analyze_patterns"]:
            if re.search(pattern, text):
                return IntentType.ANALYZE
        return IntentType.SEARCH

    def _extract_target(self, text: str, intent_type: IntentType) -> str:
        """Extract the primary target of the intent."""
        for pattern in self.patterns[f"{intent_type.value}_patterns"]:
            match = re.search(pattern, text)
            if match:
                target = match.group(1).strip()
                if target:
                    return self._clean_target(target)
        return self._extract_fallback_target(text)

    def _clean_target(self, target: str) -> str:
        """Clean extracted target text."""
        stop_words = ["in", "on", "at", "for", "with", "by", "from", "to", "of"]
        words = target.split()
        while words and words[-1] in stop_words:
            words.pop()
        return " ".join(words) if words else target

    def _extract_fallback_target(self, text: str) -> str:
        """Extract target when pattern matching fails."""
        words = text.split()
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "find",
            "search",
            "look",
            "show",
            "get",
            "understand",
            "explain",
            "analyze",
            "review",
            "check",
            "examine",
        }
        key_words = [word for word in words if word not in common_words and len(word) > 2]
        return " ".join(key_words[:3]) if key_words else "code"

    def _assess_scope(self, text: str, target: str) -> Scope:
        """Assess the scope of the intent."""
        system_indicators = [
            "entire",
            "whole",
            "all",
            "system",
            "project",
            "codebase",
            "application",
            "app",
            "everywhere",
            "globally",
        ]
        if any((indicator in text for indicator in system_indicators)):
            return Scope.SYSTEM
        project_indicators = ["project", "repository", "repo", "across", "throughout"]
        if any((indicator in text for indicator in project_indicators)):
            return Scope.PROJECT
        module_indicators = ["module", "package", "directory", "folder", "component"]
        if any((indicator in text for indicator in module_indicators)):
            return Scope.MODULE
        file_indicators = ["file", "this file", "current file", "single file"]
        if any((indicator in text for indicator in file_indicators)):
            return Scope.FILE
        if len(target.split()) > 3:
            return Scope.PROJECT
        return Scope.MODULE

    def _assess_complexity(self, text: str, scope: Scope, intent_type: IntentType) -> Complexity:
        """Assess the complexity level of the intent."""
        complex_indicators = [
            "performance",
            "optimization",
            "bottleneck",
            "security",
            "vulnerability",
            "architecture",
            "design pattern",
            "refactor",
            "complex",
            "advanced",
            "deep",
            "thorough",
            "comprehensive",
            "detailed analysis",
        ]
        if any((indicator in text for indicator in complex_indicators)):
            return Complexity.COMPLEX
        moderate_indicators = [
            "how",
            "why",
            "relationship",
            "connection",
            "workflow",
            "process",
            "integration",
            "dependency",
            "multiple",
            "several",
            "various",
        ]
        if any((indicator in text for indicator in moderate_indicators)):
            return Complexity.MODERATE
        if scope in [Scope.SYSTEM, Scope.PROJECT]:
            return Complexity.MODERATE
        if intent_type == IntentType.ANALYZE or intent_type == IntentType.UNDERSTAND:
            return Complexity.MODERATE
        return Complexity.SIMPLE

    def _extract_filters(self, text: str) -> dict[str, Any]:
        """Extract additional filtering constraints."""
        filters = {}
        languages = ["python", "javascript", "typescript", "java", "go", "rust", "c++", "c#"]
        for lang in languages:
            if lang in text:
                filters["language"] = lang
                break
        if "test" in text:
            filters["include_tests"] = True
        if "config" in text or "configuration" in text:
            filters["include_config"] = True
        if "recent" in text or "latest" in text or "new" in text:
            filters["recent_only"] = True
        return filters

    def _get_matched_patterns(self, text: str, intent_type: IntentType) -> list[str]:
        """Get list of patterns that matched the text."""
        matched = []
        for pattern in self.patterns[f"{intent_type.value}_patterns"]:
            if re.search(pattern, text):
                matched.append(pattern)
        return matched
