# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Confidence scoring system for parsed intents."""

import logging

from codeweaver.cw_types import IntentType


class BasicConfidenceScorer:
    """
    Confidence scoring for parsed intents.

    This scorer evaluates how confident we are in the intent parsing
    based on multiple factors:
    - Pattern match strength
    - Text clarity and specificity
    - Presence of key indicators
    - Absence of ambiguity markers

    Returns scores from 0.0 (no confidence) to 1.0 (high confidence).
    """

    def __init__(self):
        """Initialize the confidence scorer."""
        self.logger = logging.getLogger(__name__)

        # Strong indicator words for each intent type
        self.strong_indicators = {
            IntentType.SEARCH: ["find", "search", "locate", "show", "get", "where", "list"],
            IntentType.UNDERSTAND: ["understand", "explain", "how", "what", "describe", "overview"],
            IntentType.ANALYZE: ["analyze", "review", "investigate", "check", "examine", "audit"],
        }

        # Weak/ambiguous words that reduce confidence
        self.ambiguity_markers = [
            "maybe",
            "perhaps",
            "might",
            "could",
            "possibly",
            "seems",
            "kind of",
            "sort of",
            "something",
            "anything",
            "whatever",
        ]

        # Technical terms that increase confidence
        self.technical_terms = [
            "function",
            "class",
            "method",
            "variable",
            "api",
            "endpoint",
            "database",
            "authentication",
            "authorization",
            "security",
            "performance",
            "protocol",
            "optimization",
            "bug",
            "error",
            "exception",
            "module",
            "component",
            "service",
            "struct",
            "interface",
            "configuration",
        ]

    async def score(self, intent_text: str, intent_type: IntentType, primary_target: str) -> float:
        """
        Score confidence of intent parsing (0.0-1.0).

        Args:
            intent_text: Original intent text
            primary_target: Extracted primary target
            intent_type: Detected intent type

        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            self.logger.debug("Scoring confidence for intent: %s", intent_text[:50])

            # Start with base confidence
            confidence = 0.5

            # Factor 1: Pattern match strength (30% weight)
            pattern_score = self._score_pattern_match(intent_text, intent_type)
            confidence += (pattern_score - 0.5) * 0.3

            # Factor 2: Text clarity and specificity (25% weight)
            clarity_score = self._score_text_clarity(intent_text, primary_target)
            confidence += (clarity_score - 0.5) * 0.25

            # Factor 3: Presence of strong indicators (20% weight)
            indicator_score = self._score_strong_indicators(intent_text, intent_type)
            confidence += (indicator_score - 0.5) * 0.2

            # Factor 4: Technical term presence (15% weight)
            technical_score = self._score_technical_terms(intent_text, primary_target)
            confidence += (technical_score - 0.5) * 0.15

            # Factor 5: Ambiguity penalty (10% weight)
            ambiguity_penalty = self._score_ambiguity_penalty(intent_text)
            confidence -= ambiguity_penalty * 0.1

            # Ensure score is within bounds
            confidence = max(0.0, min(1.0, confidence))

            self.logger.debug(
                "Confidence score: %.3f (pattern=%.2f, clarity=%.2f, indicators=%.2f, technical=%.2f, ambiguity=%.2f)",
                confidence,
                pattern_score,
                clarity_score,
                indicator_score,
                technical_score,
                ambiguity_penalty,
            )

        except Exception as e:
            self.logger.warning("Failed to score confidence: %s", e)
            return 0.5  # Default neutral confidence
        else:
            return confidence

    def _score_pattern_match(self, text: str, intent_type: IntentType) -> float:
        """Score based on how well the text matches expected patterns."""
        text_lower = text.lower()
        strong_patterns = self.strong_indicators[intent_type]

        # Check for exact matches of strong indicators
        exact_matches = sum(pattern in text_lower for pattern in strong_patterns)

        if exact_matches >= 2:
            return 0.9  # Very strong match
        if exact_matches == 1:
            return 0.7  # Good match
        # Check for partial matches or similar words
        partial_score = self._check_partial_matches(text_lower, strong_patterns)
        return 0.3 + partial_score * 0.4  # Moderate to weak match

    def _check_partial_matches(self, text: str, patterns: list[str]) -> float:
        """Check for partial matches with pattern words."""
        score = 0.0

        for pattern in patterns:
            if len(pattern) > 3:
                # Look for partial matches (e.g., "search" matches "searching")
                pattern_root = pattern[:4]
                if pattern_root in text:
                    score += 0.5
                    break

        return min(1.0, score)

    def _score_text_clarity(self, text: str, target: str) -> float:
        """Score based on text clarity and specificity."""
        score = 0.5

        # Length indicators
        word_count = len(text.split())
        if 5 <= word_count <= 15:
            score += 0.2  # Good length
        elif word_count < 3:
            score -= 0.3  # Too short
        elif word_count > 20:
            score -= 0.1  # Might be too verbose

        # Target specificity
        target_words = len(target.split())
        if target_words >= 2:
            score += 0.2  # Specific target
        elif not target or target == "code":
            score -= 0.2  # Generic target

        # Grammar and structure
        if text.endswith("?"):
            score += 0.1  # Well-formed question

        # Check for complete sentences
        if text.count(" ") >= 2 and any(word in text.lower() for word in ["the", "a", "an"]):
            score += 0.1  # Better grammar

        return max(0.0, min(1.0, score))

    def _score_strong_indicators(self, text: str, intent_type: IntentType) -> float:
        """Score based on presence of strong intent indicators."""
        text_lower = text.lower()
        indicators = self.strong_indicators[intent_type]

        if found_indicators := [ind for ind in indicators if ind in text_lower]:
            return 0.6 if len(found_indicators) == 1 else 0.9
        return 0.2  # No strong indicators

    def _score_technical_terms(self, text: str, target: str) -> float:
        """Score based on presence of technical programming terms."""
        combined_text = f"{text} {target}".lower()

        # Count technical terms
        found_terms = [term for term in self.technical_terms if term in combined_text]
        match len(found_terms):
            case 0:
                return 0.3  # No technical context
            case 1:
                return 0.6  # Some technical context
            case 2:
                return 0.8  # Good technical context
            case _:
                return 0.9  # Rich technical context

    def _score_ambiguity_penalty(self, text: str) -> float:
        """Calculate penalty for ambiguous language."""
        text_lower = text.lower()

        # Count ambiguity markers
        ambiguity_count = sum(marker in text_lower for marker in self.ambiguity_markers)

        if ambiguity_count == 0:
            return 0.0  # No penalty
        return 0.2 if ambiguity_count == 1 else 0.5

    def get_confidence_explanation(self, confidence: float) -> str:
        """Get human-readable explanation of confidence level."""
        if confidence >= 0.8:
            return "High confidence - clear intent with specific technical terms"
        if confidence >= 0.6:
            return "Good confidence - intent is clear with some specificity"
        if confidence >= 0.4:
            return "Moderate confidence - intent is somewhat clear but could be more specific"
        if confidence >= 0.2:
            return "Low confidence - intent is unclear or very generic"
        return "Very low confidence - unable to clearly determine intent"
