# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Context intelligence service provider for FastMCP context mining."""

import hashlib
import logging
import re

from datetime import UTC, datetime, timedelta
from typing import Any

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import (
    ContextAdequacy,
    ContextIntelligenceService,
    ContextIntelligenceServiceConfig,
    LLMProfile,
    ServiceType,
)


class LLMModelDetector:
    """Detects LLM model characteristics from behavioral patterns."""

    def __init__(self):
        """Initialize the LLM model detector."""
        # Known model patterns based on behavioral characteristics
        self.model_signatures = {
            "claude": {
                "indicators": ["claude", "anthropic", "thinking", "comprehensive"],
                "timing_patterns": {"avg_response_time": (0.5, 3.0)},
                "request_patterns": ["detailed", "thorough", "structured"]
            },
            "gpt": {
                "indicators": ["openai", "gpt", "chatgpt", "assistant"],
                "timing_patterns": {"avg_response_time": (0.3, 2.0)},
                "request_patterns": ["concise", "direct", "helpful"]
            },
            "gemini": {
                "indicators": ["google", "gemini", "bard"],
                "timing_patterns": {"avg_response_time": (0.4, 2.5)},
                "request_patterns": ["analytical", "precise", "factual"]
            }
        }

    async def identify_model(self, behavioral_features: dict[str, Any]) -> tuple[str | None, float]:
        """Identify LLM model from behavioral features."""
        best_match = None
        best_confidence = 0.0

        for model_name, signature in self.model_signatures.items():
            confidence = self._calculate_model_confidence(behavioral_features, signature)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = model_name

        # Only return match if confidence is above threshold
        return (best_match, best_confidence) if best_confidence > 0.6 else (None, 0.0)

    def _calculate_model_confidence(
        self,
        features: dict[str, Any],
        signature: dict[str, Any]
    ) -> float:
        """Calculate confidence that features match a model signature."""
        # Check user agent indicators
        user_agent = features.get("user_agent", "").lower()
        indicator_matches = sum(
            indicator in user_agent for indicator in signature["indicators"]
        )
        indicator_confidence = indicator_matches / len(signature["indicators"])
        confidence_factors = [indicator_confidence * 0.4]
        # Check timing patterns
        timing_confidence = self._check_timing_patterns(
            features.get("timing", {}),
            signature["timing_patterns"]
        )
        confidence_factors.append(timing_confidence * 0.3)

        # Check request patterns
        request_confidence = self._check_request_patterns(
            features.get("request_patterns", []),
            signature["request_patterns"]
        )
        confidence_factors.append(request_confidence * 0.3)

        return sum(confidence_factors)

    def _check_timing_patterns(
        self,
        timing_data: dict[str, float],
        expected_patterns: dict[str, tuple[float, float]]
    ) -> float:
        """Check if timing data matches expected patterns."""
        if not timing_data or not expected_patterns:
            return 0.5  # Neutral confidence

        matches = 0
        total_checks = 0

        for pattern_name, (min_val, max_val) in expected_patterns.items():
            if pattern_name in timing_data:
                value = timing_data[pattern_name]
                if min_val <= value <= max_val:
                    matches += 1
                total_checks += 1

        return matches / total_checks if total_checks > 0 else 0.5

    def _check_request_patterns(
        self,
        request_patterns: list[str],
        expected_patterns: list[str]
    ) -> float:
        """Check if request patterns match expected patterns."""
        if not request_patterns or not expected_patterns:
            return 0.5  # Neutral confidence

        matches = sum(
            any(

                    pattern in req_pattern.lower()
                    for req_pattern in request_patterns

            )
            for pattern in expected_patterns
        )

        return matches / len(expected_patterns)


class ContextAdequacyPredictor:
    """Predicts context adequacy for intent processing."""

    def __init__(self):
        """Initialize the context adequacy predictor."""
        self.required_context_elements = {
            "search": ["query", "scope", "filters"],
            "analysis": ["target", "depth", "focus"],
            "explanation": ["topic", "detail_level", "audience"],
            "implementation": ["requirements", "constraints", "technology"]
        }

    async def assess(
        self,
        intent: str,
        context: dict[str, Any],
        contextual_features: dict[str, Any]
    ) -> ContextAdequacy:
        """Assess adequacy of available context for intent processing."""
        # Classify intent type
        intent_type = self._classify_intent_type(intent)

        # Get required elements for this intent type
        required_elements = self.required_context_elements.get(intent_type, ["query"])

        # Check which elements are present
        present_elements = []
        missing_elements = []

        for element in required_elements:
            if self._has_context_element(element, context, intent):
                present_elements.append(element)
            else:
                missing_elements.append(element)

        # Calculate adequacy score
        adequacy_score = len(present_elements) / len(required_elements) if required_elements else 1.0

        # Calculate richness score (amount of detail in present elements)
        richness_score = self._calculate_richness_score(context, present_elements)

        # Calculate clarity score (how clear the intent and context are)
        clarity_score = contextual_features.get("clarity_score", 0.7)

        # Generate recommendations for missing elements
        recommendations = self._generate_recommendations(missing_elements, intent_type)

        return ContextAdequacy(
            score=adequacy_score,
            missing_elements=missing_elements,
            richness_score=richness_score,
            clarity_score=clarity_score,
            recommendations=recommendations
        )

    def _classify_intent_type(self, intent: str) -> str:
        """Classify intent type for context requirements."""
        intent_lower = intent.lower()

        if any(word in intent_lower for word in ["find", "search", "look", "get", "list"]):
            return "search"
        if any(word in intent_lower for word in ["analyze", "review", "check", "examine", "assess"]):
            return "analysis"
        if any(word in intent_lower for word in ["explain", "describe", "what", "how", "why"]):
            return "explanation"
        if any(word in intent_lower for word in ["create", "build", "implement", "develop", "make"]):
            return "implementation"
        return "general"

    def _has_context_element(self, element: str, context: dict[str, Any], intent: str) -> bool:
        """Check if a context element is present."""
        element_checks = {
            "query": lambda: bool(intent.strip()),
            "scope": lambda: any(key in context for key in ["scope", "path", "directory", "file"]),
            "filters": lambda: any(key in context for key in ["filters", "language", "file_type"]),
            "target": lambda: any(key in context for key in ["target", "component", "function", "class"]),
            "depth": lambda: any(key in context for key in ["depth", "detail", "level"]),
            "focus": lambda: any(key in context for key in ["focus", "aspect", "area"]),
            "topic": lambda: any(key in context for key in ["topic", "subject", "concept"]),
            "detail_level": lambda: any(key in context for key in ["detail", "depth", "complexity"]),
            "audience": lambda: any(key in context for key in ["audience", "level", "experience"]),
            "requirements": lambda: any(key in context for key in ["requirements", "specs", "criteria"]),
            "constraints": lambda: any(key in context for key in ["constraints", "limitations", "bounds"]),
            "technology": lambda: any(key in context for key in ["technology", "framework", "language"])
        }

        check_func = element_checks.get(element)
        return check_func() if check_func else False

    def _calculate_richness_score(
        self,
        context: dict[str, Any],
        present_elements: list[str]
    ) -> float:
        """Calculate richness score based on detail in present elements."""
        if not present_elements:
            return 0.0

        richness_factors = []

        # Check for detailed vs. sparse context
        for value in context.values():
            if isinstance(value, str):
                # Longer strings indicate more detailed context
                detail_score = min(len(value) / 100.0, 1.0)  # Max score at 100 chars
                richness_factors.append(detail_score)
            elif isinstance(value, list | dict):
                # More complex structures indicate richer context
                complexity_score = min(len(str(value)) / 200.0, 1.0)
                richness_factors.append(complexity_score)

        return sum(richness_factors) / len(richness_factors) if richness_factors else 0.3

    def _generate_recommendations(
        self,
        missing_elements: list[str],
        intent_type: str
    ) -> list[str]:
        """Generate recommendations for improving context."""
        recommendations = []

        element_recommendations = {
            "query": "Provide a more specific search query or question",
            "scope": "Specify the scope (file, directory, or project level)",
            "filters": "Add filtering criteria (language, file type, etc.)",
            "target": "Identify the specific target (function, class, component)",
            "depth": "Specify the level of detail needed",
            "focus": "Clarify the specific aspect or area of interest",
            "topic": "Provide more context about the topic or subject",
            "detail_level": "Specify the desired level of detail",
            "audience": "Indicate the target audience or expertise level",
            "requirements": "Provide clear requirements or specifications",
            "constraints": "Specify any constraints or limitations",
            "technology": "Identify the technology stack or framework"
        }

        recommendations.extend(
            element_recommendations[element]
            for element in missing_elements
            if element in element_recommendations
        )
        # Add intent-type specific recommendations
        if intent_type == "search" and "scope" in missing_elements:
            recommendations.append("Consider specifying search scope to improve relevance")
        elif intent_type == "analysis" and "focus" in missing_elements:
            recommendations.append("Define the analysis focus for better results")

        return recommendations


class FastMCPContextMiningProvider(BaseServiceProvider, ContextIntelligenceService):
    """Context intelligence service provider for FastMCP context mining."""

    def __init__(
        self,
        service_type: ServiceType,
        config: ContextIntelligenceServiceConfig,
        logger: logging.Logger | None = None,
    ):
        """Initialize the FastMCP context mining provider."""
        super().__init__(service_type, config, logger)
        self._config: ContextIntelligenceServiceConfig = config

        # Core components
        self.model_detector = LLMModelDetector()
        self.context_predictor = ContextAdequacyPredictor()

        # Session tracking
        self._session_profiles: dict[str, LLMProfile] = {}
        self._session_contexts: dict[str, list[dict[str, Any]]] = {}

        # Cleanup old sessions periodically
        self._last_cleanup = datetime.now(UTC)

    async def _initialize_provider(self) -> None:
        """Initialize the FastMCP context mining provider."""
        self._logger.info(
            "FastMCP context mining provider initialized with config: %s",
            {
                "llm_identification_enabled": self._config.llm_identification_enabled,
                "behavioral_fingerprinting": self._config.behavioral_fingerprinting,
                "privacy_mode": self._config.privacy_mode,
            }
        )

    async def _shutdown_provider(self) -> None:
        """Shutdown the FastMCP context mining provider."""
        self._logger.info(
            "FastMCP context mining provider shutdown. Tracked %d sessions.",
            len(self._session_profiles)
        )

    async def _check_health(self) -> bool:
        """Check if the context intelligence service is healthy."""
        # Clean up old sessions
        await self._cleanup_old_sessions()

        # Service is healthy if we can process context and detect patterns
        return len(self._session_profiles) <= 1000  # Prevent memory issues

    async def extract_llm_characteristics(self, ctx: Any) -> LLMProfile:
        """Extract LLM behavioral characteristics from context."""
        session_id = self._get_session_id(ctx)

        # Check if we already have a profile for this session
        if session_id in self._session_profiles:
            profile = self._session_profiles[session_id]
            # Update with new data
            await self._update_profile_with_context(profile, ctx)
            return profile

        # Extract basic profile data
        profile_data = {
            "session_id": session_id,
            "user_agent": self._get_user_agent(ctx),
            "request_timing": self._get_request_timing(ctx),
            "interaction_patterns": []
        }

        # Analyze HTTP patterns if available
        if hasattr(ctx, "request") and ctx.request:
            http_profile = await self._analyze_http_patterns(ctx.request)
            profile_data |= http_profile

        # Build behavioral fingerprint
        behavioral_features = await self._extract_behavioral_features(profile_data)

        # Identify model if enabled
        identified_model = None
        confidence = 0.0
        if self._config.llm_identification_enabled:
            identified_model, confidence = await self.model_detector.identify_model(
                behavioral_features
            )

        # Create profile
        profile = LLMProfile(
            session_id=session_id,
            identified_model=identified_model,
            confidence=confidence,
            request_patterns=behavioral_features.get("request_patterns", []),
            timing_characteristics=behavioral_features.get("timing", {}),
            behavioral_features=behavioral_features
        )

        # Store profile
        self._session_profiles[session_id] = profile

        # Initialize session context tracking
        self._session_contexts[session_id] = []

        return profile

    async def analyze_context_adequacy(
        self,
        intent: str,
        available_context: dict[str, Any]
    ) -> ContextAdequacy:
        """Analyze adequacy of available context for intent processing."""
        # Extract contextual features
        contextual_features = await self._extract_contextual_features(intent, available_context)

        # Use context predictor to assess adequacy
        return await self.context_predictor.assess(intent, available_context, contextual_features)

    def _get_session_id(self, ctx: Any) -> str:
        """Get or generate session ID from context."""
        if hasattr(ctx, "session_id") and ctx.session_id:
            return str(ctx.session_id)

        # Generate session ID from context attributes
        session_data = []

        if hasattr(ctx, "request") and ctx.request and (hasattr(ctx.request, "client") and ctx.request.client):
            client_info = f"{ctx.request.client.host}:{ctx.request.client.port}"
            session_data.append(client_info)

        if hasattr(ctx, "user_agent"):
            session_data.append(str(ctx.user_agent))

        # Apply privacy mode
        if self._config.privacy_mode == "hash_identifiers":
            joined_session = "|".join(session_data)
            return hashlib.sha256(joined_session.encode()).hexdigest()[:16]
        if self._config.privacy_mode == "strict":
            # Generate random session ID for strict privacy
            import uuid
            return str(uuid.uuid4())[:16]
        # Use direct identifiers (less private)
        return "|".join(session_data)[:16] if session_data else "anonymous"

    def _get_user_agent(self, ctx: Any) -> str | None:
        """Extract user agent from context."""
        if hasattr(ctx, "user_agent"):
            return str(ctx.user_agent)

        if hasattr(ctx, "request") and ctx.request and hasattr(ctx.request, "headers"):
            return ctx.request.headers.get("user-agent")

        return None

    def _get_request_timing(self, ctx: Any) -> dict[str, float]:
        """Extract request timing information."""
        timing = {}

        if hasattr(ctx, "start_time"):
            timing["start_time"] = ctx.start_time

        if hasattr(ctx, "request") and ctx.request and hasattr(ctx.request, "receive_time"):
            timing["receive_time"] = ctx.request.receive_time

        return timing

    async def _analyze_http_patterns(self, request) -> dict[str, Any]:
        """Analyze HTTP request for LLM behavioral patterns."""
        # Sanitize headers (remove sensitive information)
        headers = self._sanitize_headers(getattr(request, "headers", {}))
        patterns = {"headers": headers}
        # Extract timing information
        timing = {}
        if hasattr(request, "receive_time"):
            timing["connection_time"] = request.receive_time
        if hasattr(request, "start_time"):
            timing["processing_start"] = request.start_time
        patterns["timing"] = timing

        # Extract client information (anonymized)
        client_info = {}
        if hasattr(request, "client") and request.client:
            if self._config.privacy_mode == "hash_identifiers":
                client_info["ip_hash"] = hashlib.sha256(
                    str(request.client.host).encode()
                ).hexdigest()[:16]
            client_info["port"] = request.client.port
        patterns["client_info"] = client_info

        return patterns

    def _sanitize_headers(self, headers: dict[str, Any]) -> dict[str, Any]:
        """Sanitize HTTP headers for privacy."""
        safe_headers = {}

        # List of headers that are safe to include
        safe_header_names = {
            "content-type", "accept", "accept-encoding", "accept-language",
            "cache-control", "connection", "host", "user-agent"
        }

        for name, value in headers.items():
            name_lower = name.lower()
            if name_lower in safe_header_names:
                if name_lower == "user-agent":
                    # Partially sanitize user agent
                    safe_headers[name] = self._sanitize_user_agent(str(value))
                else:
                    safe_headers[name] = str(value)

        return safe_headers

    def _sanitize_user_agent(self, user_agent: str) -> str:
        """Sanitize user agent string for privacy."""
        if self._config.privacy_mode == "strict":
            # Remove all identifying information
            return "sanitized_user_agent"
        if self._config.privacy_mode == "hash_identifiers":
            # Keep general pattern but hash specific versions
            ua_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:8]
            return f"client_{ua_hash}"
        # Keep user agent but remove potential sensitive info
        return re.sub(r'\b\d+\.\d+\.\d+\b', 'x.x.x', user_agent)  # Remove version numbers

    async def _extract_behavioral_features(self, profile_data: dict[str, Any]) -> dict[str, Any]:
        """Extract behavioral features from profile data."""
        features = {
            "user_agent": profile_data.get("user_agent", ""),
            "timing": profile_data.get("request_timing", {}),
            "request_patterns": [],
            "interaction_patterns": profile_data.get("interaction_patterns", [])
        }

        if user_agent := features["user_agent"]:
            features["request_patterns"] = self._extract_ua_patterns(user_agent)

        if timing_data := features["timing"]:
            features["timing_patterns"] = self._extract_timing_patterns(timing_data)

        return features

    def _extract_ua_patterns(self, user_agent: str) -> list[str]:
        """Extract patterns from user agent string."""
        patterns = []
        ua_lower = user_agent.lower()

        # Common LLM client patterns
        if "claude" in ua_lower:
            patterns.append("anthropic_client")
        if "openai" in ua_lower or "gpt" in ua_lower:
            patterns.append("openai_client")
        if "google" in ua_lower or "gemini" in ua_lower:
            patterns.append("google_client")

        # Browser patterns
        if "mozilla" in ua_lower:
            patterns.append("browser_client")
        if "curl" in ua_lower or "wget" in ua_lower:
            patterns.append("cli_client")

        # API client patterns
        if "python" in ua_lower:
            patterns.append("python_client")
        if "javascript" in ua_lower or "node" in ua_lower:
            patterns.append("js_client")

        return patterns

    def _extract_timing_patterns(self, timing_data: dict[str, float]) -> dict[str, float]:
        """Extract timing patterns from timing data."""
        patterns = {}

        # Calculate derived timing metrics
        if "start_time" in timing_data and "receive_time" in timing_data:
            processing_time = timing_data["start_time"] - timing_data["receive_time"]
            patterns["processing_delay"] = processing_time

        # Add raw timing data
        patterns |= timing_data

        return patterns

    async def _extract_contextual_features(
        self,
        intent: str,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract contextual features for adequacy assessment."""
        features = {"clarity_score": self._assess_intent_clarity(intent)}

        # Context richness assessment
        features["richness_score"] = self._assess_context_richness(context)

        # Intent complexity assessment
        features["complexity_score"] = self._assess_intent_complexity(intent)

        return features

    def _assess_intent_clarity(self, intent: str) -> float:
        """Assess clarity of the intent statement."""
        # Length factor (too short or too long reduces clarity)
        length = len(intent.strip())
        if 10 <= length <= 200:
            length_score = 1.0
        elif 5 <= length <= 300:
            length_score = 0.8
        else:
            length_score = 0.5
        # Specificity factor (presence of specific terms)
        specific_terms = ["find", "search", "analyze", "explain", "create", "show", "get"]
        specificity_score = 1.0 if any(term in intent.lower() for term in specific_terms) else 0.6
        # Question word factor (questions are often clearer)
        question_words = ["what", "how", "why", "where", "when", "which", "who"]
        question_score = 1.0 if any(word in intent.lower() for word in question_words) else 0.8
        clarity_factors = [length_score, specificity_score, question_score]
        return sum(clarity_factors) / len(clarity_factors)

    def _assess_context_richness(self, context: dict[str, Any]) -> float:
        """Assess richness of the provided context."""
        if not context:
            return 0.1

        # Number of context elements
        element_count = len(context)
        element_score = min(element_count / 5.0, 1.0)  # Max score at 5 elements
        # Depth of context elements
        total_content_length = sum(
            len(str(value)) for value in context.values()
        )
        depth_score = min(total_content_length / 500.0, 1.0)  # Max score at 500 chars
        richness_factors = [element_score, depth_score]
        # Variety of context types
        value_types = {type(value).__name__ for value in context.values()}
        variety_score = min(len(value_types) / 3.0, 1.0)  # Max score at 3 types
        richness_factors.append(variety_score)

        return sum(richness_factors) / len(richness_factors)

    def _assess_intent_complexity(self, intent: str) -> float:
        """Assess complexity of the intent."""
        # Compound operations (multiple verbs or connectors)
        connectors = ["and", "or", "then", "also", "but", "however", "additionally"]
        connector_count = sum(conn in intent.lower() for conn in connectors)
        connector_score = min(connector_count / 2.0, 1.0)
        complexity_factors = [connector_score]
        # Technical terms
        technical_terms = [
            "function", "class", "method", "variable", "algorithm", "pattern",
            "implementation", "architecture", "framework", "library", "api"
        ]
        tech_count = sum(term in intent.lower() for term in technical_terms)
        tech_score = min(tech_count / 3.0, 1.0)
        complexity_factors.extend((tech_score, min(len(intent.split()) / 20.0, 1.0)))
        return sum(complexity_factors) / len(complexity_factors)

    async def _update_profile_with_context(self, profile: LLMProfile, ctx: Any) -> None:
        """Update existing profile with new context information."""
        # Update timing characteristics
        new_timing = self._get_request_timing(ctx)
        for key, value in new_timing.items():
            if key in profile.timing_characteristics:
                # Running average
                profile.timing_characteristics[key] = (
                    profile.timing_characteristics[key] * 0.8 + value * 0.2
                )
            else:
                profile.timing_characteristics[key] = value

        # Update behavioral features
        if hasattr(ctx, "request") and ctx.request:
            http_profile = await self._analyze_http_patterns(ctx.request)
            profile.behavioral_features.update(http_profile)

    async def _cleanup_old_sessions(self) -> None:
        """Clean up old session data to prevent memory leaks."""
        # Only cleanup if enough time has passed
        if (datetime.now(UTC) - self._last_cleanup).total_seconds() < 3600:  # 1 hour
            return

        cutoff_time = datetime.now(UTC) - timedelta(seconds=self._config.session_timeout)

        # Remove old profiles
        expired_sessions = [
            session_id for session_id, profile in self._session_profiles.items()
            if profile.created_at < cutoff_time
        ]

        for session_id in expired_sessions:
            del self._session_profiles[session_id]
            if session_id in self._session_contexts:
                del self._session_contexts[session_id]

        self._last_cleanup = datetime.now(UTC)

        if expired_sessions:
            self._logger.debug("Cleaned up %d expired sessions", len(expired_sessions))
