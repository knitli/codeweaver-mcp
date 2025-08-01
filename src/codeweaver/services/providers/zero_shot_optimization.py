# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Zero-shot optimization service provider for first-attempt success optimization."""

import hashlib
import logging

from datetime import UTC, datetime
from typing import Any

from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.types import (
    OptimizationStrategy,
    OptimizedIntentPlan,
    ServiceType,
    SuccessPrediction,
    ZeroShotOptimizationService,
    ZeroShotOptimizationServiceConfig,
)


class ContextAdequacyPredictor:
    """Predicts context adequacy for zero-shot success."""

    def __init__(self):
        """Initialize the context adequacy predictor."""
        self.adequacy_thresholds = {
            "excellent": 0.9,
            "good": 0.7,
            "acceptable": 0.5,
            "poor": 0.3
        }

    async def assess(self, intent: str, context: dict[str, Any]) -> float:
        """Assess context adequacy for zero-shot success."""
        # Intent specificity
        specificity_score = self._assess_intent_specificity(intent)
        adequacy_factors = [specificity_score * 0.3]
        # Context completeness
        completeness_score = self._assess_context_completeness(intent, context)
        adequacy_factors.append(completeness_score * 0.4)

        # Context relevance
        relevance_score = self._assess_context_relevance(intent, context)
        adequacy_factors.append(relevance_score * 0.3)

        return sum(adequacy_factors)

    def _assess_intent_specificity(self, intent: str) -> float:
        """Assess how specific the intent is."""
        specificity_indicators = [
            ("specific terms", ["find", "get", "analyze", "explain", "create", "show"]),
            ("technical terms", ["function", "class", "method", "variable", "file", "directory"]),
            ("scope indicators", ["in", "from", "within", "inside", "under", "across"]),
            ("filter terms", ["type", "kind", "format", "extension", "containing", "matching"])
        ]

        intent_lower = intent.lower()
        specificity_score = 0.0

        for _category, terms in specificity_indicators:
            category_score = sum(term in intent_lower for term in terms)
            category_score = min(category_score / len(terms), 1.0)
            specificity_score += category_score

        return specificity_score / len(specificity_indicators)

    def _assess_context_completeness(self, intent: str, context: dict[str, Any]) -> float:
        """Assess completeness of context for the given intent."""
        # Determine required context elements based on intent
        required_elements = self._get_required_elements(intent)

        if not required_elements:
            return 0.8  # Default score if no specific requirements

        present_elements = sum(bool(self._has_context_element(element, context))
                           for element in required_elements)
        return present_elements / len(required_elements)

    def _assess_context_relevance(self, intent: str, context: dict[str, Any]) -> float:
        """Assess relevance of provided context to the intent."""
        if not context:
            return 0.2

        relevance_score = 0.0
        intent_keywords = set(intent.lower().split())

        for key, value in context.items():
            key_relevance = self._calculate_keyword_relevance(key.lower(), intent_keywords)
            value_relevance = self._calculate_keyword_relevance(str(value).lower(), intent_keywords)

            element_relevance = max(key_relevance, value_relevance)
            relevance_score += element_relevance

        return min(relevance_score / len(context), 1.0)

    def _get_required_elements(self, intent: str) -> list[str]:
        """Get required context elements based on intent type."""
        intent_lower = intent.lower()

        if any(word in intent_lower for word in ["search", "find", "look"]):
            return ["query", "scope"]
        if any(word in intent_lower for word in ["analyze", "review", "examine"]):
            return ["target", "scope", "focus"]
        if any(word in intent_lower for word in ["explain", "describe", "what"]):
            return ["topic", "scope"]
        if any(word in intent_lower for word in ["create", "build", "generate"]):
            return ["requirements", "specifications"]
        return ["scope"]

    def _has_context_element(self, element: str, context: dict[str, Any]) -> bool:
        """Check if context contains the required element."""
        element_mappings = {
            "query": ["query", "search", "term", "keyword", "text"],
            "scope": ["scope", "path", "directory", "file", "location", "area"],
            "target": ["target", "component", "function", "class", "method", "object"],
            "focus": ["focus", "aspect", "area", "type", "kind", "category"],
            "topic": ["topic", "subject", "concept", "theme"],
            "requirements": ["requirements", "specs", "criteria", "rules"],
            "specifications": ["specifications", "details", "parameters", "config"]
        }

        possible_keys = element_mappings.get(element, [element])

        return any(
            any(key in context_key.lower() for key in possible_keys)
            for context_key in context
        )

    def _calculate_keyword_relevance(self, text: str, intent_keywords: set[str]) -> float:
        """Calculate relevance score based on keyword overlap."""
        text_keywords = set(text.split())
        if not text_keywords:
            return 0.0

        overlap = len(intent_keywords.intersection(text_keywords))
        return overlap / len(intent_keywords) if intent_keywords else 0.0


class SuccessPatternDatabase:
    """Database of historical success patterns for prediction."""

    def __init__(self):
        """Initialize the success pattern database."""
        self.patterns: dict[str, dict[str, Any]] = {}
        self.pattern_frequencies: dict[str, int] = {}

    async def get_similar_patterns(
        self,
        intent: str,
        contextual_features: dict[str, Any]
    ) -> dict[str, Any]:
        """Get similar patterns from the database."""
        intent_signature = self._generate_intent_signature(intent)
        context_signature = self._generate_context_signature(contextual_features)

        # Find similar patterns
        similar_patterns = []
        for pattern_data in self.patterns.values():
            similarity = self._calculate_similarity(
                intent_signature,
                context_signature,
                pattern_data
            )
            if similarity > 0.6:  # Threshold for similarity
                similar_patterns.append((similarity, pattern_data))

        if not similar_patterns:
            # Return default pattern if no similar ones found
            return {
                "average_success_rate": 0.6,
                "strategy_confidence": 0.5,
                "top_strategies": ["adaptive", "simple_search"]
            }

        # Calculate weighted averages
        total_weight = sum(similarity for similarity, _ in similar_patterns)
        weighted_success_rate = sum(
            similarity * pattern["success_rate"]
            for similarity, pattern in similar_patterns
        ) / total_weight

        # Get most common strategies
        strategy_counts = {}
        for _, pattern in similar_patterns:
            for strategy in pattern.get("successful_strategies", []):
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        top_strategies = sorted(
            strategy_counts.keys(),
            key=lambda x: strategy_counts[x],
            reverse=True
        )[:3]

        return {
            "average_success_rate": weighted_success_rate,
            "strategy_confidence": min(len(similar_patterns) / 10.0, 1.0),
            "top_strategies": top_strategies
        }

    async def store_pattern(
        self,
        intent: str,
        context: dict[str, Any],
        *,
        success: bool,
        strategy: str
    ) -> None:
        """Store a new success pattern."""
        pattern_id = self._generate_pattern_id(intent, context)

        if pattern_id in self.patterns:
            # Update existing pattern
            pattern = self.patterns[pattern_id]
            pattern["frequency"] += 1

            # Update success rate with running average
            old_rate = pattern["success_rate"]
            new_success = 1.0 if success else 0.0
            pattern["success_rate"] = (old_rate * 0.9) + (new_success * 0.1)

            # Update strategy success
            if success and strategy and strategy not in pattern["successful_strategies"]:
                pattern["successful_strategies"].append(strategy)
        else:
            # Create new pattern
            self.patterns[pattern_id] = {
                "intent_signature": self._generate_intent_signature(intent),
                "context_signature": self._generate_context_signature(context),
                "frequency": 1,
                "success_rate": 1.0 if success else 0.0,
                "successful_strategies": [strategy] if success and strategy else [],
                "created_at": datetime.now(UTC)
            }

    def _generate_intent_signature(self, intent: str) -> str:
        """Generate a signature for the intent."""
        # Extract key characteristics
        intent_lower = intent.lower()

        # Identify intent type
        intent_type = "other"
        if any(word in intent_lower for word in ["search", "find", "look"]):
            intent_type = "search"
        elif any(word in intent_lower for word in ["analyze", "review"]):
            intent_type = "analysis"
        elif any(word in intent_lower for word in ["explain", "describe"]):
            intent_type = "explanation"

        # Extract complexity indicators
        complexity = "simple"
        if len(intent.split()) > 10:
            complexity = "complex"
        elif len(intent.split()) > 5:
            complexity = "medium"

        return f"{intent_type}:{complexity}"

    def _generate_context_signature(self, context: dict[str, Any]) -> str:
        """Generate a signature for the context."""
        if not context:
            return "no_context"

        # Count context elements
        element_count = len(context)

        # Assess context richness
        richness = "sparse"
        total_content = sum(len(str(v)) for v in context.values())
        if total_content > 200:
            richness = "rich"
        elif total_content > 50:
            richness = "moderate"

        return f"elements:{element_count}:richness:{richness}"

    def _generate_pattern_id(self, intent: str, context: dict[str, Any]) -> str:
        """Generate unique pattern ID."""
        intent_sig = self._generate_intent_signature(intent)
        context_sig = self._generate_context_signature(context)
        pattern = f"{intent_sig}|{context_sig}"

        return hashlib.sha256(pattern.encode()).hexdigest()[:16]

    def _calculate_similarity(
        self,
        intent_sig: str,
        context_sig: str,
        pattern_data: dict[str, Any]
    ) -> float:
        """Calculate similarity between current request and stored pattern."""
        # Intent similarity
        intent_similarity = 1.0 if intent_sig == pattern_data["intent_signature"] else 0.5

        # Context similarity
        context_similarity = 1.0 if context_sig == pattern_data["context_signature"] else 0.3

        # Frequency weighting (more frequent patterns are more reliable)
        frequency_weight = min(pattern_data["frequency"] / 10.0, 1.0)

        return (intent_similarity * 0.5 + context_similarity * 0.3 + frequency_weight * 0.2)


class ContextAdequacyOptimizationProvider(BaseServiceProvider, ZeroShotOptimizationService):
    """Zero-shot optimization service provider for context adequacy optimization."""

    def __init__(
        self,
        service_type: ServiceType,
        config: ZeroShotOptimizationServiceConfig,
        logger: logging.Logger | None = None,
    ):
        """Initialize the context adequacy optimization provider."""
        super().__init__(service_type, config, logger)
        self._config: ZeroShotOptimizationServiceConfig = config

        # Core components
        self.context_adequacy_predictor = ContextAdequacyPredictor()
        self.success_pattern_db = SuccessPatternDatabase()

        # Optimization state
        self._optimization_cache: dict[str, OptimizedIntentPlan] = {}
        self._success_metrics: dict[str, float] = {}

    async def _initialize_provider(self) -> None:
        """Initialize the context adequacy optimization provider."""
        self._logger.info(
            "Context adequacy optimization provider initialized with config: %s",
            {
                "success_threshold": self._config.success_threshold,
                "optimization_aggressiveness": self._config.optimization_aggressiveness,
                "enable_success_prediction": self._config.enable_success_prediction,
            }
        )

    async def _shutdown_provider(self) -> None:
        """Shutdown the context adequacy optimization provider."""
        self._logger.info(
            "Context adequacy optimization provider shutdown. Cached %d optimization plans.",
            len(self._optimization_cache)
        )

    async def _check_health(self) -> bool:
        """Check if the zero-shot optimization service is healthy."""
        # Service is healthy if we can make predictions and optimizations
        return len(self._optimization_cache) <= 10000  # Prevent memory issues

    async def optimize_for_zero_shot_success(
        self,
        ctx: Any,
        intent: str,
        available_context: dict[str, Any],
    ) -> OptimizedIntentPlan:
        """Optimize intent processing for first-attempt success."""
        # Check cache first
        cache_key = self._generate_cache_key(intent, available_context)
        if cache_key in self._optimization_cache:
            cached_plan = self._optimization_cache[cache_key]
            # Return cached plan if it's recent (within 1 hour)
            if (datetime.now(UTC) - cached_plan.created_at).total_seconds() < 3600:
                return cached_plan

        # Predict zero-shot success probability
        success_prediction = await self._predict_zero_shot_success(
            ctx, intent, available_context
        )

        optimization_plan = OptimizedIntentPlan(
            original_intent=intent,
            success_probability=success_prediction.probability,
            optimizations=[],
            enhanced_context=available_context.copy(),
            optimization_metadata={"prediction": success_prediction.dict()}
        )

        # Apply optimizations if success probability is low
        if success_prediction.probability < self._config.success_threshold:
            await self._apply_optimizations(
                optimization_plan,
                success_prediction,
                available_context
            )

        # Cache the optimization plan
        self._optimization_cache[cache_key] = optimization_plan

        return optimization_plan

    async def predict_success_probability(
        self,
        intent: str,
        context: dict[str, Any]
    ) -> SuccessPrediction:
        """Predict likelihood of zero-shot success."""
        return await self._predict_zero_shot_success(None, intent, context)

    async def _predict_zero_shot_success(
        self,
        ctx: Any,
        intent: str,
        context: dict[str, Any],
    ) -> SuccessPrediction:
        """Predict likelihood of zero-shot success."""
        # Extract contextual features
        contextual_features = await self._extract_contextual_features(ctx, intent, context)

        # Assess context adequacy
        context_adequacy = await self.context_adequacy_predictor.assess(intent, context)

        # Get historical success patterns
        historical_patterns = await self.success_pattern_db.get_similar_patterns(
            intent, contextual_features
        )

        # Calculate composite success probability
        success_probability = self._calculate_success_probability(
            context_adequacy=context_adequacy,
            historical_success=historical_patterns["average_success_rate"],
            intent_clarity=contextual_features.get("clarity_score", 0.7),
            contextual_richness=contextual_features.get("richness_score", 0.5)
        )

        # Identify missing context elements
        missing_context = self._identify_missing_context(intent, context)

        # Get recommended strategies
        recommended_strategies = historical_patterns["top_strategies"]

        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            intent, context, success_probability, contextual_features
        )

        return SuccessPrediction(
            probability=success_probability,
            context_adequacy=context_adequacy,
            strategy_confidence=historical_patterns["strategy_confidence"],
            missing_context=missing_context,
            recommended_strategies=recommended_strategies,
            risk_factors=risk_factors
        )

    async def _apply_optimizations(
        self,
        plan: OptimizedIntentPlan,
        prediction: SuccessPrediction,
        available_context: dict[str, Any]
    ) -> None:
        """Apply optimizations to improve success probability."""
        optimization_count = 0
        max_optimizations = self._config.max_optimization_attempts

        # Context enrichment optimization
        if (self._config.enable_context_enrichment and
            prediction.context_adequacy < 0.7 and
            optimization_count < max_optimizations):

            context_optimizations = await self._suggest_context_enrichment(
                plan.original_intent,
                available_context,
                prediction.missing_context
            )
            plan.optimizations.extend(context_optimizations)
            optimization_count += len(context_optimizations)

        # Strategy selection optimization
        if (self._config.enable_strategy_optimization and
            prediction.strategy_confidence < 0.8 and
            optimization_count < max_optimizations):

            strategy_optimizations = await self._suggest_strategy_improvements(
                plan.original_intent,
                prediction.recommended_strategies
            )
            plan.optimizations.extend(strategy_optimizations)
            optimization_count += len(strategy_optimizations)

        # Apply optimizations to enhanced context
        for optimization in plan.optimizations:
            if optimization.strategy_type == "context_enrichment":
                plan.enhanced_context.update(optimization.metadata.get("context_additions", {}))
            elif optimization.strategy_type == "strategy_selection":
                plan.recommended_strategy = optimization.metadata.get("recommended_strategy")
                plan.fallback_strategies = optimization.metadata.get("fallback_strategies", [])

    async def _suggest_context_enrichment(
        self,
        intent: str,
        context: dict[str, Any],
        missing_elements: list[str]
    ) -> list[OptimizationStrategy]:
        """Suggest context enrichment optimizations."""
        optimizations = []

        for element in missing_elements[:3]:  # Limit to top 3
            optimization = OptimizationStrategy(
                strategy_type="context_enrichment",
                description=f"Add missing context element: {element}",
                priority=1,
                expected_improvement=0.2,  # Estimate 20% improvement per element
                implementation_cost=0.1,   # Low cost for context additions
                metadata={
                    "missing_element": element,
                    "context_additions": self._generate_context_suggestions(element, intent)
                }
            )
            optimizations.append(optimization)

        return optimizations

    async def _suggest_strategy_improvements(
        self,
        intent: str,
        recommended_strategies: list[str]
    ) -> list[OptimizationStrategy]:
        """Suggest strategy selection optimizations."""
        optimizations = []

        if recommended_strategies:
            primary_strategy = recommended_strategies[0]
            fallback_strategies = recommended_strategies[1:3]  # Up to 2 fallbacks

            optimization = OptimizationStrategy(
                strategy_type="strategy_selection",
                description=f"Use recommended strategy: {primary_strategy}",
                priority=1,
                expected_improvement=0.3,  # Estimate 30% improvement
                implementation_cost=0.0,   # No cost for strategy selection
                metadata={
                    "recommended_strategy": primary_strategy,
                    "fallback_strategies": fallback_strategies
                }
            )
            optimizations.append(optimization)

        return optimizations

    def _generate_context_suggestions(self, element: str, intent: str) -> dict[str, Any]:
        """Generate suggested context additions for missing elements."""
        element_suggestions = {
            "scope": {"scope": "project", "search_depth": "recursive"},
            "query": {"search_terms": self._extract_search_terms(intent)},
            "target": {"target_type": "function", "include_related": True},
            "focus": {"analysis_focus": "implementation", "detail_level": "detailed"},
            "filters": {"file_types": ["py", "js", "ts"], "exclude_tests": False}
        }

        return element_suggestions.get(element, {})

    def _extract_search_terms(self, intent: str) -> list[str]:
        """Extract potential search terms from intent."""
        import re

        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}

        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', intent.lower())
        search_terms = [word for word in words if word not in stop_words]

        return search_terms[:5]  # Limit to 5 terms

    async def _extract_contextual_features(
        self,
        ctx: Any,
        intent: str,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract contextual features for success prediction."""
        features = {"clarity_score": self._assess_intent_clarity(intent)}

        features["complexity_score"] = self._assess_intent_complexity(intent)

        # Context analysis
        features["richness_score"] = self._assess_context_richness(context)
        features["completeness_score"] = await self.context_adequacy_predictor.assess(intent, context)

        # Session features (if available)
        if ctx:
            features["session_features"] = self._extract_session_features(ctx)

        return features

    def _assess_intent_clarity(self, intent: str) -> float:
        """Assess clarity of the intent."""
        # Length appropriateness
        word_count = len(intent.split())
        if 3 <= word_count <= 20:
            length_score = 1.0
        elif 1 <= word_count <= 30:
            length_score = 0.8
        else:
            length_score = 0.5
        # Presence of action words
        action_words = ["find", "search", "get", "show", "analyze", "explain", "create", "build"]
        has_action = any(word in intent.lower() for word in action_words)
        # Specificity indicators
        specific_terms = ["function", "class", "file", "method", "variable", "component"]
        specificity_score = min(
            (sum(term in intent.lower() for term in specific_terms) / 3.0), 1.0
        )
        clarity_factors = [length_score, 1.0 if has_action else 0.6, specificity_score]
        return sum(clarity_factors) / len(clarity_factors)

    def _assess_intent_complexity(self, intent: str) -> float:
        """Assess complexity of the intent."""
        # Compound operations
        conjunctions = ["and", "or", "then", "also", "but", "however"]
        conjunction_count = sum(conj in intent.lower() for conj in conjunctions)
        complexity_factors = [min(conjunction_count / 2.0, 1.0)]
        # Technical terminology
        tech_terms = ["algorithm", "pattern", "architecture", "framework", "api", "interface"]
        tech_count = sum(term in intent.lower() for term in tech_terms)
        complexity_factors.append(min(tech_count / 3.0, 1.0))

        # Question depth
        question_words = ["what", "how", "why", "where", "when", "which"]
        has_question = any(word in intent.lower() for word in question_words)
        complexity_factors.append(0.8 if has_question else 0.5)

        return sum(complexity_factors) / len(complexity_factors)

    def _assess_context_richness(self, context: dict[str, Any]) -> float:
        """Assess richness of the provided context."""
        if not context:
            return 0.1

        # Element count
        element_count = len(context)
        element_score = min(element_count / 5.0, 1.0)
        # Content depth
        total_content = sum(len(str(value)) for value in context.values())
        depth_score = min(total_content / 300.0, 1.0)
        richness_factors = [element_score, depth_score]
        # Structure variety
        value_types = {type(value).__name__ for value in context.values()}
        variety_score = min(len(value_types) / 3.0, 1.0)
        richness_factors.append(variety_score)

        return sum(richness_factors) / len(richness_factors)

    def _extract_session_features(self, ctx: Any) -> dict[str, Any]:
        """Extract session-related features from context."""
        features = {}

        # User agent information
        if hasattr(ctx, "user_agent"):
            features["user_agent_type"] = self._classify_user_agent(ctx.user_agent)

        # Timing information
        if hasattr(ctx, "start_time"):
            features["request_start_time"] = ctx.start_time

        # Request characteristics
        if hasattr(ctx, "request") and ctx.request:
            features["has_http_context"] = True
            if hasattr(ctx.request, "headers"):
                features["header_count"] = len(ctx.request.headers)

        return features

    def _classify_user_agent(self, user_agent: str) -> str:
        """Classify user agent type."""
        ua_lower = user_agent.lower()

        if "curl" in ua_lower or "wget" in ua_lower:
            return "cli_client"
        if "python" in ua_lower:
            return "python_client"
        if "javascript" in ua_lower or "node" in ua_lower:
            return "js_client"
        return "browser_client" if "mozilla" in ua_lower else "unknown_client"

    def _calculate_success_probability(
        self,
        context_adequacy: float,
        historical_success: float,
        intent_clarity: float,
        contextual_richness: float
    ) -> float:
        """Calculate composite success probability."""
        # Weighted combination based on configuration
        weights = {
            "context_adequacy": self._config.context_adequacy_weight,
            "historical_success": self._config.historical_success_weight,
            "intent_clarity": 0.2,  # Remaining weight split
            "contextual_richness": 0.2
        }

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        success_probability = (
            context_adequacy * normalized_weights["context_adequacy"] +
            historical_success * normalized_weights["historical_success"] +
            intent_clarity * normalized_weights["intent_clarity"] +
            contextual_richness * normalized_weights["contextual_richness"]
        )

        return min(max(success_probability, 0.0), 1.0)

    def _identify_missing_context(self, intent: str, context: dict[str, Any]) -> list[str]:
        """Identify missing context elements."""
        required_elements = self.context_adequacy_predictor._get_required_elements(intent)
        missing_elements = []

        missing_elements.extend(
            element
            for element in required_elements
            if not self.context_adequacy_predictor._has_context_element(
                element, context
            )
        )
        return missing_elements

    def _identify_risk_factors(
        self,
        intent: str,
        context: dict[str, Any],
        success_probability: float,
        features: dict[str, Any]
    ) -> list[str]:
        """Identify risk factors that might reduce success."""
        risk_factors = []

        # Low success probability
        if success_probability < 0.5:
            risk_factors.append("low_predicted_success")

        # Unclear intent
        if features.get("clarity_score", 1.0) < 0.6:
            risk_factors.append("unclear_intent")

        # Insufficient context
        if features.get("richness_score", 1.0) < 0.4:
            risk_factors.append("insufficient_context")

        # High complexity
        if features.get("complexity_score", 0.0) > 0.8:
            risk_factors.append("high_complexity")

        # Missing required elements
        missing_elements = self._identify_missing_context(intent, context)
        if len(missing_elements) > 2:
            risk_factors.append("missing_critical_context")

        return risk_factors

    def _generate_cache_key(self, intent: str, context: dict[str, Any]) -> str:
        """Generate cache key for optimization plans."""
        intent_hash = hashlib.sha256(intent.encode()).hexdigest()[:16]
        joined_context = "|".join(f"{k}:{v!s}" for k, v in sorted(context.items()))
        context_hash = hashlib.sha256(joined_context.encode()).hexdigest()[:16]

        return f"{intent_hash}:{context_hash}"
