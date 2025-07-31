<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Intent Layer Phase 2: Enhanced Features Implementation Plan (Updated)

## 🎯 Overview

**Phase Duration**: 3-4 weeks  
**Priority**: SHOULD HAVE for production readiness  
**Prerequisite**: Phase 1 complete with all services operational

This updated phase restructures SpaCy as a **Provider** following CodeWeaver's architecture patterns, leverages SpaCy 3.7+ features, and maintains full configurability and extensibility for future enhancements including custom training.

## 🏗️ Key Architectural Changes

### 1. SpaCy as Provider Architecture
- **SpaCy Provider**: Located in `src/codeweaver/providers/nlp/` following provider patterns
- **Protocol-Based Interface**: Universal `NLPProvider` protocol for extensibility
- **Factory Integration**: Registered with provider factory system
- **Configuration-Driven**: Full TOML/environment variable configuration
- **Multiple Implementations**: Support for different SpaCy models and custom training

### 2. SpaCy 3.7+ Integration
- **Modern Pipeline Architecture**: Uses `en_core_web_trf` with fallback to `en_core_web_sm`
- **TextCategorizer**: Built-in intent classification via `doc.cats`
- **EntityRuler**: Domain-specific entity recognition with proper pattern definitions
- **Transformer Support**: Leverages `doc.tensor` and `doc._.trf_data` for embeddings
- **Curated Transformers**: Optional integration with `spacy-curated-transformers`

### 3. Enhanced Extensibility
- **Custom Training Support**: Framework for training custom models
- **Dynamic Configuration**: Runtime model switching and configuration updates
- **Multi-Language Ready**: Architecture supports future language expansion
- **Plugin System**: Custom components and pipelines via factory registration

## 📊 Weekly Breakdown

### Week 1: NLP Provider Architecture

#### Deliverables

**1. NLP Provider Protocol** (`src/codeweaver/types/provider_nlp.py`)
```python
from typing import Protocol, Any, runtime_checkable
from dataclasses import dataclass

@dataclass
class NLPResult:
    """Result from NLP processing."""
    intent_type: str | None
    confidence: float
    entities: list[dict[str, Any]]
    primary_target: str | None
    metadata: dict[str, Any]
    embeddings: list[float] | None = None

@dataclass
class NLPModelInfo:
    """Information about available NLP models."""
    name: str
    language: str
    capabilities: list[str]  # e.g., ["intent_classification", "entity_recognition", "embeddings"]
    model_size: str  # "sm", "md", "lg", "trf"
    requires_download: bool

@runtime_checkable
class NLPProvider(Protocol):
    """Protocol for natural language processing providers."""
    
    async def initialize(self) -> None:
        """Initialize the NLP provider."""
        ...
    
    async def process_text(self, text: str, context: dict[str, Any] | None = None) -> NLPResult:
        """Process text and extract intent, entities, and embeddings."""
        ...
    
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings."""
        ...
    
    async def classify_intent(self, text: str) -> tuple[str, float]:
        """Classify intent with confidence score."""
        ...
    
    async def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities."""
        ...
    
    async def health_check(self) -> dict[str, Any]:
        """Check provider health and model availability."""
        ...
    
    def get_available_models(self) -> list[NLPModelInfo]:
        """Get list of available models."""
        ...
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch to different model at runtime."""
        ...
```

**2. SpaCy Provider Implementation** (`src/codeweaver/providers/nlp/spacy_provider.py`)
```python
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler, TextCategorizer
from typing import Any
import logging

from codeweaver.types import NLPProvider, NLPResult, NLPModelInfo
from codeweaver.providers.nlp.domain_patterns import CodeDomainPatterns

logger = logging.getLogger(__name__)

class SpaCyProvider:
    """SpaCy-based NLP provider with 3.7+ features."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.nlp: Language | None = None
        self.domain_patterns = CodeDomainPatterns()
        self.model_name = config.get("model", "en_core_web_sm")
        self.use_transformers = config.get("use_transformers", False)
        self.enable_intent_classification = config.get("enable_intent_classification", True)
        self.custom_components = config.get("custom_components", [])
        
    async def initialize(self) -> None:
        """Initialize SpaCy pipeline with 3.7+ features."""
        try:
            # Load model with transformer support
            if self.use_transformers:
                try:
                    # Try curated transformers first
                    import spacy_curated_transformers
                    self.nlp = spacy.load("en_core_web_trf")
                    logger.info("Loaded transformer-based SpaCy model")
                except (ImportError, OSError):
                    logger.warning("Transformers not available, falling back to standard model")
                    self.nlp = spacy.load(self.model_name)
            else:
                self.nlp = spacy.load(self.model_name)
                
            # Add domain-specific entity ruler
            if "entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
                self.domain_patterns.add_patterns_to_ruler(ruler)
                logger.info("Added domain-specific entity patterns")
            
            # Add text categorizer for intent classification
            if self.enable_intent_classification and "textcat" not in self.nlp.pipe_names:
                textcat = self.nlp.add_pipe("textcat", config={
                    "exclusive_classes": True,
                    "architecture": "simple_cnn"  # Could be "ensemble" for transformers
                })
                # Add intent labels
                for intent in self.config.get("intent_labels", ["SEARCH", "DOCUMENTATION", "ANALYSIS"]):
                    textcat.add_label(intent)
                logger.info("Added text categorizer for intent classification")
            
            # Add custom components
            for component_config in self.custom_components:
                self._add_custom_component(component_config)
                
        except OSError as e:
            logger.error(f"Failed to load SpaCy model {self.model_name}: {e}")
            raise RuntimeError(f"SpaCy model initialization failed: {e}")
    
    async def process_text(self, text: str, context: dict[str, Any] | None = None) -> NLPResult:
        """Process text using SpaCy 3.7+ pipeline."""
        if not self.nlp:
            raise RuntimeError("SpaCy provider not initialized")
            
        doc = self.nlp(text)
        
        # Extract intent using TextCategorizer
        intent_type = None
        confidence = 0.0
        if doc.cats:
            intent_type = max(doc.cats, key=doc.cats.get)
            confidence = doc.cats[intent_type]
        
        # Extract entities
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, "_", {}).get("confidence", 0.95)
            }
            for ent in doc.ents
        ]
        
        # Find primary target from entities
        primary_target = self._extract_primary_target(entities)
        
        # Get embeddings if available
        embeddings = None
        if hasattr(doc, 'tensor') and doc.tensor is not None:
            embeddings = doc.tensor.tolist()
        elif hasattr(doc, 'vector') and doc.vector is not None:
            embeddings = doc.vector.tolist()
        
        return NLPResult(
            intent_type=intent_type,
            confidence=confidence,
            entities=entities,
            primary_target=primary_target,
            embeddings=embeddings,
            metadata={
                "model": self.nlp.meta["name"],
                "pipeline": list(self.nlp.pipe_names),
                "has_transformer": "transformer" in self.nlp.pipe_names,
                "tokens": len(doc),
                "pos_tags": [(token.text, token.pos_) for token in doc[:10]]  # First 10 for debugging
            }
        )
    
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings using SpaCy pipeline."""
        if not self.nlp:
            raise RuntimeError("SpaCy provider not initialized")
            
        embeddings = []
        for doc in self.nlp.pipe(texts):
            if hasattr(doc, 'tensor') and doc.tensor is not None:
                embeddings.append(doc.tensor.tolist())
            elif hasattr(doc, 'vector') and doc.vector is not None:
                embeddings.append(doc.vector.tolist())
            else:
                # Fallback to zero embedding
                embeddings.append([0.0] * 300)  # Default spaCy vector size
                
        return embeddings
    
    def get_available_models(self) -> list[NLPModelInfo]:
        """Get available SpaCy models."""
        models = [
            NLPModelInfo(
                name="en_core_web_sm",
                language="en",
                capabilities=["entity_recognition", "pos_tagging", "dependency_parsing"],
                model_size="sm",
                requires_download=True
            ),
            NLPModelInfo(
                name="en_core_web_md", 
                language="en",
                capabilities=["entity_recognition", "pos_tagging", "dependency_parsing", "word_vectors"],
                model_size="md",
                requires_download=True
            ),
            NLPModelInfo(
                name="en_core_web_lg",
                language="en", 
                capabilities=["entity_recognition", "pos_tagging", "dependency_parsing", "word_vectors"],
                model_size="lg",
                requires_download=True
            ),
            NLPModelInfo(
                name="en_core_web_trf",
                language="en",
                capabilities=["entity_recognition", "pos_tagging", "dependency_parsing", "transformers", "embeddings"],
                model_size="trf",
                requires_download=True
            )
        ]
        
        return models
    
    async def switch_model(self, model_name: str) -> bool:
        """Switch SpaCy model at runtime."""
        try:
            old_config = dict(self.config)
            self.model_name = model_name
            self.config["model"] = model_name
            
            # Reinitialize with new model
            await self.initialize()
            
            logger.info(f"Successfully switched to model: {model_name}")
            return True
            
        except Exception as e:
            # Restore previous config
            self.config = old_config
            logger.error(f"Failed to switch to model {model_name}: {e}")
            return False
    
    def _extract_primary_target(self, entities: list[dict[str, Any]]) -> str | None:
        """Extract primary target from entities."""
        # Priority order for entity types
        priority_labels = ["CODE_ELEMENT", "LANGUAGE", "FRAMEWORK", "OPERATION", "DATABASE"]
        
        for label in priority_labels:
            for entity in entities:
                if entity["label"] == label:
                    return entity["text"]
        
        # Return first entity if no priority match
        return entities[0]["text"] if entities else None
    
    def _add_custom_component(self, component_config: dict[str, Any]) -> None:
        """Add custom component to pipeline."""
        component_name = component_config.get("name")
        component_factory = component_config.get("factory")
        component_config_dict = component_config.get("config", {})
        
        if component_name and component_factory:
            self.nlp.add_pipe(component_factory, name=component_name, config=component_config_dict)
            logger.info(f"Added custom component: {component_name}")
    
    async def health_check(self) -> dict[str, Any]:
        """Check provider health and model availability."""
        return {
            "provider_name": "spacy",
            "model_loaded": self.nlp is not None,
            "model_name": self.nlp.meta["name"] if self.nlp else "none",
            "capabilities": [
                "intent_classification" if self.enable_intent_classification else None,
                "entity_recognition",
                "embeddings" if self.use_transformers else "word_vectors",
                "pos_tagging",
                "dependency_parsing"
            ],
            "memory_usage_mb": self._get_memory_usage(),
            "pipeline": list(self.nlp.pipe_names) if self.nlp else []
        }
    
    def _get_memory_usage(self) -> float:
        """Get approximate memory usage."""
        # This is a simplified implementation
        # In production, you might want to use more sophisticated memory tracking
        return 0.0
```

**3. Domain Pattern Management** (`src/codeweaver/providers/nlp/domain_patterns.py`)
```python
from typing import Any
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CodeDomainPatterns:
    """Manages domain-specific patterns for code understanding."""
    
    def __init__(self, custom_patterns_path: str | None = None):
        self.custom_patterns_path = custom_patterns_path
        self.base_patterns = self._load_base_patterns()
        self.custom_patterns = self._load_custom_patterns() if custom_patterns_path else []
    
    def _load_base_patterns(self) -> list[dict[str, Any]]:
        """Load base code domain patterns."""
        return [
            # Programming languages
            {"label": "LANGUAGE", "pattern": [{"LOWER": {"IN": [
                "python", "javascript", "typescript", "java", "go", "rust", 
                "c++", "csharp", "ruby", "php", "swift", "kotlin", "scala"
            ]}}]},
            
            # Code elements
            {"label": "CODE_ELEMENT", "pattern": [{"LOWER": {"IN": [
                "function", "class", "method", "variable", "constant", "interface",
                "module", "package", "component", "service", "endpoint", "route"
            ]}}]},
            
            # Frameworks and libraries
            {"label": "FRAMEWORK", "pattern": [{"LOWER": {"IN": [
                "react", "vue", "angular", "django", "flask", "express", "spring",
                "fastapi", "nextjs", "nuxt", "laravel", "rails", "dotnet"
            ]}}]},
            
            # Database technologies
            {"label": "DATABASE", "pattern": [{"LOWER": {"IN": [
                "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                "sqlite", "oracle", "cassandra", "dynamodb", "firestore"
            ]}}]},
            
            # Operations and concepts
            {"label": "OPERATION", "pattern": [{"LOWER": {"IN": [
                "authentication", "authorization", "validation", "logging",
                "caching", "monitoring", "deployment", "testing", "debugging",
                "optimization", "refactoring", "migration", "backup"
            ]}}]},
            
            # File types and formats
            {"label": "FILE_TYPE", "pattern": [{"LOWER": {"IN": [
                "json", "yaml", "xml", "csv", "sql", "dockerfile", "makefile",
                "config", "env", "properties", "manifest", "schema"
            ]}}]},
            
            # Development tools
            {"label": "TOOL", "pattern": [{"LOWER": {"IN": [
                "git", "docker", "kubernetes", "jenkins", "github", "gitlab",
                "vscode", "intellij", "webpack", "babel", "eslint", "pytest"
            ]}}]}
        ]
    
    def _load_custom_patterns(self) -> list[dict[str, Any]]:
        """Load custom patterns from configuration file."""
        if not self.custom_patterns_path:
            return []
            
        try:
            path = Path(self.custom_patterns_path)
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load custom patterns: {e}")
        
        return []
    
    def add_patterns_to_ruler(self, ruler) -> None:
        """Add all patterns to SpaCy EntityRuler."""
        all_patterns = self.base_patterns + self.custom_patterns
        ruler.add_patterns(all_patterns)
        logger.info(f"Added {len(all_patterns)} patterns to EntityRuler")
    
    def add_custom_pattern(self, label: str, pattern: Any) -> None:
        """Add a custom pattern at runtime."""
        custom_pattern = {"label": label, "pattern": pattern}
        self.custom_patterns.append(custom_pattern)
        logger.info(f"Added custom pattern: {label}")
    
    def save_custom_patterns(self) -> None:
        """Save custom patterns to file."""
        if self.custom_patterns_path:
            path = Path(self.custom_patterns_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(self.custom_patterns, f, indent=2)
            logger.info(f"Saved {len(self.custom_patterns)} custom patterns")
```

#### Success Criteria - Week 1
- [ ] SpaCy Provider follows CodeWeaver provider protocols
- [ ] Modern SpaCy 3.7+ API integration functional
- [ ] Domain patterns system supports runtime customization
- [ ] Provider factory registration and configuration working

### Week 2: Intent Layer Integration with NLP Provider

#### Deliverables

**1. Enhanced Intent Parser** (`src/codeweaver/intent/parsing/nlp_enhanced_parser.py`)
```python
from codeweaver.types import ParsedIntent, IntentType, NLPProvider, NLPResult
from codeweaver.intent.parsing.pattern_matcher import PatternBasedParser
from codeweaver.intent.parsing.confidence_scorer import EnhancedConfidenceScorer
import logging

logger = logging.getLogger(__name__)

class NLPEnhancedParser:
    """Intent parser using NLP Provider architecture."""
    
    def __init__(self, nlp_provider: NLPProvider | None = None):
        self.nlp_provider = nlp_provider
        self.pattern_parser = PatternBasedParser()  # Fallback
        self.confidence_scorer = EnhancedConfidenceScorer()
        self.confidence_threshold = 0.7
        
    async def parse(self, intent_text: str) -> ParsedIntent:
        """Parse intent with NLP provider and pattern fallback."""
        # Try NLP provider first if available
        if self.nlp_provider:
            try:
                nlp_result = await self.nlp_provider.process_text(intent_text)
                
                if nlp_result.confidence > self.confidence_threshold:
                    return self._convert_nlp_result_to_parsed_intent(nlp_result, intent_text)
                    
            except Exception as e:
                logger.warning(f"NLP provider failed, using fallback: {e}")
        
        # Fallback to pattern matching
        pattern_result = await self.pattern_parser.parse(intent_text)
        
        # Enhance pattern result with NLP insights if available
        if self.nlp_provider:
            enhanced_result = await self._enhance_pattern_result_with_nlp(
                pattern_result, intent_text
            )
            return enhanced_result
            
        return pattern_result
    
    def _convert_nlp_result_to_parsed_intent(
        self, 
        nlp_result: NLPResult, 
        original_text: str
    ) -> ParsedIntent:
        """Convert NLP provider result to ParsedIntent."""
        # Map NLP intent types to internal intent types
        intent_type_mapping = {
            "SEARCH": IntentType.SEARCH,
            "DOCUMENTATION": IntentType.DOCUMENTATION,
            "ANALYSIS": IntentType.ANALYSIS,
            # Add more mappings as needed
        }
        
        intent_type = intent_type_mapping.get(
            nlp_result.intent_type, 
            IntentType.SEARCH  # Default fallback
        )
        
        # Extract filters from entities
        filters = self._extract_filters_from_entities(nlp_result.entities)
        
        # Assess scope and complexity from entities and text
        scope = self._assess_scope_from_nlp(nlp_result)
        complexity = self._assess_complexity_from_nlp(nlp_result)
        
        return ParsedIntent(
            intent_type=intent_type,
            primary_target=nlp_result.primary_target or self._extract_target_fallback(original_text),
            scope=scope,
            complexity=complexity,
            confidence=nlp_result.confidence,
            filters=filters,
            metadata={
                "parser": "nlp_enhanced",
                "nlp_provider": "spacy",
                "entities": nlp_result.entities,
                "embeddings_available": nlp_result.embeddings is not None,
                **nlp_result.metadata
            }
        )
    
    async def _enhance_pattern_result_with_nlp(
        self, 
        pattern_result: ParsedIntent, 
        intent_text: str
    ) -> ParsedIntent:
        """Enhance pattern matching result with NLP insights."""
        try:
            nlp_result = await self.nlp_provider.process_text(intent_text)
            
            # Enhance with NLP entities if pattern result is weak
            if pattern_result.confidence < 0.8 and nlp_result.entities:
                enhanced_filters = pattern_result.filters.copy()
                enhanced_filters.update(self._extract_filters_from_entities(nlp_result.entities))
                
                # Update metadata with NLP insights
                enhanced_metadata = pattern_result.metadata.copy()
                enhanced_metadata["nlp_enhancement"] = True
                enhanced_metadata["nlp_entities"] = nlp_result.entities
                
                return ParsedIntent(
                    intent_type=pattern_result.intent_type,
                    primary_target=pattern_result.primary_target,
                    scope=pattern_result.scope,
                    complexity=pattern_result.complexity,
                    confidence=min(pattern_result.confidence + 0.1, 1.0),  # Slight boost
                    filters=enhanced_filters,
                    metadata=enhanced_metadata
                )
                
        except Exception as e:
            logger.warning(f"NLP enhancement failed: {e}")
        
        return pattern_result
    
    def _extract_filters_from_entities(self, entities: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract search filters from NLP entities."""
        filters = {}
        
        for entity in entities:
            label = entity["label"]
            text = entity["text"].lower()
            
            if label == "LANGUAGE":
                filters["language"] = text
            elif label == "FRAMEWORK":
                filters["framework"] = text
            elif label == "FILE_TYPE":
                filters["file_type"] = text
            elif label == "OPERATION":
                filters["operation"] = text
                
        return filters
    
    def _assess_scope_from_nlp(self, nlp_result: NLPResult) -> str:
        """Assess intent scope from NLP result."""
        # Simple heuristic based on entities
        entity_count = len(nlp_result.entities)
        
        if entity_count >= 3:
            return "project"
        elif entity_count >= 2:
            return "module"
        else:
            return "file"
    
    def _assess_complexity_from_nlp(self, nlp_result: NLPResult) -> str:
        """Assess intent complexity from NLP result."""
        # Simple heuristic based on entities and confidence
        entity_count = len(nlp_result.entities)
        has_multiple_types = len(set(e["label"] for e in nlp_result.entities)) > 2
        
        if entity_count >= 4 or has_multiple_types:
            return "complex"
        elif entity_count >= 2:
            return "moderate"
        else:
            return "simple"
    
    def _extract_target_fallback(self, text: str) -> str:
        """Extract target using simple fallback method."""
        # Simple keyword extraction as fallback
        words = text.lower().split()
        
        # Look for common target patterns
        target_words = [word for word in words if len(word) > 3 and word.isalpha()]
        
        return target_words[0] if target_words else "unknown"
```

**2. NLP Provider Factory Integration** (`src/codeweaver/factories/nlp_provider_registry.py`)
```python
from typing import Dict, Type, Any
from codeweaver.types import NLPProvider, ComponentInfo
from codeweaver.providers.nlp.spacy_provider import SpaCyProvider
import logging

logger = logging.getLogger(__name__)

class NLPProviderRegistry:
    """Registry for NLP providers following CodeWeaver patterns."""
    
    def __init__(self):
        self._providers: Dict[str, Type[NLPProvider]] = {}
        self._component_info: Dict[str, ComponentInfo] = {}
        self._register_builtin_providers()
    
    def _register_builtin_providers(self) -> None:
        """Register built-in NLP providers."""
        self.register_provider(
            "spacy", 
            SpaCyProvider,
            ComponentInfo(
                name="SpaCy NLP Provider",
                description="SpaCy-based natural language processing with 3.7+ features",
                version="1.0.0",
                capabilities=[
                    "intent_classification",
                    "entity_recognition", 
                    "text_embeddings",
                    "pos_tagging",
                    "dependency_parsing",
                    "custom_training"
                ],
                config_schema={
                    "model": {"type": "string", "default": "en_core_web_sm"},
                    "use_transformers": {"type": "boolean", "default": False},
                    "enable_intent_classification": {"type": "boolean", "default": True},
                    "intent_labels": {"type": "array", "items": {"type": "string"}},
                    "custom_patterns_path": {"type": "string", "optional": True},
                    "custom_components": {"type": "array", "items": {"type": "object"}}
                }
            )
        )
    
    def register_provider(
        self, 
        name: str, 
        provider_class: Type[NLPProvider], 
        info: ComponentInfo
    ) -> None:
        """Register an NLP provider."""
        self._providers[name] = provider_class
        self._component_info[name] = info
        logger.info(f"Registered NLP provider: {name}")
    
    def create_provider(self, name: str, config: Dict[str, Any]) -> NLPProvider:
        """Create NLP provider instance."""
        if name not in self._providers:
            raise ValueError(f"Unknown NLP provider: {name}")
        
        provider_class = self._providers[name]
        return provider_class(config)
    
    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return list(self._providers.keys())
    
    def get_provider_info(self, name: str) -> ComponentInfo | None:
        """Get information about a specific provider."""
        return self._component_info.get(name)
    
    def get_all_provider_info(self) -> Dict[str, ComponentInfo]:
        """Get information about all registered providers."""
        return self._component_info.copy()
```

**3. Configuration Integration** (`src/codeweaver/config.py` - additions)
```python
# Add to existing ConfigManager class

@dataclass
class NLPConfig:
    """NLP provider configuration."""
    provider: str = "spacy"
    model: str = "en_core_web_sm"
    use_transformers: bool = False
    enable_intent_classification: bool = True
    intent_labels: list[str] = field(default_factory=lambda: ["SEARCH", "DOCUMENTATION", "ANALYSIS"])
    custom_patterns_path: str | None = None
    confidence_threshold: float = 0.7
    fallback_enabled: bool = True

@dataclass 
class IntentConfig:
    """Intent layer configuration."""
    nlp: NLPConfig = field(default_factory=NLPConfig)
    caching_enabled: bool = True
    semantic_caching_enabled: bool = True
    performance_optimization_enabled: bool = True
    
# Add to main Config class
intent: IntentConfig = field(default_factory=IntentConfig)
```

#### Success Criteria - Week 2
- [ ] Intent layer cleanly integrates with NLP Provider
- [ ] Factory registration and configuration functional
- [ ] Graceful fallback to pattern matching when NLP unavailable
- [ ] Provider switching at runtime works correctly

### Week 3: Semantic Caching with NLP Provider Integration

#### Deliverables

**1. NLP-Integrated Semantic Cache** (`src/codeweaver/intent/caching/nlp_semantic_cache.py`)
```python
from codeweaver.types import CacheService, VectorBackend, NLPProvider
from codeweaver.intent.caching.intent_cache import IntentCacheManager
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class NLPSemanticCache(IntentCacheManager):
    """Semantic caching using NLP Provider embeddings."""
    
    def __init__(
        self,
        cache_service: CacheService | None,
        vector_backend: VectorBackend | None,
        nlp_provider: NLPProvider | None
    ):
        super().__init__(cache_service)
        self.vector_backend = vector_backend
        self.nlp_provider = nlp_provider
        self.similarity_threshold = 0.85
        self.semantic_cache_enabled = (
            vector_backend is not None and nlp_provider is not None
        )
    
    async def get_cached_result(self, intent_text: str) -> IntentResult | None:
        """Get cached result using NLP provider embeddings."""
        # Try exact match first (fast path)
        exact_result = await super().get_cached_result(intent_text)
        if exact_result:
            return exact_result
        
        # Try semantic similarity using NLP provider
        if self.semantic_cache_enabled:
            return await self._get_nlp_semantic_cached_result(intent_text)
        
        return None
    
    async def _get_nlp_semantic_cached_result(self, intent_text: str) -> IntentResult | None:
        """Find semantically similar cached results using NLP provider."""
        try:
            # Get embeddings from NLP provider
            embeddings = await self.nlp_provider.get_embeddings([intent_text])
            if not embeddings or not embeddings[0]:
                return None
            
            query_embedding = embeddings[0]
            
            # Search for similar intents
            search_results = await self.vector_backend.search(
                query_embedding,
                limit=5,
                metadata_filter={"type": "intent_cache"}
            )
            
            # Check if any results meet similarity threshold
            for result in search_results:
                if result.score >= self.similarity_threshold:
                    cache_key = result.metadata.get("cache_key")
                    if cache_key:
                        cached_result = await self.cache_service.get(cache_key)
                        if cached_result:
                            cached_result.metadata["nlp_semantic_cache_hit"] = True
                            cached_result.metadata["similarity_score"] = result.score
                            cached_result.metadata["original_query"] = result.metadata.get("intent_text")
                            return cached_result
            
            return None
            
        except Exception as e:
            logger.warning(f"NLP semantic cache lookup failed: {e}")
            return None
    
    async def cache_result(
        self,
        intent_text: str,
        result: IntentResult,
        ttl: int = 3600
    ) -> None:
        """Cache result with NLP provider embeddings."""
        # Standard caching
        await super().cache_result(intent_text, result, ttl)
        
        # NLP semantic indexing if enabled
        if self.semantic_cache_enabled and result.success:
            await self._index_nlp_semantic_cache(intent_text, result)
    
    async def _index_nlp_semantic_cache(
        self,
        intent_text: str,
        result: IntentResult
    ) -> None:
        """Index intent using NLP provider embeddings."""
        try:
            # Get embeddings from NLP provider
            embeddings = await self.nlp_provider.get_embeddings([intent_text])
            if not embeddings or not embeddings[0]:
                return
            
            # Store in vector backend
            cache_key = self._generate_cache_key(intent_text)
            await self.vector_backend.upsert_points([{
                "id": f"intent_cache_{hash(intent_text)}",
                "vector": embeddings[0],
                "metadata": {
                    "type": "intent_cache",
                    "cache_key": cache_key,
                    "intent_text": intent_text,
                    "intent_type": result.metadata.get("intent_type"),
                    "nlp_provider": self.nlp_provider.__class__.__name__,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }])
            
            logger.debug(f"Indexed intent in semantic cache: {intent_text}")
            
        except Exception as e:
            logger.warning(f"NLP semantic cache indexing failed: {e}")
    
    async def get_cache_statistics(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        base_stats = await super().get_cache_statistics()
        
        if self.semantic_cache_enabled:
            # Add semantic cache specific statistics
            semantic_stats = await self._get_semantic_cache_stats()
            base_stats.update(semantic_stats)
        
        return base_stats
    
    async def _get_semantic_cache_stats(self) -> dict[str, Any]:
        """Get semantic cache specific statistics."""
        try:
            # Get vector backend statistics
            vector_stats = await self.vector_backend.get_collection_info()
            
            return {
                "semantic_cache_enabled": True,
                "vector_count": vector_stats.get("vectors_count", 0),
                "nlp_provider": self.nlp_provider.__class__.__name__,
                "similarity_threshold": self.similarity_threshold
            }
        except Exception as e:
            logger.warning(f"Failed to get semantic cache stats: {e}")
            return {"semantic_cache_enabled": True, "error": str(e)}
```

**2. Custom Training Framework** (`src/codeweaver/providers/nlp/training/`)

**Custom Trainer** (`src/codeweaver/providers/nlp/training/custom_trainer.py`)
```python
from pathlib import Path
import spacy
from spacy.training import Example
from typing import Any, List
import logging

logger = logging.getLogger(__name__)

class SpaCyCustomTrainer:
    """Framework for custom SpaCy model training."""
    
    def __init__(self, base_model: str = "en_core_web_sm"):
        self.base_model = base_model
        self.nlp = None
        
    async def prepare_training_data(
        self, 
        training_examples: list[dict[str, Any]]
    ) -> list[Example]:
        """Prepare training data in SpaCy format."""
        if not self.nlp:
            self.nlp = spacy.load(self.base_model)
            
        examples = []
        for example in training_examples:
            text = example["text"]
            annotations = example.get("annotations", {})
            
            doc = self.nlp.make_doc(text)
            example_obj = Example.from_dict(doc, annotations)
            examples.append(example_obj)
            
        return examples
    
    async def train_intent_classifier(
        self,
        training_examples: list[dict[str, Any]],
        output_path: str,
        iterations: int = 10
    ) -> None:
        """Train custom intent classifier."""
        if not self.nlp:
            self.nlp = spacy.load(self.base_model)
        
        # Add text categorizer if not present
        if "textcat" not in self.nlp.pipe_names:
            textcat = self.nlp.add_pipe("textcat", config={"exclusive_classes": True})
        else:
            textcat = self.nlp.get_pipe("textcat")
        
        # Add labels from training data
        labels = set()
        for example in training_examples:
            for label in example.get("labels", []):
                labels.add(label)
                
        for label in labels:
            textcat.add_label(label)
        
        # Prepare training examples
        examples = await self.prepare_training_data(training_examples)
        
        # Train the model
        self.nlp.initialize()
        for i in range(iterations):
            losses = {}
            self.nlp.update(examples, losses=losses)
            logger.info(f"Training iteration {i+1}/{iterations}, Losses: {losses}")
        
        # Save the trained model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(output_path)
        
        logger.info(f"Custom model saved to: {output_path}")
    
    async def train_entity_recognizer(
        self,
        training_examples: list[dict[str, Any]],
        output_path: str,
        iterations: int = 10
    ) -> None:
        """Train custom entity recognizer."""
        if not self.nlp:
            self.nlp = spacy.load(self.base_model)
        
        # Prepare training examples with entity annotations
        examples = []
        for example in training_examples:
            text = example["text"]
            entities = example.get("entities", [])
            
            doc = self.nlp.make_doc(text)
            example_obj = Example.from_dict(doc, {"entities": entities})
            examples.append(example_obj)
        
        # Train the NER component
        ner = self.nlp.get_pipe("ner")
        
        # Add new entity labels
        for example in training_examples:
            for _, _, label in example.get("entities", []):
                ner.add_label(label)
        
        # Train
        self.nlp.initialize()
        for i in range(iterations):
            losses = {}
            self.nlp.update(examples, losses=losses)
            logger.info(f"NER training iteration {i+1}/{iterations}, Losses: {losses}")
        
        # Save the trained model
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(output_path)
        
        logger.info(f"Custom NER model saved to: {output_path}")
    
    async def evaluate_model(
        self,
        test_examples: list[dict[str, Any]],
        model_path: str | None = None
    ) -> dict[str, Any]:
        """Evaluate trained model performance."""
        if model_path:
            evaluation_nlp = spacy.load(model_path)
        else:
            evaluation_nlp = self.nlp
        
        if not evaluation_nlp:
            raise ValueError("No model available for evaluation")
        
        # Prepare test examples
        test_data = []
        for example in test_examples:
            text = example["text"]
            annotations = example.get("annotations", {})
            doc = evaluation_nlp.make_doc(text)
            test_data.append(Example.from_dict(doc, annotations))
        
        # Evaluate
        scores = evaluation_nlp.evaluate(test_data)
        
        return {
            "textcat_accuracy": scores.get("textcat_accuracy", 0.0),
            "ents_precision": scores.get("ents_p", 0.0),
            "ents_recall": scores.get("ents_r", 0.0),
            "ents_f1": scores.get("ents_f", 0.0),
            "token_accuracy": scores.get("token_acc", 0.0)
        }
```

**Training Data Manager** (`src/codeweaver/providers/nlp/training/data_manager.py`)
```python
import json
from pathlib import Path
from typing import Any, List
import logging

logger = logging.getLogger(__name__)

class TrainingDataManager:
    """Manages training data for custom model training."""
    
    def __init__(self, data_directory: str = "training_data"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
    
    def save_training_examples(
        self, 
        examples: list[dict[str, Any]], 
        filename: str
    ) -> None:
        """Save training examples to file."""
        file_path = self.data_directory / f"{filename}.json"
        
        with open(file_path, 'w') as f:
            json.dump(examples, f, indent=2)
        
        logger.info(f"Saved {len(examples)} training examples to {file_path}")
    
    def load_training_examples(self, filename: str) -> list[dict[str, Any]]:
        """Load training examples from file."""
        file_path = self.data_directory / f"{filename}.json" 
        
        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")
        
        with open(file_path) as f:
            examples = json.load(f)
        
        logger.info(f"Loaded {len(examples)} training examples from {file_path}")
        return examples
    
    def create_intent_training_example(
        self,
        text: str,
        intent_label: str,
        confidence: float = 1.0
    ) -> dict[str, Any]:
        """Create a training example for intent classification."""
        return {
            "text": text,
            "labels": [intent_label],
            "annotations": {
                "cats": {intent_label: confidence}
            }
        }
    
    def create_entity_training_example(
        self,
        text: str,
        entities: list[tuple[int, int, str]]
    ) -> dict[str, Any]:
        """Create a training example for entity recognition."""
        return {
            "text": text,
            "entities": entities,
            "annotations": {
                "entities": entities
            }
        }
    
    def validate_training_data(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate training data format and quality."""
        validation_results = {
            "total_examples": len(examples),
            "valid_examples": 0,
            "errors": [],
            "warnings": []
        }
        
        for i, example in enumerate(examples):
            try:
                # Check required fields
                if "text" not in example:
                    validation_results["errors"].append(f"Example {i}: Missing 'text' field")
                    continue
                
                # Check text quality
                text = example["text"]
                if not text or len(text.strip()) < 3:
                    validation_results["warnings"].append(f"Example {i}: Text too short")
                
                # Check annotations if present
                if "annotations" in example:
                    self._validate_annotations(example["annotations"], i, validation_results)
                
                validation_results["valid_examples"] += 1
                
            except Exception as e:
                validation_results["errors"].append(f"Example {i}: {str(e)}")
        
        return validation_results
    
    def _validate_annotations(
        self, 
        annotations: dict[str, Any], 
        example_index: int, 
        validation_results: dict[str, Any]
    ) -> None:
        """Validate annotation format."""
        if "entities" in annotations:
            entities = annotations["entities"]
            if not isinstance(entities, list):
                validation_results["errors"].append(
                    f"Example {example_index}: 'entities' must be a list"
                )
        
        if "cats" in annotations:
            cats = annotations["cats"]
            if not isinstance(cats, dict):
                validation_results["errors"].append(
                    f"Example {example_index}: 'cats' must be a dictionary"
                )
```

#### Success Criteria - Week 3
- [ ] Semantic caching uses NLP Provider embeddings instead of external API
- [ ] Custom training framework supports intent classification training
- [ ] Cache performance monitoring includes NLP provider metrics
- [ ] Integration maintains >85% cache hit rate

### Week 4: Enhanced Monitoring and Production Features

#### Deliverables

**1. NLP Provider Health Monitoring** (`src/codeweaver/intent/monitoring/nlp_health_monitor.py`)
```python
from dataclasses import dataclass
from codeweaver.types import NLPProvider
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class NLPHealthMetrics:
    """Health metrics for NLP providers."""
    provider_name: str
    model_loaded: bool
    model_name: str
    capabilities: list[str]
    processing_time_avg: float
    memory_usage_mb: float
    error_rate: float
    confidence_scores: list[float]
    requests_per_minute: float
    
class NLPHealthMonitor:
    """Monitor NLP provider health and performance."""
    
    def __init__(self, nlp_provider: NLPProvider):
        self.nlp_provider = nlp_provider
        self.metrics_history = []
        self.processing_times = []
        self.confidence_scores = []
        self.error_count = 0
        self.request_count = 0
        self.start_time = time.time()
        
    async def collect_health_metrics(self) -> NLPHealthMetrics:
        """Collect comprehensive NLP provider health metrics."""
        # Get provider health check
        health_info = await self.nlp_provider.health_check()
        
        # Calculate performance metrics
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        error_rate = (
            self.error_count / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        # Calculate requests per minute
        elapsed_minutes = (time.time() - self.start_time) / 60
        requests_per_minute = (
            self.request_count / elapsed_minutes 
            if elapsed_minutes > 0 else 0.0
        )
        
        return NLPHealthMetrics(
            provider_name=health_info.get("provider_name", "unknown"),
            model_loaded=health_info.get("model_loaded", False),
            model_name=health_info.get("model_name", "unknown"),
            capabilities=health_info.get("capabilities", []),
            processing_time_avg=avg_processing_time,
            memory_usage_mb=health_info.get("memory_usage_mb", 0.0),
            error_rate=error_rate,
            confidence_scores=self.confidence_scores[-100:],  # Last 100 scores
            requests_per_minute=requests_per_minute
        )
    
    def record_processing_time(self, processing_time: float) -> None:
        """Record processing time for metrics."""
        self.processing_times.append(processing_time)
        self.request_count += 1
        
        # Keep only recent times (last 1000)
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-500:]
    
    def record_confidence_score(self, confidence: float) -> None:
        """Record confidence score for metrics."""
        self.confidence_scores.append(confidence)
        
        # Keep only recent scores (last 1000)
        if len(self.confidence_scores) > 1000:
            self.confidence_scores = self.confidence_scores[-500:]
    
    def record_error(self) -> None:
        """Record an error for metrics."""
        self.error_count += 1
        self.request_count += 1
    
    def get_performance_alerts(self, metrics: NLPHealthMetrics) -> list[str]:
        """Check for performance issues and generate alerts."""
        alerts = []
        
        # Check processing time
        if metrics.processing_time_avg > 2.0:  # 2 seconds
            alerts.append(f"High processing time: {metrics.processing_time_avg:.2f}s")
        
        # Check error rate
        if metrics.error_rate > 0.05:  # 5%
            alerts.append(f"High error rate: {metrics.error_rate:.1%}")
        
        # Check model availability
        if not metrics.model_loaded:
            alerts.append("NLP model not loaded")
        
        # Check confidence scores
        if metrics.confidence_scores:
            avg_confidence = sum(metrics.confidence_scores) / len(metrics.confidence_scores)
            if avg_confidence < 0.6:
                alerts.append(f"Low average confidence: {avg_confidence:.2f}")
        
        return alerts
    
    async def run_health_check_loop(self, interval: int = 300) -> None:
        """Run continuous health monitoring loop."""
        while True:
            try:
                metrics = await self.collect_health_metrics()
                alerts = self.get_performance_alerts(metrics)
                
                if alerts:
                    for alert in alerts:
                        logger.warning(f"NLP Health Alert: {alert}")
                
                # Store metrics history
                self.metrics_history.append(metrics)
                
                # Keep only recent history (last 100 entries)
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-50:]
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
            
            await asyncio.sleep(interval)
```

**2. Production Configuration** (`src/codeweaver/intent/production/intent_orchestrator_enhanced.py`)
```python
from codeweaver.intent.orchestrator import IntentOrchestrator
from codeweaver.factories.nlp_provider_registry import NLPProviderRegistry
from codeweaver.intent.caching.nlp_semantic_cache import NLPSemanticCache
from codeweaver.intent.parsing.nlp_enhanced_parser import NLPEnhancedParser
from codeweaver.intent.monitoring.nlp_health_monitor import NLPHealthMonitor
from codeweaver.types import NLPProvider, ServicesManager
import logging

logger = logging.getLogger(__name__)

class ProductionIntentOrchestrator(IntentOrchestrator):
    """Production-ready intent orchestrator with NLP Provider integration."""
    
    def __init__(self, services_manager: ServicesManager, config: dict[str, Any]):
        super().__init__(services_manager)
        self.config = config
        self.nlp_provider_registry = NLPProviderRegistry()
        self.nlp_provider: NLPProvider | None = None
        self.nlp_enhanced_parser: NLPEnhancedParser | None = None
        self.nlp_semantic_cache: NLPSemanticCache | None = None
        self.nlp_health_monitor: NLPHealthMonitor | None = None
        
    async def initialize(self) -> None:
        """Initialize with NLP provider integration."""
        await super().initialize()
        
        # Initialize NLP provider if configured
        nlp_config = self.config.get("nlp", {})
        if nlp_config.get("enabled", True):
            await self._initialize_nlp_provider(nlp_config)
        
        # Initialize enhanced parser
        self.nlp_enhanced_parser = NLPEnhancedParser(self.nlp_provider)
        
        # Initialize NLP semantic cache
        await self._initialize_nlp_semantic_cache()
        
        # Initialize health monitoring
        if self.nlp_provider:
            self.nlp_health_monitor = NLPHealthMonitor(self.nlp_provider)
            
            # Start health monitoring loop if enabled
            if self.config.get("monitoring", {}).get("enabled", True):
                import asyncio
                asyncio.create_task(self.nlp_health_monitor.run_health_check_loop())
        
    async def _initialize_nlp_provider(self, nlp_config: dict[str, Any]) -> None:
        """Initialize NLP provider from configuration."""
        provider_name = nlp_config.get("provider", "spacy")
        
        try:
            self.nlp_provider = self.nlp_provider_registry.create_provider(
                provider_name, 
                nlp_config
            )
            await self.nlp_provider.initialize()
            
            logger.info(f"Initialized NLP provider: {provider_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP provider {provider_name}: {e}")
            if nlp_config.get("required", False):
                raise
            
            logger.warning("Continuing without NLP provider")
    
    async def _initialize_nlp_semantic_cache(self) -> None:
        """Initialize NLP-integrated semantic cache."""
        try:
            cache_service = await self.services_manager.get_cache_service()
            vector_backend = getattr(self.services_manager, 'vector_backend', None)
            
            self.nlp_semantic_cache = NLPSemanticCache(
                cache_service=cache_service,
                vector_backend=vector_backend,
                nlp_provider=self.nlp_provider
            )
            
            logger.info("Initialized NLP semantic cache")
            
        except Exception as e:
            logger.warning(f"Failed to initialize NLP semantic cache: {e}")
    
    async def process_intent(
        self, 
        intent_text: str, 
        context: dict[str, Any] | None = None
    ) -> IntentResult:
        """Process intent with enhanced NLP capabilities."""
        start_time = time.time()
        
        try:
            # Check NLP semantic cache first
            if self.nlp_semantic_cache:
                cached_result = await self.nlp_semantic_cache.get_cached_result(intent_text)
                if cached_result:
                    if self.nlp_health_monitor:
                        processing_time = time.time() - start_time
                        self.nlp_health_monitor.record_processing_time(processing_time)
                    return cached_result
            
            # Parse intent with NLP enhancement
            if self.nlp_enhanced_parser:
                parsed_intent = await self.nlp_enhanced_parser.parse(intent_text)
            else:
                # Fallback to basic parser
                parsed_intent = await self.basic_parser.parse(intent_text)
            
            # Record confidence score for monitoring
            if self.nlp_health_monitor:
                self.nlp_health_monitor.record_confidence_score(parsed_intent.confidence)
            
            # Execute strategy
            result = await self._execute_strategy(parsed_intent, context or {})
            
            # Cache result
            if self.nlp_semantic_cache and result.success:
                await self.nlp_semantic_cache.cache_result(intent_text, result)
            
            # Record processing time
            if self.nlp_health_monitor:
                processing_time = time.time() - start_time
                self.nlp_health_monitor.record_processing_time(processing_time)
            
            return result
            
        except Exception as e:
            # Record error for monitoring
            if self.nlp_health_monitor:
                self.nlp_health_monitor.record_error()
            
            logger.error(f"Intent processing failed: {e}")
            raise
    
    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status including NLP provider."""
        base_status = await super().get_health_status()
        
        if self.nlp_health_monitor:
            nlp_metrics = await self.nlp_health_monitor.collect_health_metrics()
            base_status["nlp"] = {
                "provider": nlp_metrics.provider_name,
                "model": nlp_metrics.model_name,
                "healthy": nlp_metrics.model_loaded and nlp_metrics.error_rate < 0.1,
                "processing_time_avg": nlp_metrics.processing_time_avg,
                "error_rate": nlp_metrics.error_rate,
                "requests_per_minute": nlp_metrics.requests_per_minute
            }
        
        if self.nlp_semantic_cache:
            cache_stats = await self.nlp_semantic_cache.get_cache_statistics()
            base_status["semantic_cache"] = cache_stats
        
        return base_status
    
    async def switch_nlp_model(self, model_name: str) -> bool:
        """Switch NLP model at runtime."""
        if not self.nlp_provider:
            return False
        
        try:
            success = await self.nlp_provider.switch_model(model_name)
            if success:
                logger.info(f"Successfully switched NLP model to: {model_name}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to switch NLP model: {e}")
            return False
```

**3. Enhanced Configuration Schema** (`config/intent_nlp_config.toml`)
```toml
[intent]
enabled = true

[intent.nlp]
provider = "spacy"
model = "en_core_web_sm"
use_transformers = false
enable_intent_classification = true
intent_labels = ["SEARCH", "DOCUMENTATION", "ANALYSIS", "DEBUGGING", "OPTIMIZATION"]
confidence_threshold = 0.7
fallback_enabled = true
custom_patterns_path = "config/custom_patterns.json"

[intent.nlp.custom_components]
# Example custom component configuration
# [[intent.nlp.custom_components]]
# name = "custom_sentiment"
# factory = "sentencizer"
# config = {}

[intent.caching]
enabled = true
semantic_caching_enabled = true
similarity_threshold = 0.85
ttl = 3600

[intent.monitoring]
enabled = true
health_check_interval = 300
performance_alerts = true
metrics_retention = 100

[intent.performance]
optimization_enabled = true
parallel_processing = true
max_workers = 4
timeout = 30
```

#### Success Criteria - Week 4
- [ ] Production orchestrator integrates all NLP Provider features
- [ ] Health monitoring provides comprehensive NLP metrics
- [ ] Configuration supports all NLP provider options
- [ ] Performance meets production requirements (<3s response time)

## 📊 Updated Success Metrics - Phase 2

### NLP Provider Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| **Provider Initialization Time** | <2s | SpaCy model loading time |
| **Intent Classification Accuracy** | >92% | Using TextCategorizer vs pattern matching |
| **Entity Recognition Accuracy** | >90% | Domain-specific entity extraction |
| **Embedding Generation Speed** | <100ms | Per text embedding generation |
| **Model Switching Time** | <5s | Runtime model switching capability |

### Integration Metrics  
| Metric | Target | Measurement |
|--------|--------|-------------|
| **Cache Hit Rate with NLP Embeddings** | >85% | Semantic similarity matching |
| **NLP Provider Availability** | >99% | Health check success rate |
| **Fallback Success Rate** | >95% | Pattern matching when NLP unavailable |
| **Configuration Flexibility Score** | 100% | All provider options configurable |

### Performance Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| **Response Time P95** | <3s | End-to-end intent processing |
| **Memory Usage** | <200MB | Including NLP models |
| **Error Rate** | <1% | Overall system error rate |
| **Throughput** | >100 req/min | Concurrent request handling |

## 🚀 Updated Phase 2 Completion Criteria

✅ **SpaCy Provider Architecture**: Clean separation following CodeWeaver patterns  
✅ **SpaCy 3.7+ Integration**: Modern API with transformer support and graceful fallbacks  
✅ **Full Configurability**: TOML/environment configuration with runtime switching  
✅ **Custom Training Support**: Framework for training custom intent classifiers  
✅ **Multi-Language Ready**: Architecture supports future language expansion  
✅ **Production Monitoring**: Comprehensive health and performance monitoring  
✅ **Semantic Caching**: NLP provider-based embeddings for cache similarity  
✅ **Factory Integration**: Provider registration and discovery via factory system  
✅ **Error Recovery**: Intelligent fallbacks with context preservation  
✅ **Performance Optimization**: Sub-3s response times with resource monitoring  

**Ready for Phase 3**: Advanced features including user learning, multi-provider NLP, and debugging tools

---

## 🎯 Key Architectural Benefits

1. **True Provider Architecture**: SpaCy is now a swappable provider that can be used by any part of CodeWeaver
2. **SpaCy 3.7+ Compliance**: Leverages modern features like TextCategorizer, EntityRuler, and transformer support
3. **Custom Training Ready**: Framework supports training custom models for specific domains
4. **Configuration-Driven**: All NLP features configurable via TOML and environment variables
5. **Multi-Provider Ready**: Architecture supports adding other NLP providers (e.g., transformers, OpenAI)
6. **Production Monitoring**: Comprehensive health monitoring and performance tracking
7. **Graceful Degradation**: Multiple fallback layers ensure system reliability
8. **Extensible Patterns**: Domain patterns can be customized and extended at runtime
9. **Performance Optimized**: Intelligent caching and batch processing for production workloads
10. **Developer Experience**: Clear configuration, monitoring, and debugging capabilities

This updated plan transforms SpaCy from a tightly-coupled component into a flexible, extensible provider that enhances the entire CodeWeaver system while maintaining the clean architectural patterns you've established. The implementation follows modern SpaCy 3.7+ best practices and provides a solid foundation for future enhancements.

---

*This phase significantly enhances the intent layer's intelligence and performance while maintaining full architectural compliance, delivering a production-ready natural language interface for CodeWeaver with modern NLP capabilities.*