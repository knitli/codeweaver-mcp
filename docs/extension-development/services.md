# Building Custom Services

This guide covers building custom service providers for CodeWeaver. Services provide middleware functionality and cross-cutting concerns like chunking, filtering, validation, and caching.

## 🎯 Overview

Services in CodeWeaver provide middleware functionality that enhances the core capabilities. CodeWeaver supports various service types:

- **Chunking Services**: Intelligent content segmentation and processing
- **Filtering Services**: File discovery and content filtering
- **Validation Services**: Content validation and quality checks
- **Cache Services**: Performance optimization through intelligent caching
- **Monitoring Services**: Health monitoring and metrics collection
- **Custom Services**: Specialized middleware and processing capabilities

## 🏗️ Service Architecture

### Core ServiceProvider Protocol

```python
from typing import Protocol, runtime_checkable
from codeweaver.cw_types import ServiceType, ServiceHealth, ProviderStatus

@runtime_checkable
class ServiceProvider(Protocol):
    """Protocol for service providers."""
    
    # Properties
    @property
    def service_type(self) -> ServiceType:
        """Type of service provided."""
        ...
    
    @property
    def status(self) -> ProviderStatus:
        """Current provider status."""
        ...
    
    # Lifecycle Management
    async def initialize(self) -> None:
        """Initialize service provider."""
        ...
    
    async def shutdown(self) -> None:
        """Shutdown service provider."""
        ...
    
    # Health Monitoring
    async def health_check(self) -> ServiceHealth:
        """Check service health."""
        ...
    
    def record_operation(self, *, success: bool, error: str | None = None) -> None:
        """Record operation result for monitoring."""
        ...
```

### Base Service Provider Class

CodeWeaver provides an abstract base class for service implementations:

```python
from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.cw_types import ServiceType, ServiceConfig, ProviderStatus
import logging
from abc import ABC, abstractmethod

class BaseServiceProvider(ServiceProvider, ABC):
    """Abstract base class for service providers."""
    
    def __init__(self, service_type: ServiceType, config: ServiceConfig, logger: logging.Logger | None = None):
        self._service_type = service_type
        self._config = config
        self._status = ProviderStatus.REGISTERED
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._operation_stats = {"success_count": 0, "error_count": 0}
    
    # Required implementations
    @abstractmethod
    async def _initialize_provider(self) -> None:
        """Initialize provider-specific resources."""
        ...
    
    @abstractmethod
    async def _shutdown_provider(self) -> None:
        """Cleanup provider-specific resources."""
        ...
    
    @abstractmethod
    async def _check_health(self) -> bool:
        """Check provider-specific health."""
        ...
    
    # Lifecycle management (implemented)
    async def initialize(self) -> None:
        """Initialize service provider."""
        try:
            await self._initialize_provider()
            self._status = ProviderStatus.ACTIVE
        except Exception as e:
            self._status = ProviderStatus.ERROR
            raise ServiceInitializationError(f"Failed to initialize service: {e}") from e
    
    async def shutdown(self) -> None:
        """Shutdown service provider."""
        try:
            await self._shutdown_provider()
            self._status = ProviderStatus.STOPPED
        except Exception as e:
            self._status = ProviderStatus.ERROR
            raise ServiceStopError(f"Failed to shutdown service: {e}") from e
```

## 🚀 Implementation Guide

### Step 1: Define Service Configuration

Create a Pydantic configuration model for your service:

```python
from pydantic import BaseModel, Field
from typing import Annotated, Literal

class MyServiceConfig(BaseModel):
    """Configuration for MyService."""
    
    # Provider identification
    provider: Annotated[str, Field(description="Service provider name")]
    
    # Performance settings
    max_concurrent_operations: Annotated[int, Field(default=10, ge=1, le=100)]
    operation_timeout: Annotated[int, Field(default=30, ge=1, le=300)]
    cache_enabled: Annotated[bool, Field(default=True)]
    cache_ttl: Annotated[int, Field(default=3600, ge=0)]
    
    # Quality settings
    quality_threshold: Annotated[float, Field(default=0.8, ge=0.0, le=1.0)]
    retry_attempts: Annotated[int, Field(default=3, ge=0, le=10)]
    
    # Resource limits
    max_memory_mb: Annotated[int, Field(default=500, ge=100, le=2000)]
    temp_dir: Annotated[str | None, Field(default=None)]
    
    # Monitoring
    metrics_enabled: Annotated[bool, Field(default=True)]
    health_check_interval: Annotated[int, Field(default=60, ge=10, le=300)]
```

### Step 2: Implement Chunking Service

```python
from codeweaver.services.providers.base_provider import BaseServiceProvider
from codeweaver.cw_types import ServiceType, ChunkingService
import asyncio
import time
from typing import Any

class MyChunkingService(BaseServiceProvider, ChunkingService):
    """Custom chunking service implementation."""
    
    def __init__(self, config: MyServiceConfig):
        super().__init__(ServiceType.CHUNKING, config)
        self.parser_cache: dict[str, Any] = {}
        self.chunk_cache: dict[str, list[str]] = {}
        self._semaphore: asyncio.Semaphore | None = None
    
    async def _initialize_provider(self) -> None:
        """Initialize chunking service resources."""
        # Create semaphore for concurrent operations
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)
        
        # Initialize any external libraries
        await self._initialize_parsers()
        
        # Setup temporary directory
        if self.config.temp_dir:
            import os
            os.makedirs(self.config.temp_dir, exist_ok=True)
        
        self._logger.info("Chunking service initialized")
    
    async def _shutdown_provider(self) -> None:
        """Cleanup chunking service resources."""
        # Clear caches
        self.parser_cache.clear()
        self.chunk_cache.clear()
        
        # Cleanup temporary files
        if self.config.temp_dir:
            import shutil
            try:
                shutil.rmtree(self.config.temp_dir)
            except Exception as e:
                self._logger.warning(f"Failed to cleanup temp dir: {e}")
        
        self._logger.info("Chunking service shutdown complete")
    
    async def _check_health(self) -> bool:
        """Check chunking service health."""
        try:
            # Test basic chunking operation
            test_content = "This is a test document for health checking."
            chunks = await self.chunk_content(test_content)
            return len(chunks) > 0
        except Exception:
            return False
    
    async def chunk_content(
        self, 
        content: str, 
        file_path: str | None = None
    ) -> list[str]:
        """
        Chunk content into segments with intelligent processing.
        
        Args:
            content: Content to chunk
            file_path: Optional file path for context
            
        Returns:
            List of content chunks
            
        Raises:
            ChunkingError: If chunking fails
            ValidationError: If content is invalid
        """
        if not content or not content.strip():
            return []
        
        # Rate limiting
        async with self._semaphore:
            start_time = time.time()
            
            try:
                # Check cache first
                cache_key = self._generate_cache_key(content, file_path)
                if self.config.cache_enabled and cache_key in self.chunk_cache:
                    self.record_operation(success=True)
                    return self.chunk_cache[cache_key]
                
                # Determine chunking strategy
                chunks = await self._chunk_with_strategy(content, file_path)
                
                # Validate chunks
                validated_chunks = await self._validate_chunks(chunks)
                
                # Cache results
                if self.config.cache_enabled:
                    self.chunk_cache[cache_key] = validated_chunks
                
                # Record success
                processing_time = time.time() - start_time
                self.record_operation(success=True)
                self._logger.debug(f"Chunked content in {processing_time:.3f}s, {len(validated_chunks)} chunks")
                
                return validated_chunks
                
            except Exception as e:
                self.record_operation(success=False, error=str(e))
                self._logger.error(f"Chunking failed: {e}")
                raise ChunkingError(f"Failed to chunk content: {e}") from e
    
    async def _chunk_with_strategy(self, content: str, file_path: str | None) -> list[str]:
        """Choose and apply appropriate chunking strategy."""
        content_length = len(content)
        
        # Strategy selection based on content and file type
        if file_path and self._is_code_file(file_path):
            return await self._chunk_code_ast(content, file_path)
        elif content_length > 100000:
            return await self._chunk_streaming(content)
        elif self._is_structured_content(content):
            return await self._chunk_structured(content)
        else:
            return await self._chunk_semantic(content)
    
    async def _chunk_code_ast(self, content: str, file_path: str) -> list[str]:
        """Chunk code using AST parsing."""
        language = self._detect_language(file_path)
        
        # Get or create parser
        parser = await self._get_ast_parser(language)
        if not parser:
            return await self._chunk_fallback(content)
        
        try:
            # Parse with timeout
            tree = await asyncio.wait_for(
                self._parse_content_ast(parser, content),
                timeout=self.config.operation_timeout
            )
            
            # Extract semantic chunks from AST
            chunks = self._extract_ast_chunks(tree, content, language)
            return chunks
            
        except asyncio.TimeoutError:
            self._logger.warning("AST parsing timed out, using fallback")
            return await self._chunk_fallback(content)
        except Exception as e:
            self._logger.warning(f"AST parsing failed: {e}, using fallback")
            return await self._chunk_fallback(content)
    
    async def _get_ast_parser(self, language: str):
        """Get cached AST parser for language."""
        if language not in self.parser_cache:
            try:
                # Try to load tree-sitter parser
                import tree_sitter
                
                # Load language-specific parser
                parser = tree_sitter.Parser()
                # Implementation would load specific language grammar
                self.parser_cache[language] = parser
                
            except ImportError:
                self._logger.warning("tree-sitter not available, disabling AST parsing")
                self.parser_cache[language] = None
            except Exception as e:
                self._logger.warning(f"Failed to load parser for {language}: {e}")
                self.parser_cache[language] = None
        
        return self.parser_cache[language]
    
    async def _parse_content_ast(self, parser, content: str):
        """Parse content using AST parser."""
        # Run parsing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: parser.parse(content.encode())
        )
    
    def _extract_ast_chunks(self, tree, content: str, language: str) -> list[str]:
        """Extract semantic chunks from AST tree."""
        chunks = []
        
        # Language-specific chunk extraction
        if language == "python":
            chunks = self._extract_python_chunks(tree, content)
        elif language in ["javascript", "typescript"]:
            chunks = self._extract_js_chunks(tree, content)
        else:
            # Generic extraction
            chunks = self._extract_generic_chunks(tree, content)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _extract_python_chunks(self, tree, content: str) -> list[str]:
        """Extract Python-specific chunks."""
        chunks = []
        lines = content.splitlines()
        
        # Find functions, classes, and top-level statements
        for node in tree.root_node.children:
            if node.type in ["function_definition", "class_definition"]:
                start_line = node.start_point[0]
                end_line = node.end_point[0] + 1
                chunk = "\n".join(lines[start_line:end_line])
                chunks.append(chunk)
            elif node.type in ["import_statement", "import_from_statement"]:
                # Group imports together
                if not chunks or not chunks[-1].startswith(("import ", "from ")):
                    chunks.append(lines[node.start_point[0]])
                else:
                    chunks[-1] += "\n" + lines[node.start_point[0]]
        
        return chunks
    
    async def _chunk_streaming(self, content: str) -> list[str]:
        """Chunk large content using streaming approach."""
        chunks = []
        current_chunk = ""
        max_chunk_size = 1500
        min_chunk_size = 50
        
        # Process content line by line
        for line in content.splitlines(keepends=True):
            if len(current_chunk) + len(line) > max_chunk_size:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = line
                else:
                    # Line is too long, force split
                    current_chunk += line
                    if len(current_chunk) > max_chunk_size:
                        chunks.append(current_chunk[:max_chunk_size])
                        current_chunk = current_chunk[max_chunk_size:]
            else:
                current_chunk += line
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _chunk_semantic(self, content: str) -> list[str]:
        """Chunk content using semantic boundaries."""
        # Split by paragraphs and sentences
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        max_chunk_size = 1500
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    # Paragraph is too long, split by sentences
                    sentences = self._split_sentences(paragraph)
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > max_chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting (could be enhanced with NLP libraries)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _chunk_structured(self, content: str) -> list[str]:
        """Chunk structured content (JSON, XML, etc.)."""
        # Detect content type and chunk accordingly
        if content.strip().startswith('{') or content.strip().startswith('['):
            return await self._chunk_json(content)
        elif content.strip().startswith('<'):
            return await self._chunk_xml(content)
        else:
            return await self._chunk_semantic(content)
    
    async def _chunk_json(self, content: str) -> list[str]:
        """Chunk JSON content."""
        import json
        try:
            data = json.loads(content)
            
            if isinstance(data, list):
                # Chunk array elements
                chunks = []
                for i in range(0, len(data), 10):  # 10 items per chunk
                    chunk_data = data[i:i+10]
                    chunks.append(json.dumps(chunk_data, indent=2))
                return chunks
            elif isinstance(data, dict):
                # Chunk by top-level keys
                chunks = []
                for key, value in data.items():
                    chunk = {key: value}
                    chunks.append(json.dumps(chunk, indent=2))
                return chunks
            else:
                return [content]
                
        except json.JSONDecodeError:
            return await self._chunk_semantic(content)
    
    async def _chunk_fallback(self, content: str) -> list[str]:
        """Fallback chunking strategy."""
        return await self._chunk_semantic(content)
    
    async def _validate_chunks(self, chunks: list[str]) -> list[str]:
        """Validate and filter chunks."""
        validated = []
        min_length = 10
        max_length = 2000
        
        for chunk in chunks:
            # Skip empty or very short chunks
            if len(chunk.strip()) < min_length:
                continue
            
            # Truncate very long chunks
            if len(chunk) > max_length:
                chunk = chunk[:max_length] + "..."
            
            # Quality check
            if self._meets_quality_threshold(chunk):
                validated.append(chunk.strip())
        
        return validated
    
    def _meets_quality_threshold(self, chunk: str) -> bool:
        """Check if chunk meets quality threshold."""
        # Simple quality metrics
        word_count = len(chunk.split())
        char_count = len(chunk)
        
        # Must have minimum words and reasonable word-to-char ratio
        if word_count < 3:
            return False
        
        avg_word_length = char_count / word_count if word_count > 0 else 0
        if avg_word_length < 2 or avg_word_length > 20:
            return False
        
        return True
    
    def _generate_cache_key(self, content: str, file_path: str | None) -> str:
        """Generate cache key for content."""
        import hashlib
        key_data = f"{content[:100]}{file_path or ''}{len(content)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file."""
        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c',
            '.rs', '.go', '.php', '.rb', '.cs', '.scala', '.kt'
        }
        from pathlib import Path
        return Path(file_path).suffix.lower() in code_extensions
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file path."""
        from pathlib import Path
        extension = Path(file_path).suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.rs': 'rust',
            '.go': 'go',
            '.php': 'php',
            '.rb': 'ruby',
        }
        
        return language_map.get(extension, 'text')
    
    def _is_structured_content(self, content: str) -> bool:
        """Check if content is structured (JSON, XML, etc.)."""
        stripped = content.strip()
        return (
            (stripped.startswith('{') and stripped.endswith('}')) or
            (stripped.startswith('[') and stripped.endswith(']')) or
            (stripped.startswith('<') and stripped.endswith('>'))
        )
    
    def get_chunk_stats(self, chunks: list[str]) -> dict[str, Any]:
        """Get statistics about chunks."""
        if not chunks:
            return {"count": 0}
        
        chunk_lengths = [len(chunk) for chunk in chunks]
        word_counts = [len(chunk.split()) for chunk in chunks]
        
        return {
            "count": len(chunks),
            "total_length": sum(chunk_lengths),
            "avg_length": sum(chunk_lengths) / len(chunks),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
            "total_words": sum(word_counts),
            "avg_words": sum(word_counts) / len(chunks),
        }
```

### Step 3: Implement Filtering Service

```python
from pathlib import Path
from codeweaver.cw_types import FilteringService
import fnmatch
import asyncio

class MyFilteringService(BaseServiceProvider, FilteringService):
    """Custom filtering service implementation."""
    
    def __init__(self, config: MyServiceConfig):
        super().__init__(ServiceType.FILTERING, config)
        self.gitignore_patterns: list[str] = []
        self.custom_filters: dict[str, callable] = {}
    
    async def _initialize_provider(self) -> None:
        """Initialize filtering service."""
        # Load .gitignore patterns
        await self._load_gitignore_patterns()
        
        # Initialize custom filters
        self._setup_custom_filters()
        
        self._logger.info("Filtering service initialized")
    
    async def _shutdown_provider(self) -> None:
        """Cleanup filtering service."""
        self.gitignore_patterns.clear()
        self.custom_filters.clear()
        
        self._logger.info("Filtering service shutdown complete")
    
    async def _check_health(self) -> bool:
        """Check filtering service health."""
        try:
            # Test basic filtering operation
            test_path = Path("/tmp/test.py")
            return self.should_include_file(test_path)
        except Exception:
            return False
    
    async def discover_files(
        self, 
        base_path: Path, 
        patterns: list[str] | None = None
    ) -> list[Path]:
        """
        Discover files based on filtering criteria.
        
        Args:
            base_path: Base directory to search
            patterns: Optional glob patterns to match
            
        Returns:
            List of discovered file paths
        """
        if not base_path.exists():
            raise FilteringError(f"Base path does not exist: {base_path}")
        
        start_time = time.time()
        discovered_files = []
        
        try:
            # Use async discovery for large directories
            if await self._is_large_directory(base_path):
                discovered_files = await self._discover_files_async(base_path, patterns)
            else:
                discovered_files = await self._discover_files_sync(base_path, patterns)
            
            # Apply filters
            filtered_files = []
            for file_path in discovered_files:
                if self.should_include_file(file_path):
                    filtered_files.append(file_path)
            
            processing_time = time.time() - start_time
            self.record_operation(success=True)
            self._logger.debug(
                f"Discovered {len(filtered_files)} files in {processing_time:.3f}s"
            )
            
            return filtered_files
            
        except Exception as e:
            self.record_operation(success=False, error=str(e))
            raise FilteringError(f"File discovery failed: {e}") from e
    
    async def _is_large_directory(self, path: Path) -> bool:
        """Check if directory is large enough to warrant async processing."""
        try:
            # Quick check of immediate children
            children = list(path.iterdir())
            return len(children) > 1000
        except Exception:
            return False
    
    async def _discover_files_async(
        self, 
        base_path: Path, 
        patterns: list[str] | None
    ) -> list[Path]:
        """Async file discovery for large directories."""
        discovered = []
        semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)
        
        async def process_directory(dir_path: Path):
            async with semaphore:
                try:
                    for item in dir_path.iterdir():
                        if item.is_file():
                            if not patterns or any(
                                fnmatch.fnmatch(item.name, pattern) 
                                for pattern in patterns
                            ):
                                discovered.append(item)
                        elif item.is_dir() and not self._should_skip_directory(item):
                            # Recursive discovery
                            await process_directory(item)
                except PermissionError:
                    self._logger.warning(f"Permission denied: {dir_path}")
        
        await process_directory(base_path)
        return discovered
    
    async def _discover_files_sync(
        self, 
        base_path: Path, 
        patterns: list[str] | None
    ) -> list[Path]:
        """Synchronous file discovery for smaller directories."""
        discovered = []
        
        def _walk_directory(dir_path: Path):
            try:
                for item in dir_path.iterdir():
                    if item.is_file():
                        if not patterns or any(
                            fnmatch.fnmatch(item.name, pattern) 
                            for pattern in patterns
                        ):
                            discovered.append(item)
                    elif item.is_dir() and not self._should_skip_directory(item):
                        _walk_directory(item)
            except PermissionError:
                self._logger.warning(f"Permission denied: {dir_path}")
        
        _walk_directory(base_path)
        return discovered
    
    def should_include_file(self, file_path: Path) -> bool:
        """
        Check if file should be included based on filters.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be included
        """
        # Check file size
        try:
            if file_path.stat().st_size > self.config.max_memory_mb * 1024 * 1024:
                return False
        except OSError:
            return False
        
        # Check gitignore patterns
        if self._matches_gitignore(file_path):
            return False
        
        # Check custom filters
        for filter_name, filter_func in self.custom_filters.items():
            try:
                if not filter_func(file_path):
                    return False
            except Exception as e:
                self._logger.warning(f"Filter {filter_name} failed: {e}")
        
        # Check file type
        file_type = self.get_file_type(file_path)
        if file_type and not self._is_supported_file_type(file_type):
            return False
        
        return True
    
    def get_file_type(self, file_path: Path) -> str | None:
        """
        Get file type from path.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            File type string or None if unknown
        """
        import mimetypes
        
        # Try MIME type detection
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            return mime_type
        
        # Fallback to extension-based detection
        extension = file_path.suffix.lower()
        extension_map = {
            '.py': 'text/x-python',
            '.js': 'text/javascript',
            '.ts': 'text/typescript',
            '.jsx': 'text/jsx',
            '.tsx': 'text/tsx',
            '.java': 'text/x-java',
            '.cpp': 'text/x-c++',
            '.c': 'text/x-c',
            '.rs': 'text/x-rust',
            '.go': 'text/x-go',
            '.php': 'text/x-php',
            '.rb': 'text/x-ruby',
            '.md': 'text/markdown',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.xml': 'application/xml',
            '.yaml': 'application/yaml',
            '.yml': 'application/yaml',
        }
        
        return extension_map.get(extension)
    
    def _is_supported_file_type(self, file_type: str) -> bool:
        """Check if file type is supported."""
        supported_types = {
            'text/plain', 'text/x-python', 'text/javascript', 
            'text/typescript', 'text/jsx', 'text/tsx',
            'text/x-java', 'text/x-c++', 'text/x-c',
            'text/x-rust', 'text/x-go', 'text/x-php',
            'text/x-ruby', 'text/markdown', 'application/json',
            'application/xml', 'application/yaml'
        }
        
        return file_type in supported_types
    
    async def _load_gitignore_patterns(self):
        """Load .gitignore patterns."""
        # This would load from actual .gitignore files
        default_patterns = [
            '*.pyc', '__pycache__/', '.git/', '.svn/',
            'node_modules/', '.DS_Store', '*.log',
            '.env', '.venv/', 'venv/', '.idea/',
            '.vscode/', '*.tmp', '*.swp', '*.bak'
        ]
        
        self.gitignore_patterns = default_patterns
    
    def _matches_gitignore(self, file_path: Path) -> bool:
        """Check if file matches gitignore patterns."""
        path_str = str(file_path)
        
        for pattern in self.gitignore_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
        
        return False
    
    def _should_skip_directory(self, dir_path: Path) -> bool:
        """Check if directory should be skipped."""
        skip_dirs = {
            '.git', '.svn', '__pycache__', 'node_modules',
            '.idea', '.vscode', '.venv', 'venv'
        }
        
        return dir_path.name in skip_dirs
    
    def _setup_custom_filters(self):
        """Setup custom filtering functions."""
        self.custom_filters = {
            'binary_filter': self._filter_binary_files,
            'size_filter': self._filter_large_files,
            'permission_filter': self._filter_permission_denied,
        }
    
    def _filter_binary_files(self, file_path: Path) -> bool:
        """Filter out binary files."""
        try:
            # Simple heuristic: check for null bytes in first 1KB
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return b'\x00' not in chunk
        except Exception:
            return False
    
    def _filter_large_files(self, file_path: Path) -> bool:
        """Filter out excessively large files."""
        try:
            max_size = self.config.max_memory_mb * 1024 * 1024
            return file_path.stat().st_size <= max_size
        except Exception:
            return False
    
    def _filter_permission_denied(self, file_path: Path) -> bool:
        """Filter out files we can't read."""
        try:
            return file_path.is_file() and file_path.stat()
        except PermissionError:
            return False
        except Exception:
            return False
```

### Step 4: Create Service Plugin Interface

```python
from codeweaver.factories.plugin_protocols import PluginInterface
from codeweaver.cw_types import (
    ComponentType, BaseCapabilities, BaseComponentInfo, 
    ValidationResult, ServiceCapabilities
)

class MyServicePlugin(PluginInterface):
    """Plugin interface for MyService."""
    
    @classmethod
    def get_plugin_name(cls) -> str:
        return "my_custom_service"
    
    @classmethod
    def get_component_type(cls) -> ComponentType:
        return ComponentType.SERVICE
    
    @classmethod
    def get_capabilities(cls) -> BaseCapabilities:
        return ServiceCapabilities(
            service_types=[ServiceType.CHUNKING, ServiceType.FILTERING],
            supports_async=True,
            supports_caching=True,
            supports_monitoring=True,
            max_concurrent_operations=50,
            supported_content_types=[
                "text/plain", "text/x-python", "text/javascript",
                "text/markdown", "application/json"
            ]
        )
    
    @classmethod
    def get_component_info(cls) -> BaseComponentInfo:
        return BaseComponentInfo(
            name="my_custom_service",
            display_name="My Custom Service",
            description="Custom service for content processing and filtering",
            component_type=ComponentType.SERVICE,
            version="1.0.0",
            author="Your Name",
            homepage="https://github.com/yourname/my-service"
        )
    
    @classmethod
    def validate_config(cls, config: MyServiceConfig) -> ValidationResult:
        """Validate service configuration."""
        errors = []
        warnings = []
        
        # Validate resource limits
        if config.max_memory_mb < 100:
            errors.append("Memory limit too low (minimum 100MB)")
        
        if config.max_memory_mb > 2000:
            warnings.append("High memory limit may impact system performance")
        
        # Validate timeouts
        if config.operation_timeout < 5:
            warnings.append("Low timeout may cause operations to fail")
        
        if config.operation_timeout > 300:
            warnings.append("High timeout may cause system delays")
        
        # Validate concurrency
        if config.max_concurrent_operations > 50:
            warnings.append("High concurrency may overwhelm the system")
        
        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def get_dependencies(cls) -> list[str]:
        """Get required dependencies."""
        return ["asyncio", "pathlib", "fnmatch"]
    
    @classmethod
    def get_service_class(cls) -> type:
        """Get service implementation class."""
        return MyChunkingService  # or return a factory that creates both services
```

### Step 5: Register the Service

```python
from codeweaver.services.manager import ServicesManager
from codeweaver.cw_types import ServicesConfig

# Create service configuration
services_config = ServicesConfig(
    chunking=MyServiceConfig(
        provider="my_custom_service",
        max_concurrent_operations=10,
        cache_enabled=True
    ),
    filtering=MyServiceConfig(
        provider="my_custom_service",
        max_memory_mb=500
    )
)

# Create services manager
services_manager = ServicesManager(services_config)

# Register custom service
services_manager.register_service_provider(
    ServiceType.CHUNKING,
    "my_custom_service",
    MyChunkingService
)

services_manager.register_service_provider(
    ServiceType.FILTERING,
    "my_custom_service", 
    MyFilteringService
)

# Initialize services
await services_manager.initialize()
```

## 🧪 Testing Your Service

### Unit Tests

```python
import pytest
from unittest.mock import AsyncMock, patch
from my_package.service import MyChunkingService, MyServiceConfig
from codeweaver.cw_types import ServiceType

@pytest.fixture
def service_config():
    return MyServiceConfig(
        provider="my_custom_service",
        max_concurrent_operations=5,
        operation_timeout=30,
        cache_enabled=True
    )

@pytest.fixture
def chunking_service(service_config):
    return MyChunkingService(service_config)

class TestMyChunkingService:
    """Test suite for MyChunkingService."""
    
    async def test_service_initialization(self, chunking_service):
        """Test service initialization."""
        await chunking_service.initialize()
        
        assert chunking_service.status == ProviderStatus.ACTIVE
        assert chunking_service._semaphore is not None
        
        await chunking_service.shutdown()
        assert chunking_service.status == ProviderStatus.STOPPED
    
    async def test_chunk_content_basic(self, chunking_service):
        """Test basic content chunking."""
        await chunking_service.initialize()
        
        content = "This is a test document. It has multiple sentences. Each sentence should be preserved in chunking."
        
        chunks = await chunking_service.chunk_content(content)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        
        await chunking_service.shutdown()
    
    async def test_chunk_content_code(self, chunking_service):
        """Test code content chunking."""
        await chunking_service.initialize()
        
        python_code = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")

class MyClass:
    """Example class."""
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''
        
        chunks = await chunking_service.chunk_content(python_code, "test.py")
        
        assert len(chunks) >= 2  # Should have function and class chunks
        assert any("def hello_world" in chunk for chunk in chunks)
        assert any("class MyClass" in chunk for chunk in chunks)
        
        await chunking_service.shutdown()
    
    async def test_chunk_caching(self, chunking_service):
        """Test chunk caching functionality."""
        await chunking_service.initialize()
        
        content = "This is test content for caching."
        
        # First call
        chunks1 = await chunking_service.chunk_content(content)
        
        # Second call should use cache
        with patch.object(chunking_service, '_chunk_with_strategy') as mock_strategy:
            chunks2 = await chunking_service.chunk_content(content)
            
            # Strategy should not be called due to cache hit
            mock_strategy.assert_not_called()
            assert chunks1 == chunks2
        
        await chunking_service.shutdown()
    
    async def test_chunk_stats(self, chunking_service):
        """Test chunk statistics."""
        chunks = ["Short chunk", "This is a longer chunk with more content", "Medium chunk here"]
        
        stats = chunking_service.get_chunk_stats(chunks)
        
        assert stats["count"] == 3
        assert stats["total_length"] > 0
        assert stats["avg_length"] > 0
        assert stats["min_length"] > 0
        assert stats["max_length"] > 0
        assert stats["total_words"] > 0
    
    async def test_health_check(self, chunking_service):
        """Test service health check."""
        await chunking_service.initialize()
        
        health = await chunking_service.health_check()
        
        assert health.status in [ProviderStatus.ACTIVE, ProviderStatus.HEALTHY]
        assert health.last_check > 0
        
        await chunking_service.shutdown()
    
    async def test_error_handling(self, chunking_service):
        """Test error handling."""
        await chunking_service.initialize()
        
        # Test with invalid content
        with pytest.raises(ChunkingError):
            await chunking_service.chunk_content(None)
        
        await chunking_service.shutdown()
```

### Integration Tests

```python
@pytest.mark.integration
class TestServiceIntegration:
    """Integration tests for services."""
    
    async def test_service_manager_integration(self):
        """Test integration with ServicesManager."""
        from codeweaver.services.manager import ServicesManager
        from codeweaver.cw_types import ServicesConfig
        
        config = ServicesConfig(
            chunking=MyServiceConfig(
                provider="my_custom_service",
                cache_enabled=True
            )
        )
        
        services_manager = ServicesManager(config)
        
        # Register service
        services_manager.register_service_provider(
            ServiceType.CHUNKING,
            "my_custom_service",
            MyChunkingService
        )
        
        # Initialize
        await services_manager.initialize()
        
        # Get service
        chunking_service = services_manager.get_chunking_service()
        assert chunking_service is not None
        
        # Test service
        chunks = await chunking_service.chunk_content("Test content")
        assert isinstance(chunks, list)
        
        # Health check
        health_report = await services_manager.get_health_report()
        assert health_report.overall_status in ["healthy", "active"]
        
        # Cleanup
        await services_manager.shutdown()
```

## 📊 Performance Guidelines

### Service Performance
- **Initialize efficiently**: Minimize startup time
- **Use async operations**: Prevent blocking the event loop
- **Implement caching**: Cache expensive operations appropriately
- **Monitor resource usage**: Track memory and CPU usage
- **Handle failures gracefully**: Implement proper error handling

### Memory Management
- **Clean up resources**: Implement proper cleanup in shutdown
- **Use streaming**: Process large datasets in chunks
- **Monitor memory leaks**: Track memory usage over time
- **Implement limits**: Set reasonable resource limits

### Concurrency
- **Use semaphores**: Control concurrent operations
- **Avoid blocking**: Use async/await properly
- **Handle timeouts**: Implement operation timeouts
- **Scale appropriately**: Configure concurrency limits

## 🚀 Next Steps

- **[Testing Framework →](./testing.md)**: Comprehensive testing strategies
- **[Performance Guidelines →](./performance.md)**: Optimization best practices
- **[Protocol Reference →](../reference/protocols.md)**: Complete protocol documentation