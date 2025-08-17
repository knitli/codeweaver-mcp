# sourcery skip: lambdas-should-be-short, no-complex-if-expressions
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Statistics tracking for CodeWeaver, including file indexing, retrieval, and session performance metrics.
"""

from __future__ import annotations

import contextlib
import statistics

from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal, cast

from pydantic import (
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    computed_field,
)
from pydantic.dataclasses import dataclass

from codeweaver._data_structures import ExtKind
from codeweaver.language import SemanticSearchLanguage


type OperationsKey = Literal["indexed", "retrieved", "processed", "reindexed", "skipped"]
type SummaryKey = Literal["total_operations", "unique_files"]
type CategoryKey = Literal["code", "config", "docs", "other"]

type McpOperationRequests = Literal[
    "on_call_tool_requests",
    "on_read_resource_requests",
    "on_get_prompt_requests",
    "on_list_tools_requests",
    "on_list_resources_requests",
    "on_list_resource_templates_requests",
    "on_list_prompts_requests",
]

type TimingRequestsDict = dict[
    Literal["averages", "counts", "lows", "medians", "highs"],
    dict[McpOperationRequests, NonNegativeFloat],
]


@dataclass(config=ConfigDict(extra="forbid"))
class TimingStatistics:
    """By-operation timing statistics for CodeWeaver operations."""

    on_call_tool_requests: Annotated[
        list[PositiveFloat],
        Field(
            default_factory=list,
            description="Time taken for on_call_tool requests in milliseconds.",
        ),
    ]
    on_read_resource_requests: Annotated[
        list[PositiveFloat],
        Field(
            default_factory=list,
            description="Time taken for on_read_resource requests in milliseconds.",
        ),
    ]
    on_get_prompt_requests: Annotated[
        list[PositiveFloat],
        Field(
            default_factory=list,
            description="Time taken for on_get_prompt requests in milliseconds.",
        ),
    ]
    on_list_tools_requests: Annotated[
        list[PositiveFloat],
        Field(
            default_factory=list,
            description="Time taken for on_list_tools requests in milliseconds.",
        ),
    ]
    on_list_resources_requests: Annotated[
        list[PositiveFloat],
        Field(
            default_factory=list,
            description="Time taken for on_list_resources requests in milliseconds.",
        ),
    ]
    on_list_resource_templates_requests: Annotated[
        list[PositiveFloat],
        Field(
            default_factory=list,
            description="Time taken for on_list_resource_templates requests in milliseconds.",
        ),
    ]
    on_list_prompts_requests: Annotated[
        list[PositiveFloat],
        Field(
            default_factory=list,
            description="Time taken for on_list_prompts requests in milliseconds.",
        ),
    ]

    def update(self, key: McpOperationRequests, response_time: PositiveFloat) -> None:
        """Update the timing statistics for a specific request type."""
        if (request_list := getattr(self, key)) and isinstance(request_list, list):
            self.__setattr__(key, [*request_list, response_time])

    @computed_field
    def timing_summary(self) -> TimingRequestsDict:
        """Get a summary of timing statistics."""
        return {
            "averages": {
                "on_call_tool_requests": (
                    sum(self.on_call_tool_requests) / len(self.on_call_tool_requests)
                    if self.on_call_tool_requests
                    else 0.0
                ),
                "on_read_resource_requests": (
                    sum(self.on_read_resource_requests) / len(self.on_read_resource_requests)
                    if self.on_read_resource_requests
                    else 0.0
                ),
                "on_get_prompt_requests": (
                    sum(self.on_get_prompt_requests) / len(self.on_get_prompt_requests)
                    if self.on_get_prompt_requests
                    else 0.0
                ),
                "on_list_tools_requests": (
                    sum(self.on_list_tools_requests) / len(self.on_list_tools_requests)
                    if self.on_list_tools_requests
                    else 0.0
                ),
                "on_list_resources_requests": (
                    sum(self.on_list_resources_requests) / len(self.on_list_resources_requests)
                    if self.on_list_resources_requests
                    else 0.0
                ),
                "on_list_resource_templates_requests": (
                    sum(self.on_list_resource_templates_requests)
                    / len(self.on_list_resource_templates_requests)
                    if self.on_list_resource_templates_requests
                    else 0.0
                ),
                "on_list_prompts_requests": (
                    sum(self.on_list_prompts_requests) / len(self.on_list_prompts_requests)
                    if self.on_list_prompts_requests
                    else 0.0
                ),
            },
            "counts": {
                "on_call_tool_requests": len(self.on_call_tool_requests),
                "on_read_resource_requests": len(self.on_read_resource_requests),
                "on_get_prompt_requests": len(self.on_get_prompt_requests),
                "on_list_tools_requests": len(self.on_list_tools_requests),
                "on_list_resources_requests": len(self.on_list_resources_requests),
                "on_list_resource_templates_requests": len(
                    self.on_list_resource_templates_requests
                ),
                "on_list_prompts_requests": len(self.on_list_prompts_requests),
            },
            "lows": {
                "on_call_tool_requests": min(self.on_call_tool_requests)
                if self.on_call_tool_requests
                else 0.0,
                "on_read_resource_requests": min(self.on_read_resource_requests)
                if self.on_read_resource_requests
                else 0.0,
                "on_get_prompt_requests": min(self.on_get_prompt_requests)
                if self.on_get_prompt_requests
                else 0.0,
                "on_list_tools_requests": min(self.on_list_tools_requests)
                if self.on_list_tools_requests
                else 0.0,
                "on_list_resources_requests": min(self.on_list_resources_requests)
                if self.on_list_resources_requests
                else 0.0,
                "on_list_resource_templates_requests": min(self.on_list_resource_templates_requests)
                if self.on_list_resource_templates_requests
                else 0.0,
                "on_list_prompts_requests": min(self.on_list_prompts_requests)
                if self.on_list_prompts_requests
                else 0.0,
            },
            "medians": {
                "on_call_tool_requests": statistics.median(self.on_call_tool_requests)
                if self.on_call_tool_requests
                else 0.0,
                "on_read_resource_requests": statistics.median(self.on_read_resource_requests)
                if self.on_read_resource_requests
                else 0.0,
                "on_get_prompt_requests": statistics.median(self.on_get_prompt_requests)
                if self.on_get_prompt_requests
                else 0.0,
                "on_list_tools_requests": statistics.median(self.on_list_tools_requests)
                if self.on_list_tools_requests
                else 0.0,
                "on_list_resources_requests": statistics.median(self.on_list_resources_requests)
                if self.on_list_resources_requests
                else 0.0,
                "on_list_resource_templates_requests": statistics.median(
                    self.on_list_resource_templates_requests
                )
                if self.on_list_resource_templates_requests
                else 0.0,
                "on_list_prompts_requests": statistics.median(self.on_list_prompts_requests)
                if self.on_list_prompts_requests
                else 0.0,
            },
            "highs": {
                "on_call_tool_requests": max(self.on_call_tool_requests)
                if self.on_call_tool_requests
                else 0.0,
                "on_read_resource_requests": max(self.on_read_resource_requests)
                if self.on_read_resource_requests
                else 0.0,
                "on_get_prompt_requests": max(self.on_get_prompt_requests)
                if self.on_get_prompt_requests
                else 0.0,
                "on_list_tools_requests": max(self.on_list_tools_requests)
                if self.on_list_tools_requests
                else 0.0,
                "on_list_resources_requests": max(self.on_list_resources_requests)
                if self.on_list_resources_requests
                else 0.0,
                "on_list_resource_templates_requests": max(self.on_list_resource_templates_requests)
                if self.on_list_resource_templates_requests
                else 0.0,
                "on_list_prompts_requests": max(self.on_list_prompts_requests)
                if self.on_list_prompts_requests
                else 0.0,
            },
        }


@dataclass(
    config=ConfigDict(extra="forbid", json_schema_extra={"noTelemetryProps": ["unique_files"]})
)
class _LanguageStatistics:
    """Statistics for a specific language within a category."""

    language: Annotated[
        str,
        Field(description="Lower case string name for the language, e.g. 'python', 'javascript'."),
    ]
    indexed: Annotated[
        NonNegativeInt, Field(description="Number of files indexed for this language.")
    ] = 0
    retrieved: Annotated[
        NonNegativeInt, Field(description="Number of files retrieved for this language.")
    ] = 0
    processed: Annotated[
        NonNegativeInt, Field(description="Number of files processed for this language.")
    ] = 0
    reindexed: Annotated[
        NonNegativeInt, Field(description="Number of files reindexed for this language.")
    ] = 0
    skipped: Annotated[
        NonNegativeInt, Field(description="Number of files skipped for this language.")
    ] = 0
    unique_files: ClassVar[
        Annotated[set[Path], Field(default_factory=set, init=False, repr=False, exclude=True)]
    ] = set()

    @property
    def unique_count(self) -> NonNegativeInt:
        """Get the number of unique files for this language (excluding skipped)."""
        return len(self.unique_files) if self.unique_files else 0

    @property
    def total_operations(self) -> NonNegativeInt:
        """Get the total number of operations for this language."""
        return self.indexed + self.retrieved + self.processed + self.reindexed + self.skipped

    def add_operation(self, operation: OperationsKey, path: Path | None = None) -> None:
        """Add an operation count and optionally track the file."""
        if operation == "indexed":
            self.indexed += 1
        elif operation == "retrieved":
            self.retrieved += 1
        elif operation == "processed":
            self.processed += 1
        elif operation == "reindexed":
            self.reindexed += 1
        elif operation == "skipped":
            self.skipped += 1

        # Track unique files (except for skipped operations)
        if path and path.is_file() and operation != "skipped":
            self.unique_files.add(path)


LanguageSummary = dict[OperationsKey | SummaryKey, NonNegativeInt]


@cache
def to_semantic_lang(lang: str) -> SemanticSearchLanguage | None:
    """Convert a language string to a SemanticSearchLanguage."""
    with contextlib.suppress(KeyError):
        return SemanticSearchLanguage.from_string(lang)


@dataclass(
    config=ConfigDict(extra="forbid", json_schema_extra={"noTelemetryProps": ["unique_files"]})
)
class _CategoryStatistics:
    """Statistics for a file category (code, config, docs, other)."""

    category: Annotated[
        CategoryKey, Field(description="The category of the files (code, config, docs, other).")
    ]
    languages: Annotated[
        dict[str, _LanguageStatistics],
        Field(
            default_factory=dict,
            description="Language statistics in this category. Keys are language names, values are _LanguageStatistics objects.",
        ),
    ]

    def get_language_stats(self, language: str) -> _LanguageStatistics:
        """Get or create language statistics for this category."""
        if language not in self.languages:
            self.languages[language] = _LanguageStatistics(language=language)
        return self.languages[language]

    @property
    def unique_count(self) -> NonNegativeInt:
        """Get the total unique file count across all languages in this category."""
        all_files: set[Path] = set()
        for lang_stats in self.languages.values():
            all_files.update(lang_stats.unique_files)
        return len(all_files)

    @property
    def _semantic_languages(self) -> frozenset[SemanticSearchLanguage]:
        """Get all semantic search languages in this category."""
        if (
            semantic_languages := [
                to_semantic_lang(lang) for lang in self.languages if to_semantic_lang(lang)
            ]
        ) and (filtered_languages := frozenset(filter(None, semantic_languages))):
            return filtered_languages
        return frozenset()

    @property
    def operations_with_semantic_support(self) -> NonNegativeInt:
        """Get the total operations with semantic support across all languages in this category."""
        return sum(
            lang_stats.total_operations
            for lang_stats in self.languages.values()
            if lang_stats.language in SemanticSearchLanguage.values()
        )

    @property
    def unique_files(self) -> frozenset[Path]:
        """Get the unique files across all languages in this category."""
        all_files: set[Path] = set()
        for lang_stats in self.languages.values():
            all_files.update(lang_stats.unique_files)
        return frozenset(all_files)

    @property
    def total_operations(self) -> NonNegativeInt:
        """Get the total operations across all languages in this category."""
        return sum(lang_stats.total_operations for lang_stats in self.languages.values())

    def add_operation(
        self, language: str, operation: OperationsKey, path: Path | None = None
    ) -> None:
        """Add an operation for a specific language in this category."""
        lang_stats = self.get_language_stats(language)
        lang_stats.add_operation(operation, path)


@dataclass(
    config=ConfigDict(extra="forbid", json_schema_extra={"noTelemetryProps": ["_other_files"]})
)
class FileStatistics:
    """Comprehensive file statistics tracking categories, languages, and operations."""

    categories: dict[CategoryKey, _CategoryStatistics] = Field(
        default_factory=lambda: {
            "code": _CategoryStatistics(category="code", languages={}),
            "config": _CategoryStatistics(category="config", languages={}),
            "docs": _CategoryStatistics(category="docs", languages={}),
            "other": _CategoryStatistics(category="other", languages={}),
        }
    )

    # TODO: This needs to come from the config; it consists of any optional includes the user sets
    _other_files: ClassVar[
        Annotated[set[Path], Field(default_factory=set, init=False, repr=False, exclude=True)]
    ] = set()

    def add_file(self, path: Path, operation: OperationsKey) -> None:
        """Add a file operation, automatically categorizing by extension."""
        if not path.is_file():
            raise ValueError(f"{path} is not a valid file")

        # Use ExtKind to determine file category and language
        if ext_kind := ExtKind.from_file(path):
            # Convert ExtKind.kind to our CategoryKey
            category_map = {"code": "code", "config": "config", "docs": "docs", "other": "other"}
            category = category_map.get(ext_kind.kind.value, "other")
            language = str(ext_kind.language)
            self.categories[cast(CategoryKey, category)].add_operation(language, operation, path)
        elif self._other_files and path in self._other_files:
            # Handle explicitly added "other" files
            language_name = f".{path.stem}" if "." in path.name else path.name
            self.categories[cast(CategoryKey, "other")].add_operation(
                language_name, operation, path
            )

    def add_other_files(self, *files: Path) -> None:
        """Add files to the 'other' category."""
        # TODO: We'd ideally want to make sure these are pushed to the indexer, unless we receive these in the same action
        self._other_files.update(files)

    @property
    def total_unique_files(self) -> NonNegativeInt:
        """Get the total unique files across all categories."""
        all_files: set[Path] = set()
        for category_stats in self.categories.values():
            for lang_stats in category_stats.languages.values():
                all_files.update(lang_stats.unique_files)
        return len(all_files)

    @property
    def total_operations(self) -> NonNegativeInt:
        """Get the total operations across all categories."""
        return sum(cat_stats.total_operations for cat_stats in self.categories.values())

    def get_summary_by_category(self) -> dict[CategoryKey, dict[str, NonNegativeInt]]:
        """Get a summary of unique files and operations by category."""
        return {
            category: {
                "unique_files": cat_stats.unique_count,
                "total_operations": cat_stats.total_operations,
                "languages": len(cat_stats.languages),
            }
            for category, cat_stats in self.categories.items()
        }

    def get_summary_by_language(self) -> MappingProxyType[str, LanguageSummary]:
        """Get a summary of statistics by language across all categories."""
        language_summary: dict[str, LanguageSummary] = defaultdict(
            lambda: {
                "unique_files": 0,
                "total_operations": 0,
                "indexed": 0,
                "retrieved": 0,
                "processed": 0,
                "reindexed": 0,
                "skipped": 0,
            }
        )

        all_files_by_language: dict[str, set[Path]] = defaultdict(set)

        for cat_stats in self.categories.values():
            for lang, lang_stats in cat_stats.languages.items():
                all_files_by_language[lang].update(lang_stats.unique_files)
                language_summary[lang]["unique_files"] += lang_stats.unique_count
                language_summary[lang] = self._summarize_stats_for_language(lang_stats)

        return MappingProxyType(language_summary)

    def _summarize_stats_for_language(self, lang_stats: _LanguageStatistics) -> LanguageSummary:
        """Summarize language statistics into the overall language summary."""
        return {
            "total_operations": lang_stats.total_operations,
            "indexed": lang_stats.indexed,
            "retrieved": lang_stats.retrieved,
            "processed": lang_stats.processed,
            "reindexed": lang_stats.reindexed,
            "skipped": lang_stats.skipped,
        }


class TokenCounter(
    Counter[
        Literal[
            "embedding_generated",
            "reranking_generated",
            "context_agent_used",
            "user_agent_received",
            "search_results",
        ]
    ]
):
    """A counter for tracking token usage by operation."""

    def __init__(self) -> None:
        super().__init__()
        self.update({
            "embedding_generated": 0,
            "reranking_generated": 0,
            "context_agent_used": 0,
            "user_agent_received": 0,
            "search_results": 0,
        })

    @property
    def total_generated(self) -> NonNegativeInt:
        """Get the total number of tokens generated across all operations."""
        return sum((self["embedding_generated"], self["reranking_generated"]))

    @property
    def total_used(self) -> NonNegativeInt:
        """Get the total number of tokens used across all operations."""
        return sum((self["context_agent_used"], self["user_agent_received"]))

    @property
    def context_saved(self) -> NonNegativeInt:
        """
        Get the total number of tokens that CodeWeaver saved from the user_agent.

        !!! note
            The number returned by `context_saved` is a low estimate of the actual number of tokens saved.

            CodeWeaver doesn't have access to the full context of the user's agent's request. To get the full picture we would need:

            - The total tokens used by the user's agent after CodeWeaver's response
            - The number of 'turns' it took for the user agent to complete the task *after* CodeWeaver's response

            Even if we had those numbers, they would still be lower bounds, because they don't account for increases in overall turns and token expenditure if CodeWeaver was never used. Let's call this the "blind bumbling savings" of CodeWeaver.
        """
        return self["search_results"] - self["user_agent_received"]

    @property
    def money_saved(self) -> NonNegativeFloat:
        """
        Estimate the money saved by using CodeWeaver based on token savings.

        TODO: To implement this correctly, we need to pull the model name from the fastmcp.Context object and use the pricing for that model. We could use [`genai_prices`](https://github.com/pydantic/genai-prices) to get the pricing information, either remotely or as a dependency.
        """
        raise NotImplementedError("Money saved estimation is not implemented yet.")


@dataclass(
    kw_only=True,
    config=ConfigDict(extra="forbid", str_strip_whitespace=True, arbitrary_types_allowed=True),
)
class SessionStatistics:
    """Statistics for tracking session performance and usage."""

    total_requests: Annotated[
        NonNegativeInt | None, Field(description="Total requests made during the session.")
    ] = None
    successful_requests: Annotated[
        NonNegativeInt | None, Field(description="Total successful requests during the session.")
    ] = None
    failed_requests: Annotated[
        NonNegativeInt | None, Field(description="Total failed requests during the session.")
    ] = None
    timing_statistics: Annotated[
        TimingStatistics | None, Field(description="Timing statistics for the session.")
    ] = None

    index_statistics: Annotated[
        FileStatistics | None,
        Field(
            default_factory=FileStatistics,
            description="Comprehensive file statistics tracking categories, languages, and operations.",
        ),
    ] = None
    token_statistics: Annotated[
        TokenCounter | None,
        Field(
            default_factory=TokenCounter,
            description="A typed Counter that tracks token usage statistics.",
        ),
    ] = None

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if not self.token_statistics:
            self.token_statistics = TokenCounter()
        self.timing_statistics = TimingStatistics([], [], [], [], [], [], [])
        for field in ("total_requests", "successful_requests", "failed_requests"):
            if getattr(self, field) is None:
                setattr(self, field, 0)

    def get_timing_statistics(self) -> TimingRequestsDict:
        """Get the current timing statistics."""
        if self.timing_statistics:
            return self.timing_statistics.timing_summary()
        self.timing_statistics = TimingStatistics([], [], [], [], [], [], [])
        return {}

    def add_successful_request(self) -> None:
        """Add a successful request count."""
        self._add_request(successful=True)

    def add_failed_request(self) -> None:
        """Add a failed request count."""
        self._add_request(successful=False)

    def _add_request(self, *, successful: bool) -> None:
        """Internal method to add a request count."""
        if self.total_requests is None:
            self.total_requests = 0
        if self.successful_requests is None:
            self.successful_requests = 0
        if self.failed_requests is None:
            self.failed_requests = 0

        self.total_requests += 1
        if successful:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def get_success_rate(self) -> NonNegativeFloat | None:
        """Calculate the success rate of requests."""
        if self.total_requests and self.total_requests > 0:
            return (self.successful_requests or 0) / self.total_requests
        return None

    def add_token_usage(
        self,
        embedding_generated: NonNegativeInt = 0,
        reranking_generated: NonNegativeInt = 0,
        context_agent_used: NonNegativeInt = 0,
        user_agent_received: NonNegativeInt = 0,
        search_results: NonNegativeInt = 0,
    ) -> None:
        """Add token usage statistics."""
        if self.token_statistics is None:
            self.token_statistics = TokenCounter()

        self.token_statistics["embedding_generated"] += embedding_generated
        self.token_statistics["reranking_generated"] += reranking_generated
        self.token_statistics["context_agent_used"] += context_agent_used
        self.token_statistics["user_agent_received"] += user_agent_received
        self.token_statistics["search_results"] += search_results

    def get_token_usage(self) -> TokenCounter:
        """Get the current token usage statistics."""
        return self.token_statistics or TokenCounter()

    def add_file_operation(self, path: Path, operation: OperationsKey) -> None:
        """Add a file operation to the index statistics."""
        if not self.index_statistics:
            self.index_statistics = FileStatistics()
        self.index_statistics.add_file(path, operation)

    def add_file_operations(self, *file_operations: tuple[Path, OperationsKey]) -> None:
        """Add multiple file operations to the index statistics."""
        for file, operation in file_operations:
            self.add_file_operation(file, operation)

    def add_other_files(self, *files: Path) -> None:
        """Add files to the 'other' category in index statistics."""
        if not self.index_statistics:
            self.index_statistics = FileStatistics()
        self.index_statistics.add_other_files(*files)

    def reset(self) -> None:
        """Reset all statistics to their initial state."""
        self.total_requests = None
        self.successful_requests = None
        self.failed_requests = None
        self._response_times = None
        self.index_statistics = FileStatistics()
        self.token_statistics = TokenCounter()
