# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Commons types for Backend sources (like vector databases).
"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from codeweaver.types.backends.enums import DistanceMetric, FilterOperator, IndexType, StorageType


class VectorPoint(BaseModel):
    """Universal vector point structure compatible with all backends."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    id: Annotated[str | int, Field(description="Unique identifier for the vector point")]
    vector: Annotated[
        list[float], Field(description="Dense vector representation", min_length=1, max_length=4096)
    ]
    payload: Annotated[
        dict[str, Any] | None, Field(default=None, description="Metadata payload for the vector")
    ]
    sparse_vector: Annotated[
        dict[int, float] | None, Field(default=None, description="Sparse vector for hybrid search")
    ]


class SearchResult(BaseModel):
    """Unified search result format across all vector databases."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    id: Annotated[str | int, Field(description="Unique identifier for the result")]
    score: Annotated[float, Field(description="Similarity score for the result", ge=0.0, le=1.0)]
    payload: Annotated[
        dict[str, Any] | None, Field(default=None, description="Metadata payload from the vector")
    ]
    vector: Annotated[
        list[float] | None,
        Field(
            default=None,
            description="Original vector data (optional, depends on backend)",
            min_length=1,
            max_length=4096,
        ),
    ]
    backend_metadata: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Additional metadata for debugging/optimization"),
    ]


class CollectionInfo(BaseModel):
    """Collection metadata structure for introspection."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    name: Annotated[str, Field(description="Collection name", min_length=1, max_length=255)]
    dimension: Annotated[int, Field(description="Vector dimension size", ge=1, le=4096)]
    points_count: Annotated[int, Field(description="Number of points in the collection", ge=0)]
    distance_metric: Annotated[
        DistanceMetric, Field(description="Distance metric used for similarity calculation")
    ]
    indexed: Annotated[bool, Field(description="Whether the collection is indexed")]

    # Backend-specific capabilities
    supports_sparse_vectors: Annotated[
        bool, Field(default=False, description="Whether backend supports sparse vectors")
    ]
    supports_hybrid_search: Annotated[
        bool, Field(default=False, description="Whether backend supports hybrid search")
    ]
    supports_filtering: Annotated[
        bool, Field(default=True, description="Whether backend supports result filtering")
    ]
    supports_updates: Annotated[
        bool, Field(default=True, description="Whether backend supports vector updates")
    ]

    # Storage and performance info
    storage_type: Annotated[
        StorageType | None, Field(default=None, description="Storage type used by the collection")
    ]
    index_type: Annotated[
        IndexType | None, Field(default=None, description="Index type used by the collection")
    ]
    backend_info: Annotated[
        dict[str, Any] | None, Field(default=None, description="Backend-specific information")
    ]


class FilterCondition(BaseModel):
    """Universal filter condition for vector search."""

    model_config = ConfigDict(extra="allow", validate_assignment=True, arbitrary_types_allowed=True)

    field: Annotated[str, Field(description="Field name to filter on", min_length=1)]
    operator: Annotated[FilterOperator, Field(description="Filter operator to apply")]
    value: Annotated[Any, Field(description="Value to compare against")]


class SearchFilter(BaseModel):
    """Composite filter for vector search operations."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    conditions: Annotated[
        list[FilterCondition] | None,
        Field(default=None, description="Individual filter conditions"),
    ]
    must: Annotated[
        list["SearchFilter"] | None,
        Field(default=None, description="Filters that must match (AND logic)"),
    ]
    should: Annotated[
        list["SearchFilter"] | None,
        Field(default=None, description="Filters that should match (OR logic)"),
    ]
    must_not: Annotated[
        list["SearchFilter"] | None,
        Field(default=None, description="Filters that must not match (NOT logic)"),
    ]
