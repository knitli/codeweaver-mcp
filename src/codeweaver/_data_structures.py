# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""A foundational enum class for the CodeWeaver project for common functionality."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Annotated, Any, Self, TypeGuard
from uuid import uuid4

from pydantic import UUID4, Field, NonNegativeInt, model_validator
from pydantic.dataclasses import dataclass


SpanTuple = tuple[
    NonNegativeInt, NonNegativeInt, UUID4
]  # Type alias for a span tuple (start, end, source_id)


@dataclass(frozen=True, slots=True)
class Span:
    """
    An immutable span of lines in a file, defined by a start and end line number, and a source identifier.

    `Span`s are a big part of CodeWeaver's foundational data structures, so they have a robust API for manipulation and comparison.
    `Span` supports intersection, union, difference, and symmetric difference operations (both by using operators and methods), as well as containment checks and subset/superset checks.

    All spans have an identifier, that they should share with the source (e.g. file) of the span. This allows for operations between spans from different sources to be safely handled, as they will not interfere with each other.

    While we want it to be intuitive to us, it might not be intuitive for everyone, so let's break down the key features and caveats:

        - Spans are inclusive of their start and end lines.
        - Operations between spans from different sources will not interfere with each other (but can return None in some cases, so get your null checks ready).
        - All spans have an identifier that they should share with the source (e.g. file) of the span.
        - Spans are immutable and cannot be modified after creation, but you can easily create new spans based on existing ones, including with a Span.
        - Spans are iterable, *and iterate over the lines*. If you want to iterate over the attributes, use `span.as_tuple`.
        - Operations:

            - `in` can test if an integer (i.e. line number), a tuple (i.e. (start, end)), or another `Span` is contained within the span.
              - For line numbers and tuples, it only checks the start and end lines, not the source identifier.
              - For spans, or tuples with a source identifier, it checks if the span overlaps with the current span in the source.
            - `difference` (`-`) operation can return a single span (if the difference is contiguous), a tuple of spans (if the other span is fully contained), or None (if the spans do not overlap).
            - `union` (`|`) operation will return a new span that covers both spans, but only if they share the same source identifier. Otherwise returns the original span.
            - `intersection` (`&`) operation will return a new span that is the overlap of both spans, or None if they do not overlap.
            - `symmetric_difference` (`^`) operation will return a tuple of spans that are the parts of each span that do not overlap with the other, or None if they do not overlap.
            - `equals` (`==`) operation will return True if the spans are equal, and False otherwise. Spans are only equal if they have the same start, end, **and** source identifier.
    """

    start: NonNegativeInt
    end: NonNegativeInt

    _source_id: Annotated[
        UUID4,
        Field(
            default_factory=uuid4,
            description="The identifier for the span's source, such as a file.",
            repr=True,
            init=True,
            alias="source_id",
            exclude=False,
        ),
    ] = uuid4()  # Unique identifier for the source of the span

    def __hash__(self):
        return hash((self.start, self.end, self._source_id))

    def __str__(self) -> str:
        """Return a string representation of the span."""
        return f"lines {self.start}-{self.end} (source: {self._source_id})"

    def __repr__(self):
        """Return a string representation of the span."""
        return f"Span({self.start}, {self.end}, {self._source_id})"

    def __or__(self, other: Span) -> Span:  # Union
        """Return the union of two spans."""
        if self._source_id != other._source_id:
            return self
        return Span(min(self.start, other.start), max(self.end, other.end))

    def __and__(self, other: Span) -> Span | None:  # Intersection
        """Return the intersection between two spans."""
        if self._source_id != other._source_id:
            return None
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return Span(start, end) if start <= end else None

    def __sub__(self, other: Span) -> Span | tuple[Span, Span] | None:  # Difference
        """Return the difference between two spans."""
        if self._source_id != other._source_id:
            return self  # Cannot subtract spans from different sources

        if self.end < other.start or self.start > other.end:
            return self  # No overlap
        if other.start <= self.start and other.end >= self.end:
            return None  # Fully covered
        if other.start > self.start and other.end < self.end:
            return (Span(self.start, other.start - 1), Span(other.end + 1, self.end))
        if other.start <= self.start:
            return Span(other.end + 1, self.end) if other.end < self.end else None
        return Span(self.start, other.start - 1) if other.start > self.start else None

    def __xor__(self, other: Span) -> tuple[Span, ...] | None:  # Symmetric Difference
        """Return the symmetric difference between two spans."""
        if self._source_id != other._source_id or not self & other:
            return (self, other)
        diff1 = self - other
        diff2 = other - self
        result: list[Span] = []
        if diff1:
            result.extend(diff1 if isinstance(diff1, tuple) else [diff1])
        if diff2:
            result.extend(diff2 if isinstance(diff2, tuple) else [diff2])
        return tuple(result) if result else None

    def __le__(self, other: Span) -> bool:  # Subset
        if self._source_id != other._source_id:
            return False
        return self.start >= other.start and self.end <= other.end

    def __ge__(self, other: Span) -> bool:  # Superset
        if self._source_id != other._source_id:
            return False
        return self.start <= other.start and self.end >= other.end

    def __eq__(self, other: object) -> bool:  # Equality
        if not isinstance(other, Span) or self._source_id != other._source_id:
            return False
        return self.start == other.start and self.end == other.end

    def __iter__(self) -> Iterator[NonNegativeInt]:
        """Return an iterator *over the lines* in the span."""
        current = self.start
        while current <= self.end:
            yield current
            current += 1

    def __len__(self) -> NonNegativeInt:
        """Return the number of lines in the span."""
        return self.end - self.start + 1

    def __contains__(self, span: Span | SpanTuple | tuple[int, int] | int) -> bool:
        """
        Check if the span contains a line number or another span or a tuple of (start, end).

        This is naive for tuples and line numbers, but does consider the source for span comparisons.
        """
        if isinstance(span, tuple):
            if len(span) == 2 or (len(span) == 3 and span[2] is None):  # type: ignore
                start, end = span
                return self.start <= start <= self.end or self.start <= end <= self.end
            return bool(self & Span.from_tuple(span))
        if isinstance(span, Span):
            return bool(self & span)
        return self.start <= span <= self.end

    @classmethod
    def __call__(cls, span: SpanTuple | Span | tuple[int, int, UUID4 | None]) -> Span:
        """Create a Span from a tuple of (start, end, source_id)."""
        if isinstance(span, Span):
            return cls(*span.as_tuple)
        start, end, source_id = span
        return cls(start=start, end=end, _source_id=source_id or uuid4())

    @classmethod
    def from_tuple(cls, span: SpanTuple) -> Span:
        """Create a Span from a tuple of (start, end, source_id)."""
        return cls(*span)

    @property
    def as_tuple(self) -> SpanTuple:
        """Return the span as a tuple of (start, end, source_id)."""
        return (self.start, self.end, self._source_id)

    @property
    def source_id(self) -> UUID4:
        """The identifier for the source of the span."""
        return self._source_id

    def from_sourced_lines(self, start: NonNegativeInt, end: NonNegativeInt) -> Span:
        """Create a Span for the same source as this one, but with a new start and end."""
        return Span(start=start, end=end, _source_id=self._source_id)

    @model_validator(mode="after")
    def check_span(self) -> Span:
        """Ensure that the start is less than or equal to the end."""
        if self.start > self.end:
            raise ValueError("Start must be less than or equal to end")
        return self

    def union(self, other: Span) -> Span:
        """Combine this span with another span."""
        return self | other

    def intersection(self, other: Span) -> Span | None:
        """Return the intersection of this span with another span."""
        return None if self._source_id != other._source_id else self & other

    def difference(self, other: Span) -> Span | tuple[Span, Span] | None:
        """
        Return the difference between this span and another span.

        If the spans don't overlap,
        """
        return self - other

    def symmetric_difference(self, other: Span) -> tuple[Span, ...] | None:
        """Return the symmetric difference between this span and another span."""
        return self ^ other

    def contains_line(self, line: int) -> bool:
        """
        Check if this span contains a specific line.

        Note: This is a naive check that assumes the line is from the same source.

        """
        return line in self

    def is_subset(self, other: Span) -> bool:
        """Check if this span is a subset of another span."""
        return self.start >= other.start and self.end <= other.end

    def is_superset(self, other: Span) -> bool:
        """Check if this span is a superset of another span."""
        return self.start <= other.start and self.end >= other.end

    def is_adjacent(self, other: Span) -> bool:
        """Check if this span is adjacent to another span."""
        return (
            self.end == other.start
            or self.start == other.end
            or self.end + 1 == other.start
            or self.start - 1 == other.end
        )


@dataclass
class SpanGroup:
    """A group of spans that can be manipulated as a single unit."""

    spans: Annotated[
        set[Span],
        Field(
            default_factory=set, description="A set of spans that can be manipulated as a group."
        ),
    ]

    def __post_init__(self):
        self.spans = self.spans or set()
        self._normalize()

    @classmethod
    def from_simple_spans(cls, simple_spans: Sequence[tuple[int, int]]) -> SpanGroup:
        """
        Create a SpanGroup from a sequence of simple spans. Assumes all input spans are from the same source.

        Intended for ingestion, where a parser identifies spans as simple tuples of (start, end) from a single source/file, and passes them for grouping into a SpanGroup.
        """
        source_id = uuid4()  # Default source_id for the group
        spans = {Span(start, end, source_id) for start, end in simple_spans}
        return cls(spans)

    def _ensure_set(self, spans: Sequence[Any]) -> TypeGuard[set[Span]]:
        """Ensure that spans is a set of Span objects."""
        return bool(spans and isinstance(spans, set) and all(isinstance(s, Span) for s in spans))

    def _normalize(self) -> None:
        """Merge overlapping/adjacent spans with same source_id."""
        normalized: list[Span] = []
        for span in sorted(self.spans, key=lambda s: (s.source_id, s.start)):
            if normalized and normalized[-1] & span:
                normalized[-1] |= span
            else:
                normalized.append(span)
        self.spans = set(normalized)

    # ---- Set-like operators ----
    def __or__(self, other: Self) -> SpanGroup:  # Union
        """Return a new SpanGroup that is the union of this group and another."""
        return SpanGroup(self.spans | other.spans)

    def __and__(self, other: Self) -> SpanGroup:  # Intersection
        intersected = {
            s1 & s2
            for s1 in self.spans
            for s2 in other.spans
            if s1.source_id == s2.source_id and (s1 & s2) is not None
        }
        return SpanGroup({s for s in intersected if s})

    def __sub__(self, other: Self) -> SpanGroup:  # Difference
        """Return a new SpanGroup that is the difference between this group and another."""
        result: set[Span] = set()
        for s1 in self.spans:
            leftovers = [s1]
            for s2 in other.spans:
                if s1.source_id == s2.source_id:
                    new_leftovers: list[Span] = []
                    for lf in leftovers:
                        if diff := lf - s2:
                            new_leftovers.extend(diff if isinstance(diff, tuple) else [diff])
                    leftovers = new_leftovers
            result.update(leftovers)
        return SpanGroup({r for r in result if r})

    def __xor__(self, other: Self) -> SpanGroup:  # Symmetric difference
        """Return a new SpanGroup that is the symmetric difference between this group and another."""
        return (self - other) | (other - self)

    # ---- Utility ----
    def add(self, span: Span) -> Self:
        """Add a span to the group."""
        self.spans.add(span)
        self._normalize()
        return self

    def __iter__(self) -> Iterator[Span]:
        """Iterate over the spans in the group, sorted by source_id and start line."""
        yield from sorted(self.spans, key=lambda s: (s.source_id, s.start))

    def __len__(self) -> int:
        return len(self.spans)

    def __repr__(self) -> str:
        return f"SpanGroup({list(self)})"
