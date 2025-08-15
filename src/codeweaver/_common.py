# sourcery skip: snake-case-variable-declarations
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""A foundational enum class for the CodeWeaver project for common functionality."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from enum import Enum, unique
from typing import Self, cast

from aenum import extend_enum  # type: ignore


@unique
class BaseEnum(Enum):
    """An enum class that provides common functionality for all enums in the CodeWeaver project. Enum members must be unique and either all strings or all integers.

    `aenum.extend_enum` allows us to dynamically add members, such as for plugin systems.

    BaseEnum provides convenience methods for converting between strings and enum members, checking membership, and retrieving members and members' values, and adding new members dynamically.
    """

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Convert a string to the corresponding enum member."""
        try:
            if cls._value_type() is int and value.isdigit():
                return cls(int(value))
            normalized_value = value.replace("-", "_").replace(" ", "_").upper()
            cls.__members__: dict[str, type[BaseEnum]]  # type: ignore  # noqa: B032
            return cast(Self, cls.__members__[normalized_value])
        except KeyError:
            try:
                # try without any dividers
                normalized_value = value.replace("-", "").replace(" ", "").upper()
                return cast(Self, cls.__members__[normalized_value])
            except KeyError as e:
                raise ValueError(f"{value} is not a valid {cls.__qualname__} value") from e

    @classmethod
    def _value_type(cls) -> type[int | str]:
        """Return the type of the enum values."""
        if all(isinstance(member.value, str) for member in cls.__members__.values() if member):
            return str
        if all(
            isinstance(member.value, int)
            for member in cls.__members__.values()
            if member and member.value
        ):
            return int
        raise TypeError(
            f"All members of {cls.__qualname__} must have the same value type and must be either str or int."
        )

    @classmethod
    def __iter__(cls) -> Iterator[Self]:
        """Return an iterator over the enum members."""
        yield from cls.__members__.values()

    @classmethod
    def is_member(cls, value: str | int) -> bool:
        """Check if a value is a member of the enum."""
        return value in cls.values()

    @property
    def value_type(self) -> type[int | str]:
        """Return the type of the enum member's value."""
        return type(self)._value_type()

    @property
    def as_variable(self) -> str:
        """Return the string representation of the enum member as a variable name."""
        return self.name.lower()

    @classmethod
    def members(cls) -> Generator[Self]:
        """Return all members of the enum as a tuple."""
        yield from cls.__members__.values()

    @classmethod
    def values(cls) -> Generator[str | int]:
        """Return all enum member names as a tuple."""
        yield from (member.value for member in cls.members())

    def __str__(self) -> str:
        """Return the string representation of the enum member."""
        return self.name.replace("_", " ").lower()

    @classmethod
    def members_to_values(cls) -> dict[Self, str | int]:
        """Return a dictionary mapping member names to their values."""
        return {member: member.value for member in cls.members()}

    @classmethod
    def add_member(cls, name: str, value: str | int) -> Self:
        """Dynamically add a new member to the enum."""
        return extend_enum(
            cls,  # type: ignore -- aenum has no typing, but its EnumType and Enum are derived from stdlib Enum... the author of aenum *is* the stdlib Enum author
            name.upper().replace("-", "_").replace(" ", "_"),
            value,  # type: ignore
        )
