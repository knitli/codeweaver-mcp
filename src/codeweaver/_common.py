# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""A foundational enum class for the CodeWeaver project for common functionality."""

from collections.abc import Generator
from typing import Self

from aenum import UniqueEnum, extend_enum  # type: ignore


class BaseEnum(UniqueEnum):
    """An enum class that provides common functionality for all enums in the CodeWeaver project. Enum members must be unique and either all strings or all integers.

    BaseEnum extends [`aenum.UniqueEnum`][aenum.UniqueEnum] to ensure that all enum members are unique. `aenum` allows us to dynamically add members, such as for plugin systems.

    BaseEnum provides convenience methods for converting between strings and enum members, checking membership, and retrieving members and members' values, and adding new members dynamically.
    """

    @classmethod
    def from_string(cls, value: str) -> Self:
        """Convert a string to the corresponding enum member."""
        try:
            if cls._value_type() is int:
                return cls(int(value))
            return cls.__members__[value.upper().replace("-", "_").replace(" ", "_")]
        except KeyError:
            raise ValueError(f"{value} is not a valid {cls.__qualname__} value") from None

    @classmethod
    def _value_type(cls) -> type[str] | type[int]:
        """Return the type of the enum values."""
        if all(isinstance(member.value, str) for member in cls.__members__.values()):
            return str
        if all(isinstance(member.value, int) for member in cls.__members__.values()):
            return int
        raise TypeError(
            f"All members of {cls.__qualname__} must have the same value type and must be either str or int."
        )

    @classmethod
    def is_member(cls, value: str | int) -> bool:
        """Check if a value is a member of the enum."""
        return bool(cls.get(value, None))

    @property
    def value_type(self) -> type:
        """Return the type of the enum member's value."""
        return type(self)._value_type()

    @property
    def as_variable(self) -> str:
        """Return the string representation of the enum member as a variable name."""
        return self.name.lower()

    @classmethod
    def members(cls) -> Generator[type["BaseEnum"]]:
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
    def members_to_values(cls) -> dict[type["BaseEnum"], str | int]:
        """Return a dictionary mapping member names to their values."""
        return {member: member.value for member in cls.members()}

    @classmethod
    def add_member(cls, name: str, value: str | int) -> type["BaseEnum"]:
        """Dynamically add a new member to the enum."""
        return extend_enum(cls, name.upper().replace("-", "_").replace(" ", "_"), value)
