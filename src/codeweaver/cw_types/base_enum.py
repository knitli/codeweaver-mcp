# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
A foundational enum class for the CodeWeaver project for common functionality.
"""

import enum


@enum.unique
class BaseEnum(enum.Enum):
    """Base class for all enums in the codebase."""

    @classmethod
    def from_string(cls, value: str) -> "BaseEnum":
        """Convert a string to the corresponding enum member."""
        try:
            return cls[value.strip().upper().replace("-", "_").replace(" ", "_")]
        except KeyError:
            raise ValueError(f"{value} is not a valid {cls.__name__} value") from None

    @classmethod
    def members(cls) -> tuple["BaseEnum", ...]:
        """Return all members of the enum as a tuple."""
        return tuple(cls.__members__.values())

    @classmethod
    def get_values(cls) -> tuple[str, ...]:
        """Return all enum member names as a tuple."""
        return tuple(member.value for member in cls.members())

    def __str__(self) -> str:
        """Return the string representation of the enum member."""
        return self.name.replace("_", " ").lower()

    @classmethod
    def members_to_values(cls) -> dict["BaseEnum", str]:
        """Return a dictionary mapping member names to their values."""
        return {member: member.value for member in cls.members()}
