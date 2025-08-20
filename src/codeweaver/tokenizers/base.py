# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Abstract base class for tokenizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import LiteralString


type EncoderName = LiteralString


class Tokenizer[Encoder](ABC):
    """Abstract base class for tokenizers."""

    _encoder: Encoder

    @abstractmethod
    def __init__(self, encoder: EncoderName) -> None:
        """Initialize the tokenizer with a specific encoder."""

    @abstractmethod
    def encode(self, text: str | bytes) -> list[int]:
        """Encode text into a list of token IDs."""

    @abstractmethod
    def encode_batch(self, texts: list[str | bytes]) -> list[list[int]]:
        """Encode a batch of texts into a list of token ID lists."""

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into text."""

    @abstractmethod
    def decode_batch(self, token_lists: list[list[int]]) -> list[str]:
        """Decode a batch of token ID lists back into text."""

    @staticmethod
    @abstractmethod
    def encoders() -> list[str]:
        """List all available encoder names."""

    @property
    def encoder(self) -> Encoder:
        """Get the encoder instance."""
        return self._encoder

    def _to_string(self, value: str | bytes) -> str:
        """Convert bytes to string if necessary."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return value

    def estimate(self, text: str | bytes) -> int:
        """Estimate the number of tokens in the given text."""
        return len(self.encode(text))

    def estimate_batch(self, texts: list[str | bytes]) -> int:
        """Estimate the number of tokens in a batch of texts."""
        return sum(len(batch) for batch in self.encode_batch(texts))
