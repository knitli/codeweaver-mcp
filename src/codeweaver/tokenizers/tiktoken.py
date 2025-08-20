# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Tokenizer implementation using the Tiktoken library."""

import tiktoken

from codeweaver.tokenizers.base import Tokenizer


class TiktokenTokenizer(Tokenizer[tiktoken.Encoding]):
    """Tokenizer for the Tiktoken library.

    Inherits `estimate` and `estimate_batch` from its base class, `Tokenizer`.
    """

    _encoder: tiktoken.Encoding

    def __init__(self, encoder: str) -> None:
        """Initialize tiktoken encoder."""
        self._encoder = tiktoken.get_encoding(encoder)
        self.encoder_name = self._encoder.name

    def encode(self, text: str | bytes) -> list[int]:
        """Encode text into a list of token IDs."""
        return self._encoder.encode(self._to_string(text))

    def encode_batch(self, texts: list[str | bytes]) -> list[list[int]]:
        """Encode a batch of texts into a list of token ID lists."""
        return self._encoder.encode_batch([self._to_string(text) for text in texts])

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into text."""
        return self._encoder.decode(tokens)

    def decode_batch(self, token_lists: list[list[int]]) -> list[str]:
        """Decode a batch of token ID lists back into texts."""
        return self._encoder.decode_batch(token_lists)

    @staticmethod
    def encoders() -> list[str]:
        """List all available encoder names."""
        return tiktoken.list_encoding_names()
