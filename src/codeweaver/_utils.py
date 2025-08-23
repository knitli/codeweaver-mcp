# sourcery skip: avoid-global-variables
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Helper functions for CodeWeaver utilities.
"""

import contextlib
import os

from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from types import FunctionType, MethodType
from typing import Any, LiteralString


# SPDX-BeginSnippet
# SPDX-FileCopyrightText: Copyright (c) 2012-2020, Alexander Schepanovski
# SPDX-License-Identifier: MIT


def rpartial(
    func: FunctionType | MethodType, *args: Any, **kwargs: dict[str, Any]
) -> Callable[..., Any]:
    """Partially applies last arguments, returning a callable. Adapted from `funcy`."""
    return lambda *a, **kw: func(*(a + args), **dict(kwargs, **kw))


# SPDX-EndSnippet


def walk_down_to_git_root(path: Path | None = None) -> Path:
    """Walk up the directory tree until a .git directory is found."""
    if path is None:
        path = Path.cwd()
    if path.is_file():
        path = path.parent
    while path != path.parent:
        if (path / ".git").is_dir():
            return path
        path = path.parent
    raise FileNotFoundError("No .git directory found in the path hierarchy.")


def in_codeweaver_clone(path: Path) -> bool:
    """Check if the current repo is CodeWeaver."""
    return "codeweaver" in str(path).lower() or "code-weaver" in str(path).lower()


def estimate_tokens(text: str | bytes, encoder: str = "cl100k_base") -> int:
    """Estimate the number of tokens in a text."""
    import tiktoken

    encoding = tiktoken.get_encoding(encoder)
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    return len(encoding.encode(text))


def estimate_voyage_tokens(
    text: str | bytes | Sequence[str | bytes], model: LiteralString | None = None
) -> int:
    """Estimate the number of tokens for VoyageAI models."""
    if isinstance(text, str | bytes):
        text = [text if isinstance(text, str) else text.decode("utf-8", errors="ignore")]
    try:
        from transformers import AutoTokenizer  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model)  # type: ignore
        return len(tokenizer.encode(text))  # type: ignore
    except ImportError:
        result = None
        if key := os.environ.get(
            "VOYAGEAI_API_KEY", os.environ.get("CW_EMBEDDING_PROVIDER__API_KEY")
        ):
            with contextlib.suppress(Exception):
                import asyncio

                from voyageai.client_async import AsyncClient

                client = AsyncClient(api_key=key, max_retries=3, timeout=10)
                result = asyncio.run(client.count_tokens([text]))  # type: ignore
            if result and isinstance(result, int):
                return result

    return round(
        sum(
            estimate_tokens(txt, "cl100k_base") * 1.3 for txt in text
        )  # rough estimate -- voyageai's tokenizer tends to create about 30% more tokens
    )


@cache
def normalize_ext(ext: str) -> str:
    """Normalize a file extension to a standard format."""
    return ext.lower().strip() if ext.startswith(".") else f".{ext.lower().strip()}"
