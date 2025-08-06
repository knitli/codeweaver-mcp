#!/usr/bin/env -S uv run
# ///script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Fetches tree-sitter grammars from their repos."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys

from functools import cache, cached_property
from itertools import starmap
from pathlib import Path
from typing import Any, NamedTuple, TypeGuard

import httpx


class GrammarRetrievalError(Exception):
    """Raised when grammar retrieval fails."""


CLIENT = httpx.AsyncClient()
SAVE_DIR = Path(__file__).parent.parent / "grammars"

GH_USERNAME = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GH_USERNAME", "bashandbone")
GH_TOKEN = (
    sys.argv[2]
    if len(sys.argv) > 2
    else os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", None))
)


AST_GREP_SUPPORTED = (
    "bash",
    "c",
    "cpp",
    "csharp",
    "css",
    "elixir",
    "go",
    "haskell",
    "html",
    "java",
    "javascript",
    "jsx",
    "json",
    "kotlin",
    "lua",
    "nix",
    "php",
    "python",
    "ruby",
    "rust",
    "scala",
    "solidity",
    "swift",
    "typescript",
    "tsx",
    "yaml",
)

TREE_SITTER_REPO = (  # all master branches in the tree-sitter org
    "bash",
    "c",
    "cpp",
    "c-sharp",
    "css",
    "go",
    "haskell",
    "html",
    "java",
    "javascript",
    "json",
    "lua",
    "python",
    "php",
    "ruby",
    "rust",
    "scala",
    "typescript",
)

OTHER_REPOS = (
    ("elixir", "elixir-lang/tree-sitter-elixir", "main"),
    ("kotlin", "tree-sitter-grammars/tree-sitter-kotlin", "master"),
    ("nix", "nix-community/tree-sitter-nix", "master"),
    ("solidity", "JoranHonig/tree-sitter-solidity", "master"),
    ("swift", "alex-pincus/tree-sitter-swift", "main"),
    ("yaml", "tree-sitter-grammars/tree-sitter-yaml", "master"),
)


@cache
def get_request_headers() -> dict[str, str]:
    """Returns a GitHub client using the token from environment variables."""
    default_headers = {
        "Accept": "application/vnd.github+json",
        "Accept-Encoding": "gzip, deflate, br",
        "User-Agent": "Codeweaver-MCP",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if not GH_TOKEN:
        # If no token, we can use the public API but may hit rate limits.
        print("No GitHub token provided, using public API with limited access.")
        print("Set GH_USERNAME and GH_TOKEN or GITHUB_TOKEN environment variable for full access.")
        print("You can also pass them as command line arguments.")
        print("Usage: uv run scripts/fetch_grammars.py <GH_USERNAME> [GH_TOKEN]")
        return default_headers
    return {**default_headers, "Authorization": f"Bearer {GH_TOKEN}"}


class TreeSitterGrammarResult(NamedTuple):
    """Represents a Tree-sitter grammar file in a Github repo."""

    path: str
    type_: str
    language: str
    repo: TreeSitterRepo
    url: str
    sha: str

    @property
    def save_path(self) -> Path:
        """The filename for saving the grammar locally."""
        return (
            SAVE_DIR
            / f"{self.language}-{self.path.split('/')[-1] if '/' in self.path else self.path}"
        )


class TreeSitterRepo(NamedTuple):
    """Represents a Tree-sitter repository."""

    language: str
    repo: str
    branch: str

    @property
    def base_url(self) -> str:
        """Returns the web URL of the repository."""
        return f"https://github.com/{self.repo}"

    @property
    def clone_url(self) -> str:
        """Returns the URL of the repository."""
        return f"{self.base_url}.git"

    @property
    def branch_url(self) -> str:
        """Returns the URL of the repository branch."""
        return f"{self.base_url}/tree/{self.branch}"

    @property
    def api_url(self) -> str:
        """Returns the API URL of the repository."""
        return f"https://api.github.com/repos/{self.repo}"

    @cached_property
    async def branch_obj(self) -> dict[str, Any]:
        """Fetches the latest branch name for the repository."""
        url = f"{self.api_url}/branches/{self.branch}"
        headers = get_request_headers()
        async with CLIENT as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @cached_property
    async def sha(self) -> str:
        """Fetches the latest commit SHA for the repository branch."""
        branch_info = await self.branch_obj
        return branch_info["commit"]["sha"]

    @cached_property
    async def tree(self) -> dict[str, Any]:
        """Fetches the tree structure of the repository branch."""
        sha = await self.sha
        url = f"{self.api_url}/git/trees/{sha}?recursive=1"
        headers = get_request_headers()
        async with CLIENT as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.json()

    @cached_property
    async def grammar(self) -> TreeSitterGrammarResult:
        """Finds the Tree-sitter grammar file in the repository."""
        tree = await self.tree
        grammar_types = ("grammar.js", "grammar.json", "grammar.ts")
        for item in tree.get("tree", []):
            if next((item["path"].endswith(ext) for ext in grammar_types), None):
                return TreeSitterGrammarResult(
                    path=item["path"],
                    type_=item["type"],
                    language=self.language,
                    repo=self,
                    url=f"{self.api_url}/contents/{item['path']}?ref={self.branch}",
                    sha=item["sha"],
                )
        raise FileNotFoundError(f"Grammar file not found in {self.repo} branch {self.branch}.")


def assemble_repos() -> tuple[TreeSitterRepo, ...]:
    """Assembles a tuple of TreeSitterRepo instances for supported languages."""
    repos = [
        TreeSitterRepo(language, f"tree-sitter/tree-sitter-{language}", "master")
        for language in TREE_SITTER_REPO
    ]
    repos += list(starmap(TreeSitterRepo, OTHER_REPOS))
    return tuple(sorted(repos, key=lambda x: x.language))


def is_grammar_result(result: Any) -> TypeGuard[TreeSitterGrammarResult]:
    """Checks if the result is not an exception."""
    return isinstance(result, TreeSitterGrammarResult)


async def gather_grammars(
    repos: tuple[TreeSitterRepo, ...],
    successful_results: list[TreeSitterGrammarResult] | None = None,
    tries: int = 0,
) -> tuple[TreeSitterGrammarResult, ...]:
    """Fetches the list of Tree-sitter grammars from the GitHub API."""
    successful_results = successful_results or []
    failed = []
    tasks = tuple(repo.grammar for repo in repos)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, result in enumerate(results):
        if is_grammar_result(result):
            successful_results.append(result)
        else:
            failed.append(repos[i])
    if not failed:
        return tuple(successful_results)
    tries += 1
    if tries > 2 and (
        remaining_grammars := len([result for result in results if not is_grammar_result(result)])
    ):
        raise GrammarRetrievalError(
            f"Failed to retrieve all grammars, {remaining_grammars} grammars were not retrieved successfully"
        )
    failed = tuple(f for f in failed if isinstance(f, TreeSitterRepo))
    return await gather_grammars(failed, successful_results, tries)


async def fetch_grammar_file(grammar: TreeSitterGrammarResult) -> None:
    """Fetch and save a grammar file."""
    headers = get_request_headers()
    headers.update({"Content-Type": "application/octet-stream"})
    async with CLIENT as client:
        result = await client.get(grammar.url, headers=headers)
        if result:
            path = grammar.save_path
            if path.exists():
                path.unlink()
            if not path.parent.exists():
                path.mkdir(parents=True, exist_ok=True)
            if result.content:
                path.write_bytes(result.content)
            else:
                raise GrammarRetrievalError(f"There was no content retrieved from {grammar}")


async def main() -> None:
    """Executes the pipeline."""
    repos = assemble_repos()
    with contextlib.suppress(Exception):
        grammars = await gather_grammars(repos)
        if grammars:
            for grammar in grammars:
                await fetch_grammar_file(grammar)
        if len(grammars) != len(repos):
            print(f"Failed to fetch {len(repos) - len(grammars)} grammars.")
            print("missing:")
            missing = [
                repo for repo in repos if all(repo not in grammar.repo for grammar in grammars)
            ]
            for grammar in missing:
                print(f"\n    {grammar.language}")
            print("\n")
        else:
            print("successfully saved all grammars")


if __name__ == "__main__":
    asyncio.run(main())
