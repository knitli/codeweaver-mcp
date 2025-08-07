#!/usr/bin/env -S uv run
# ///script
# requires-python = ">=3.12"
# dependencies = ["httpx", "pydantic", "tomli-w"]
# ///
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
# SPDX-License-Identifier: MIT OR Apache-2.0
# sourcery skip: do-not-use-staticmethod, no-complex-if-expressions
"""Types and utilities for TreeSitter repositories and supported languages in AstGrep."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
import tarfile

from collections.abc import Generator, Iterable
from enum import Enum
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any, Literal, NamedTuple, Self, cast

import httpx
import tomli_w

from pydantic import BaseModel, ConfigDict, Field, RootModel


KeepDirectory = Literal["src", "php_only/src", "typescript/src", "tsx/src", "php/src"]


GH_TOKEN = os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", None))
MULTIPLIER = 1 if GH_TOKEN else 3
REPO_ROOT = Path(__file__).parent.parent
SAVE_DIR = REPO_ROOT / "parsers"
LOCK_FILE = REPO_ROOT / "parsers.lock"


class CommitFetchError(Exception):
    """Exception raised when fetching a commit fails after retries."""


class RepoFetchError(Exception):
    """Exception raised when fetching a repository fails or the response is invalid."""


class RepoExtractionError(Exception):
    """Exception raised when extracting a repository tarball fails."""


@cache
def get_request_headers() -> dict[str, str]:
    """Returns HTTP headers for GitHub API requests.

    The headers include authentication if a GitHub token is available.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "Accept-Encoding": "gzip, deflate, br",
        "User-Agent": "Codeweaver-MCP",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GH_TOKEN:
        headers["Authorization"] = f"Bearer {GH_TOKEN}"
    return headers


async def fetch_with_retry(
    url: httpx.URL, headers: dict[str, str], retries: int = 0, timer: float = 10.0
) -> httpx.Response:
    """Fetches a URL with retries on certain HTTP errors.

    Retries the request with exponential backoff if a non-404 HTTP error occurs, up to a maximum number of retries.

    Args:
        url: The URL to fetch.
        headers: The HTTP headers to use for the request.
        retries: The current retry count.
        timer: Timeout for the request.

    Returns:
        httpx.Response: The HTTP response object.

    Raises:
        CommitFetchError: If the request fails after the maximum number of retries or encounters a non-retryable error.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, follow_redirects=True, timeout=timer)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Handle redirects manually if needed
            if e.response.status_code == 302 and (location := e.response.headers.get("Location")):
                return await fetch_with_retry(httpx.URL(location), headers, retries, timer)

            # Don't retry on 404 or after max retries
            if retries >= 5 or e.response.status_code == 404:
                raise CommitFetchError(
                    f"Failed to fetch the commit after {retries} retries: {e}"
                ) from e

            # Exponential backoff
            await asyncio.sleep(2**retries * MULTIPLIER)
            return await fetch_with_retry(url, headers, retries + 1, timer)
        else:
            return response


async def fetch_latest_commit(repo: str, branch: str) -> str:
    """Fetches the latest commit SHA for a given repository branch.

    Retrieves the commit SHA from the GitHub API for the specified repository and branch.

    Args:
        repo: The GitHub repository in the format "owner/repo".
        branch: The branch name to fetch the latest commit from.

    Returns:
        str: The SHA of the latest commit.

    Raises:
        CommitFetchError: If the commit cannot be fetched after retries.
    """
    headers = get_request_headers()
    owner, repo_name = repo.split("/", 1)
    url = httpx.URL(f"https://api.github.com/repos/{owner}/{repo_name}/commits/{branch}")
    response = await fetch_with_retry(url, headers, 0, 10.0)
    return response.json()["sha"]


def unpack_and_save_tarball(response: httpx.Response, repo_obj: GrammarRepo) -> GrammarRepo:
    """Unpacks a tarball response and saves its contents to the appropriate directory.

    Extracts the relevant files from the tarball, saves them, and updates the GrammarRepo with the new revision and SHA256.

    Args:
        response: The HTTP response containing the tarball.
        repo_obj: The GrammarRepo object describing the repository and extraction details.

    Returns:
        GrammarRepo: The updated GrammarRepo with revision and SHA256.
    """
    language = repo_obj.language
    repo = repo_obj.repo.split("/", 1)[-1]
    keep_dir = repo_obj.keep_dir
    save_dir = SAVE_DIR / str(language.value)

    # Ensure save directory exists and is clean
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        with TemporaryDirectory() as temp_dir:
            tarball_path = Path(temp_dir) / f"{repo}.tar.gz"
            tarball_path.write_bytes(response.content)

            with tarfile.open(tarball_path, "r:gz") as tar:
                _extract_and_copy_tar_contents(tar, keep_dir, save_dir, temp_dir)
            saved_files = list(save_dir.glob("**/*"))
            updates = {"sha256": calculate_sha256(saved_files)}
            if (not repo_obj.license_ or not repo_obj.license_file) and (license_files := [f for f in saved_files if is_license_related(f.name)]):
                if len(license_files) > 1 and (license_file := next((f for f in license_files if "license" in f.name.lower()), None)):
                    updates["license_file"] = license_file.relative_to(REPO_ROOT)
                else:
                    updates["license_file"] = license_files[0].relative_to(REPO_ROOT)
                updates["license_"] = "MIT" if """Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:""" in updates["license_file"].read_text() else None
            return repo_obj.update_self(updates)
    except (tarfile.TarError, OSError) as e:
        raise RepoExtractionError(f"Failed to extract tarball for {repo}: {e}") from e


def is_license_related(name: str) -> bool:
    """Checks if a file name is related to license or copyright."""
    return any(keyword in name.lower() for keyword in ["license", "copying", "copyright", "distrib", "notice"])


def _extract_and_copy_tar_contents(
    tar: tarfile.TarFile, keep_dir: KeepDirectory, save_dir: Path, temp_dir: str
) -> None:
    """Extracts files from a tarball and copies them to the specified directory."""
    temp_path = Path(temp_dir)

    # Find the root directory in the tarball (usually repo-name-commit)
    root_dirs = {member.name.split('/')[0] for member in tar.getmembers() if '/' in member.name}
    if not root_dirs:
        raise RepoExtractionError("No directories found in tarball")

    # Take the first root directory (there should only be one for GitHub tarballs)
    root_dir = next(iter(root_dirs))
    target_path = f"{root_dir}/{keep_dir}"

    # Filter members that are in our target directory
    members_to_extract = [
        member for member in tar.getmembers()
        if (
            (
            member.name.startswith(target_path)
            or is_license_related(member.name)
            )
            and member.name != target_path
            and not (
                member.type != tarfile.DIRTYPE  # Exclude directories
                and (save_dir / member.name).exists()
                and (save_dir / member.name).stat().st_mtime > tar.getmember(member.name).mtime
            )
        )
    ]

    if (
        not members_to_extract
        and not any(
            member.name.startswith(target_path) for member in tar.getmembers()
        )
    ):
        raise RepoExtractionError(f"No files found in {target_path}")
    if not members_to_extract:
        print("No updated files found, skipping extraction.")
        return

    # Extract the filtered members
    tar.extractall(path=temp_dir, members=members_to_extract, filter="data")

    # Copy files from temp to save directory, preserving structure relative to keep_dir
    for member in members_to_extract:
        src = temp_path / member.name
        # Calculate relative path from the keep_dir
        # LICENSE and similar files are saved at the root if not in keep_dir
        rel_path = (
            (Path(member.name).relative_to(target_path))
            if member.name.startswith(target_path)
            else Path(member.name.split('/', 1)[-1] if '/' in member.name else member.name)
        )
        # quick check for duplicate paths since we're changing the structure with license files
        if not member.name.startswith(target_path) and is_license_related(member.name) and (any(member for member in members_to_extract if member.name == rel_path)):
            continue  # favor a license file in the keep_dir over one in the root

        dst = save_dir / rel_path

        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


async def fetch_repo(repo: str, ref: str, repo_obj: GrammarRepo) -> GrammarRepo:
    """Fetches a repository tarball from GitHub and unpacks it.

    Downloads the tarball for the specified repository and reference, then extracts and saves its contents.

    Args:
        repo: The GitHub repository in the format "owner/repo".
        ref: The commit SHA, branch, or tag to fetch.
        repo_obj: The GrammarRepo object describing the repository and extraction details.

    Returns:
        GrammarRepo: The updated GrammarRepo with revision and SHA256.

    Raises:
        RepoFetchError: If the response is not a valid tarball.
    """
    headers = get_request_headers()
    owner, repo_name = repo.split("/", 1)
    url = httpx.URL(f"https://api.github.com/repos/{owner}/{repo_name}/tarball/{ref}")

    try:
        response = await fetch_with_retry(url, headers, 0, 30.0)

        # Check if response is a tarball
        content_type = response.headers.get("Content-Type", "")
        if not (content_type.startswith(("application/gzip", "application/x-gzip"))):
            raise RepoFetchError(f"The response for {repo_name} is not a valid tarball. Content-Type: {content_type}")

        return unpack_and_save_tarball(response, repo_obj)
    except CommitFetchError as e:
        raise RepoFetchError(f"Failed to fetch repository {repo}: {e}") from e


def calculate_sha256(files: Iterable[Path]) -> str:
    """Calculates the SHA256 hash of the contents of the specified files."""
    sha256 = hashlib.sha256()
    for file in sorted(files, key=lambda f: str(f)):
        if file.is_file():
            sha256.update(file.read_bytes())
    return sha256.hexdigest()


class GrammarRepo(NamedTuple):
    """Represents a grammar repository and its relevant metadata.

    Stores information about the repository, branch, directory to keep, revision, and SHA256 hash.
    """
    language: TreeSitterParser
    repo: str  # Fixed: was KeepDirectory, should be str
    branch: str  # Fixed: was KeepDirectory, should be str
    keep_dir: KeepDirectory
    rev: str | None = None  # Fixed: was KeepDirectory, should be str
    sha256: str | None = None
    license_: Annotated[str | None, Field(description="The SPDX ID for the grammar's license", alias="license")] = None  # Optional license information
    license_file: Annotated[Path | None, Field(description="The relative path to the license file", alias="license_file")] = None

    def update_self(self, kwargs: dict[Literal["language", "repo", "branch", "keep_dir", "rev", "sha256", "license_", "license_file"], TreeSitterParser | str | Path | None]) -> Self:
        """Updates the GrammarRepo with new values from keyword arguments.

        Args:
            **kwargs: The keyword arguments to update the GrammarRepo.

        Returns:
            Self: A new GrammarRepo instance with updated values.
        """
        return self._replace(**kwargs)

    def __str__(self) -> str:
        """Returns a string representation of the GrammarRepo."""
        return f"{self.repo}@{self.branch} ({self.language.value})"


class TreeSitterParser(Enum):
    """Represents supported TreeSitter language parsers and provides utilities for repository handling.

    This enum includes all supported languages and methods for mapping, normalization, and repository metadata retrieval.
    """
    BASH = "bash"
    C_LANG = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    CSS = "css"
    ELIXIR = "elixir"
    GO = "go"
    HASKELL = "haskell"
    HTML = "html"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    JSON = "json"
    KOTLIN = "kotlin"
    LUA = "lua"
    NIX = "nix"
    PHP = "php"
    PHP_STRICT = "php_strict"
    PYTHON = "python"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SOLIDITY = "solidity"
    SWIFT = "swift"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    YAML = "yaml"

    @classmethod
    def from_str(cls, value: str) -> TreeSitterParser:
        """Converts a string to a TreeSitterParser enum member.

        Normalizes the input string and maps it to the appropriate TreeSitterParser value.

        Args:
            value: The string representation of the parser.

        Returns:
            TreeSitterParser: The corresponding enum member.

        Raises:
            KeyError: If the normalized value does not match any enum member.
        """
        normalized = value.strip().replace("-", "_").lower()
        match normalized:
            case "php_only" | "php_strict" | "just_php":
                return cls.PHP_STRICT
            case "c_sharp" | "c#":
                return cls.CSHARP
            case "yml":
                return cls.YAML
            case "c++":
                return cls.CPP
            case "c_lang":
                return cls.C_LANG
            case "ts":
                return cls.TYPESCRIPT
            case "js":
                return cls.JAVASCRIPT
            case "htm":
                return cls.HTML
            case _:
                return cls.__members__[normalized.upper()]

    @classmethod
    def members(cls) -> Generator[TreeSitterParser]:
        """Returns a generator yielding all TreeSitterParser enum members."""
        yield from cls.__members__.values()

    @property
    def repo_tuple(self) -> GrammarRepo:
        """Returns a GrammarRepo tuple for the TreeSitterParser."""
        keep = self.keep_dir
        name = (
            f"tree-sitter-{self.value}"
            if self != TreeSitterParser.CSHARP
            else "tree-sitter-c-sharp"
        )
        match self:
            case TreeSitterParser.PHP_STRICT:
                return GrammarRepo(self, "tree-sitter/tree-sitter-php", "master", keep)
            case TreeSitterParser.TSX:
                return GrammarRepo(self, "tree-sitter/tree-sitter-typescript", "master", keep)
            case TreeSitterParser.ELIXIR:
                return GrammarRepo(self, f"elixir-lang/{name}", "main", keep)
            case TreeSitterParser.KOTLIN | TreeSitterParser.LUA | TreeSitterParser.YAML:
                return GrammarRepo(
                    self,
                    f"tree-sitter-grammars/{name}",
                    "main" if self == TreeSitterParser.LUA else "master",
                    keep,
                )
            case TreeSitterParser.NIX:
                return GrammarRepo(self, f"nix-community/{name}", "master", keep)
            case TreeSitterParser.SOLIDITY:
                return GrammarRepo(self, f"JoranHonig/{name}", "master", keep)
            case TreeSitterParser.SWIFT:
                return GrammarRepo(self, f"alex-pinkus/{name}", "main", keep)
            case _:
                return GrammarRepo(self, f"tree-sitter/{name}", "master", keep)

    @property
    def keep_dir(self) -> KeepDirectory:
        """Returns the directory to keep for the TreeSitterParser from a downloaded repo."""
        if self == TreeSitterParser.PHP_STRICT:
            return "php_only/src"
        if self in (TreeSitterParser.TYPESCRIPT, TreeSitterParser.TSX, TreeSitterParser.PHP):
            return cast(KeepDirectory, f"{self.value}/src")
        return "src"


    @classmethod
    def repos(cls) -> Generator[GrammarRepo]:
        """Yields GrammarRepo tuples for all TreeSitterParser enum members."""
        yield from (lang.repo_tuple for lang in cls.members())


class ParserProperties(BaseModel):
    """Represents properties of a TreeSitter parser and its associated repository."""
    model_config = ConfigDict(
            cache_strings=True,
            extra="forbid",
            frozen=True,
            str_strip_whitespace=True,
        )

    parser: Annotated[TreeSitterParser, Field(description="The language.")]
    repo: Annotated[GrammarRepo, Field(description="The repo.")]

    def check_if_outdated(self, latest_commit: str) -> bool:
        """Checks if the parser's repository is outdated compared to the latest commit."""
        return self.repo.rev != latest_commit

    @property
    def directory(self) -> Path:
        """Returns the directory where the parser's grammar files are stored."""
        return (SAVE_DIR / self.parser.value).relative_to(REPO_ROOT)

    @property
    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary with enum values serialized as strings."""
        return {
            "parser": self.parser.value,
            "repo": {
                "language": self.repo.language.value,
                "repo": self.repo.repo,
                "branch": self.repo.branch,
                "keep_dir": self.repo.keep_dir,
                "rev": self.repo.rev,
                "sha256": self.repo.sha256,
                "directory": str(self.directory),
                "license": self.repo.license_ or "",
                "license_file": str(self.repo.license_file) or "",
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, any]) -> Self:
        """Create from a dictionary with string enum values."""
        parser = TreeSitterParser(data["parser"])
        repo_data = data["repo"]
        repo = GrammarRepo(
            language=TreeSitterParser(repo_data["language"]),
            repo=repo_data["repo"],
            branch=repo_data["branch"],
            keep_dir=repo_data["keep_dir"],
            rev=repo_data["rev"],
            sha256=repo_data["sha256"],
            license_=repo_data.get("license", ""),
            license_file=Path(repo_data.get("license_file", "")))
        # directory is not needed here, it will be computed later
        return cls(parser=parser, repo=repo)


class Parsers(RootModel[list[ParserProperties]]):
    """A collection of parser properties for TreeSitter languages."""
    model_config = ConfigDict(
        cache_strings=True,
        frozen=True,
        str_strip_whitespace=True,
    )

    @classmethod
    def from_repo_map(cls, repo_map: dict[TreeSitterParser, GrammarRepo]) -> Self:
        """Creates a Parsers instance from a mapping of TreeSitterParser to GrammarRepo."""
        parser_props = [
            ParserProperties(parser=parser, repo=repo)
            for parser, repo in repo_map.items()
        ]
        return cls.model_validate(parser_props)

    @classmethod
    def to_toml_file(cls, parsers: list[ParserProperties], file_path: Path = LOCK_FILE) -> None:
        """Saves the parsers to a TOML file."""
        import textwrap
        header = textwrap.dedent("""# SPDX-FileCopyrightText: 2025 Knitli Inc.
        # SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
        #
        # SPDX-License-Identifier: MIT OR Apache-2.0
        # This file is auto-generated by scripts/fetch_grammars.py.
        # Don't edit it manually.
        """).encode("utf-8")
        # This file is auto-generated by scripts/fetch_grammars.py.
        data = {"parsers": [p.to_dict for p in parsers]}
        with file_path.open("wb") as f:
            assembled_file = header + b"\n\n" + tomli_w.dumps(data).encode("utf-8")
            f.write(assembled_file)

    @classmethod
    def from_toml_file(cls, file_path: Path = LOCK_FILE) -> Self:
        """Load parsers from a TOML file."""
        import tomllib  # Python 3.11+ built-in, or use tomli for older versions

        with file_path.open("rb") as f:
            data = tomllib.load(f)

        parser_props = [ParserProperties.from_dict(parser_data) for parser_data in data["parsers"]]
        return cls.model_validate(parser_props)

    def match_parser(
        self,
        parser: TreeSitterParser,
    ) -> ParserProperties | None:
        """Finds a ParserProperties instance matching the given TreeSitterParser."""
        return next((prop for prop in self.root if prop.parser == parser), None)


async def fetch_all_grammars(
    parsers: list[TreeSitterParser],
) -> dict[TreeSitterParser, GrammarRepo]:
    """Fetches all TreeSitter grammars and returns a mapping of parsers to their repositories."""
    repo_map = {}
    existing_parsers = None
    with contextlib.suppress(Exception):
        existing_parsers = Parsers.from_toml_file(LOCK_FILE)

    for parser in parsers:
        repo = existing_parsers.match_parser(parser).repo if existing_parsers else parser.repo_tuple
        try:
            latest_commit = await fetch_latest_commit(repo.repo, repo.branch)

            # If we already have the latest commit
            # AND the repo has the necessary metadata, skip fetching
            if repo.rev == latest_commit and (repo.sha256 and repo.license_ and repo.license_file):
                repo_map[parser] = repo
                print(f"✓ {parser.value}: already up to date ({latest_commit[:8]})")
                continue

            # Download and extract the repository
            print(f"⬇ {parser.value}: downloading {latest_commit[:8]}")
            updated_repo = repo.update_self({"rev": latest_commit})
            final_repo = await fetch_repo(repo.repo, latest_commit, updated_repo) or updated_repo
            repo_map[parser] = final_repo
            print(f"✓ {parser.value}: downloaded and extracted")

        except (CommitFetchError, RepoFetchError, RepoExtractionError) as e:
            print(f"✗ {parser.value}: {e}")
            continue

    return repo_map


async def main() -> None:
    """Main function to fetch all TreeSitter grammars and save them to a TOML file."""
    langs = list(TreeSitterParser.members())
    print(f"Fetching {len(langs)} grammars...")

    repo_map = await fetch_all_grammars(langs)
    print(f"\nSuccessfully processed {len(repo_map)}/{len(langs)} repositories.")

    if repo_map:
        parsers = Parsers.from_repo_map(repo_map)
        out_path = Path(__file__).parent.parent / "parsers.lock"
        Parsers.to_toml_file(parsers.root, LOCK_FILE)
        print(f"Saved to {out_path}")
    else:
        print("No repositories were successfully processed.")


if __name__ == "__main__":
    asyncio.run(main())
