"""Common types for programming and configuration languages in CodeWeaver."""

from functools import cached_property
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

from codeweaver.cw_types.base_enum import BaseEnum
from codeweaver.utils import walk_down_to_git_root


PROJECT_ROOT = walk_down_to_git_root(Path.cwd()) or Path.cwd().resolve()


class LanguageConfigFile(NamedTuple):
    """
    Represents a language configuration file with its name, path, and language type.
    """

    language: "SemanticSearchLanguage"

    path: Path

    language_type: "ConfigLanguage"


class ConfigLanguage(BaseEnum):
    """
    Enum representing common configuration languages.
    """

    BASH = "bash"
    CMAKEFILE = "cmakefile"
    INI = "ini"
    JSON = "json"
    GROOVY = "groovy"  # Used for Gradle build scripts for Java
    KOTLIN = "kotlin"  # Used for Kotlin build scripts for Java
    MAKEFILE = "makefile"
    PROPERTIES = "properties"
    SELF = "self"  # Language's config is written in the same language (e.g., Kotlin, Scala)
    TOML = "toml"
    XML = "xml"
    YAML = "yaml"


class SemanticSearchLanguage(BaseEnum):
    """
    Enum representing supported languages for semantic (AST) search.

    Note: This is the list of built-in languages supported by ast-grep. Ast-grep supports dynamic languages using pre-compiled tree-sitter grammars. We haven't added support for those yet.
    """

    BASH = "bash"
    C_LANG = "c"
    C_PLUS_PLUS = "cpp"
    C_SHARP = "csharp"
    CSS = "css"
    ELIXIR = "elixir"
    GO = "go"
    HASKELL = "haskell"
    HTML = "html"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    JSX = "jsx"
    JSON = "json"
    KOTLIN = "kotlin"
    LUA = "lua"
    NIX = "nix"
    PHP = "php"
    PYTHON = "python"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SOLIDITY = "solidity"
    SWIFT = "swift"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    YAML = "yaml"

    @property
    def extensions(self) -> tuple[str]:
        """
        Returns the file extensions associated with this language.
        """
        return {
            SemanticSearchLanguage.BASH: (
                "sh",
                "bash",
                "zsh",
                "bashrc",
                "bash_profile",
                "zshrc",
                "profile",
                "ksh",
            ),
            SemanticSearchLanguage.C_LANG: ("c", "h"),
            SemanticSearchLanguage.C_PLUS_PLUS: ("cpp", "hpp", "cc", "cxx"),
            SemanticSearchLanguage.C_SHARP: ("cs", "csharp"),
            SemanticSearchLanguage.CSS: ("css"),
            SemanticSearchLanguage.ELIXIR: ("ex", "exs"),
            SemanticSearchLanguage.GO: ("go"),
            SemanticSearchLanguage.HASKELL: ("hs"),
            SemanticSearchLanguage.HTML: ("html", "htm", "xhtml"),
            SemanticSearchLanguage.JAVA: ("java"),
            SemanticSearchLanguage.JAVASCRIPT: ("js", "mjs", "cjs"),
            SemanticSearchLanguage.JSON: ("json"),
            SemanticSearchLanguage.JSX: ("jsx"),
            SemanticSearchLanguage.KOTLIN: ("kt", "kts", "ktm"),
            SemanticSearchLanguage.LUA: ("lua"),
            SemanticSearchLanguage.NIX: ("nix"),
            SemanticSearchLanguage.PHP: ("php", "phtml"),
            SemanticSearchLanguage.PYTHON: ("py", "pyi", "py3", "bzl"),
            SemanticSearchLanguage.RUBY: ("rb", "gemspec", "rake", "ru"),
            SemanticSearchLanguage.RUST: ("rs"),
            SemanticSearchLanguage.SCALA: ("scala", "sc", "sbt"),
            SemanticSearchLanguage.SOLIDITY: ("sol"),
            SemanticSearchLanguage.SWIFT: ("swift"),
            SemanticSearchLanguage.TYPESCRIPT: ("ts", "mts", "cts"),
            SemanticSearchLanguage.TSX: ("tsx"),
            SemanticSearchLanguage.YAML: ("yaml", "yml"),
        }.get(self, ())

    @property
    def config_files(self) -> tuple[LanguageConfigFile, ...] | None:  # noqa: C901
        """
        Returns the LanguageConfigFile associated with this language.
        """
        match self:
            case SemanticSearchLanguage.C_LANG:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Makefile",
                        language_type=ConfigLanguage.MAKEFILE,
                    ),
                )
            case SemanticSearchLanguage.C_PLUS_PLUS:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "CMakeLists.txt",
                        language_type=ConfigLanguage.CMAKEFILE,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Makefile",
                        language_type=ConfigLanguage.MAKEFILE,
                    ),
                )
            case SemanticSearchLanguage.C_SHARP:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "app.config",
                        language_type=ConfigLanguage.XML,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=next(iter(PROJECT_ROOT.glob("*.csproj"))),
                        language_type=ConfigLanguage.XML,
                    ),
                )
            case SemanticSearchLanguage.ELIXIR:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "mix.exs",
                        language_type=ConfigLanguage.SELF,
                    ),
                )
            case SemanticSearchLanguage.GO:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "go.mod",
                        language_type=ConfigLanguage.INI,
                    ),
                )
            case SemanticSearchLanguage.HASKELL:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "package.yaml",
                        language_type=ConfigLanguage.YAML,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "package.yml",
                        language_type=ConfigLanguage.YAML,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=next(iter(PROJECT_ROOT.glob("*.cabal"))),
                        language_type=ConfigLanguage.INI,
                    ),
                )
            case SemanticSearchLanguage.JAVA:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "pom.xml",
                        language_type=ConfigLanguage.XML,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "build.gradle",
                        language_type=ConfigLanguage.GROOVY,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "build.gradle.kts",
                        language_type=ConfigLanguage.KOTLIN,
                    ),
                )
            case (
                SemanticSearchLanguage.JAVASCRIPT
                | SemanticSearchLanguage.JSX
                | SemanticSearchLanguage.TYPESCRIPT
                | SemanticSearchLanguage.TSX
            ):
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "package.json",
                        language_type=ConfigLanguage.JSON,
                    ),
                )
            case SemanticSearchLanguage.KOTLIN:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "build.gradle.kts",
                        language_type=ConfigLanguage.SELF,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "settings.gradle.kts",
                        language_type=ConfigLanguage.SELF,
                    ),
                )
            case SemanticSearchLanguage.LUA:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "luarocks.json",
                        language_type=ConfigLanguage.JSON,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "rockspec.json",
                        language_type=ConfigLanguage.JSON,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "rockspec",
                        language_type=ConfigLanguage.INI,
                    ),
                )
            case SemanticSearchLanguage.NIX:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "default.nix",
                        language_type=ConfigLanguage.SELF,
                    ),
                )
            case SemanticSearchLanguage.PHP:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "composer.json",
                        language_type=ConfigLanguage.JSON,
                    ),
                )
            case SemanticSearchLanguage.PYTHON:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "pyproject.toml",
                        language_type=ConfigLanguage.TOML,
                    ),
                )
            case SemanticSearchLanguage.RUBY:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Gemfile",
                        language_type=ConfigLanguage.SELF,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Rakefile",
                        language_type=ConfigLanguage.SELF,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "gemspec",
                        language_type=ConfigLanguage.SELF,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "config.ru",
                        language_type=ConfigLanguage.SELF,
                    ),
                )
            case SemanticSearchLanguage.RUST:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Cargo.toml",
                        language_type=ConfigLanguage.TOML,
                    ),
                )
            case SemanticSearchLanguage.SCALA:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "build.sbt",
                        language_type=ConfigLanguage.SELF,
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "project" / "build.properties",
                        language_type=ConfigLanguage.PROPERTIES,
                    ),
                )
            case SemanticSearchLanguage.SWIFT:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Package.swift",
                        language_type=ConfigLanguage.SELF,
                    ),
                )
            case _:
                return None

    @classmethod
    def all_extensions(cls) -> tuple[str, ...]:
        """
        Returns all file extensions for all languages.
        """
        return tuple(ext for lang in cls.members() for ext in lang.extensions)

    @classmethod
    @cached_property
    def ext_map(cls) -> MappingProxyType[str, "SemanticSearchLanguage"]:
        """
        Returns a mapping of extensions to their corresponding SemanticSearchLanguage.
        """
        return MappingProxyType({
            ext: lang for lang in cls.members() for ext in lang.extensions if ext in lang.extensions
        })
