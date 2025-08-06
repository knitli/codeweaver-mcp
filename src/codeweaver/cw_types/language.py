# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Common types for programming and configuration languages in CodeWeaver."""

import os

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

    dependency_key_paths: tuple[tuple[str, ...]] | None = None
    """
    A tuple consisting of tuples. Each inner tuple represents a path to the package dependencies in the config file (not dev, build, or any other dependency groups -- just package dependencies).

    For example, in `pyproject.toml`, there are at least two paths to package dependencies:

      ```python
        dependency_key_paths=(
            ("tool", "poetry", "dependencies"),  # poetry users
            ("project", "dependencies"),         # normal people... I mean, PEP 621 followers
            )
        ```

    If there's only one path, you should still represent it as tuple of tuples, like:
      - `(("tool", "poetry", "dependencies"),)`  # <-- the trailing comma is important

    Some cases me just be a single path with a single key, like:
        - `(("dependencies",),)`

    Makefiles don't really have keys, per-se, but we instead use the `dependency_key_paths` to indicate which variable is used for dependencies, like `CXXFLAGS` or `LDFLAGS`:
    - `dependency_key_paths=(("CXXFLAGS",),)`  # for C++ Makefiles
    - `dependency_key_paths=(("LDFLAGS",),)`   # for C Makefiles
    """


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
    def config_files(self) -> tuple[LanguageConfigFile, ...] | None:  # noqa: C901  # it's long, but not complex
        """
        Returns the LanguageConfigFile associated with this language.

        TODO: Validate the `dependency_key_paths` for each config file to ensure they are correct. If you use these languages, please let us know if you find any issues with the `dependency_key_paths` in the config files. Some may well be incorrect.
        """
        match self:
            case SemanticSearchLanguage.C_LANG:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Makefile",
                        language_type=ConfigLanguage.MAKEFILE,
                        dependency_key_paths=(("CFLAGS",), ("LDFLAGS",)),
                    ),
                )
            case SemanticSearchLanguage.C_PLUS_PLUS:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "CMakeLists.txt",
                        language_type=ConfigLanguage.CMAKEFILE,
                        dependency_key_paths=(("CMAKE_CXX_FLAGS",), ("CMAKE_EXE_LINKER_FLAGS",)),
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Makefile",
                        language_type=ConfigLanguage.MAKEFILE,
                        dependency_key_paths=(("CXXFLAGS",), ("LDFLAGS",))
                    ),
                )
            case SemanticSearchLanguage.C_SHARP:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "app.config",
                        language_type=ConfigLanguage.XML,
                        dependency_key_paths=(
                            ("configuration", "appSettings", "add"),
                            ("configuration", "connectionStrings", "add"),
                            ("configuration", "runtime", "assemblyBinding", "dependentAssembly", "assemblyIdentity"),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=next(iter(PROJECT_ROOT.glob("*.csproj"))),
                        language_type=ConfigLanguage.XML,
                        dependency_key_paths=(
                            ("Project", "ItemGroup", "PackageReference"),
                            ("Project", "ItemGroup", "Reference"),
                            ("Project", "ItemGroup", "ProjectReference"),
                        ),
                    ),
                )
            case SemanticSearchLanguage.ELIXIR:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "mix.exs",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("deps",),
                            ("aliases", "deps"),
                        ),
                    ),
                )
            case SemanticSearchLanguage.GO:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "go.mod",
                        language_type=ConfigLanguage.INI,
                        dependency_key_paths=(
                            ("require",),
                            ("replace",)
                        )
                    )
                )
            case SemanticSearchLanguage.HASKELL:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "package.yaml",
                        language_type=ConfigLanguage.YAML,
                        dependency_key_paths=(
                            ("dependencies",),
                            ("build-depends",),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=Path(os.environ.get("STACK_YAML", PROJECT_ROOT / "stack.yml")),
                        language_type=ConfigLanguage.YAML,
                        dependency_key_paths=(
                            ("extra-deps",),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=next(iter(PROJECT_ROOT.glob("*.cabal"))),
                        language_type=ConfigLanguage.INI,
                        dependency_key_paths=(
                            ("build-depends",),
                            ("library", "build-depends"),
                            ("executable", "build-depends"),
                        ),
                    )
                )
            case SemanticSearchLanguage.JAVA:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "pom.xml",
                        language_type=ConfigLanguage.XML,
                        dependency_key_paths=(
                            ("project", "dependencies", "dependency"),
                            ("project", "dependencyManagement", "dependencies", "dependency"),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "build.gradle",
                        language_type=ConfigLanguage.GROOVY,
                        dependency_key_paths=(
                            ("dependencies",),
                            ("configurations", "compileClasspath", "dependencies"),
                            ("configurations", "runtimeClasspath", "dependencies"),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "build.gradle.kts",
                        language_type=ConfigLanguage.KOTLIN,
                        dependency_key_paths=(
                            ("dependencies",),
                            ("configurations", "compileClasspath", "dependencies"),
                            ("configurations", "runtimeClasspath", "dependencies"),
                        )
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
                        dependency_key_paths=(
                            ("dependencies",),
                        )
                    ),
                )
            case SemanticSearchLanguage.KOTLIN:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "build.gradle.kts",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("dependencies",),
                            ("configurations", "compileClasspath", "dependencies"),
                            ("configurations", "runtimeClasspath", "dependencies"),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "settings.gradle.kts",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("dependencyResolutionManagement", "repositories"),
                            ("pluginManagement", "repositories"),
                        )
                    ),
                )
            case SemanticSearchLanguage.LUA:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "luarocks.json",
                        language_type=ConfigLanguage.JSON,
                        dependency_key_paths=(
                            ("dependencies",),
                            ("build_dependencies",),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "rockspec.json",
                        language_type=ConfigLanguage.JSON,
                        dependency_key_paths=(
                            ("dependencies",),
                            ("build_dependencies",),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "rockspec",
                        language_type=ConfigLanguage.INI,
                        dependency_key_paths=(
                            ("dependencies",),
                            ("build_dependencies",),
                        )
                    ),
                )
            case SemanticSearchLanguage.NIX:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "default.nix",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("dependencies",),
                            ("buildInputs",),
                            ("nativeBuildInputs",),
                            ("propagatedBuildInputs",),
                            ("buildInputs", "dependencies"),
                        )
                    ),
                )
            case SemanticSearchLanguage.PHP:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "composer.json",
                        language_type=ConfigLanguage.JSON,
                        dependency_key_paths=(
                            ("require",),
                        )
                    ),
                )
            case SemanticSearchLanguage.PYTHON:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "pyproject.toml",
                        language_type=ConfigLanguage.TOML,
                        dependency_key_paths=(
                            ("tool", "poetry", "dependencies"),
                            ("project", "dependencies"),
                        ),
                    ),
                )
            case SemanticSearchLanguage.RUBY:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Gemfile",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("gems",),
                            ("source", "gems"),
                            ("source", "gemspec", "gems"),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Rakefile",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("gems",),
                            ("source", "gems"),
                            ("source", "gemspec", "gems"),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "gemspec",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("dependencies",),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "config.ru",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("gems",),
                            ("source", "gems"),
                            ("source", "gemspec", "gems"),
                        )
                    ),
                )
            case SemanticSearchLanguage.RUST:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Cargo.toml",
                        language_type=ConfigLanguage.TOML,
                        dependency_key_paths=(
                            ("dependencies",),
                        )
                    ),
                )
            case SemanticSearchLanguage.SCALA:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "build.sbt",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("libraryDependencies",),
                            ("compile", "dependencies"),
                            ("runtime", "dependencies"),
                        )
                    ),
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "project" / "build.properties",
                        language_type=ConfigLanguage.PROPERTIES,
                        dependency_key_paths=(
                            ("sbt.version",),
                        )
                    ),
                )
            case SemanticSearchLanguage.SWIFT:
                return (
                    LanguageConfigFile(
                        language=self,
                        path=PROJECT_ROOT / "Package.swift",
                        language_type=ConfigLanguage.SELF,
                        dependency_key_paths=(
                            ("dependencies",),
                            ("targets", "dependencies"),
                            ("products", "dependencies")
                        )
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
    def ext_map(cls) -> MappingProxyType[str, "SemanticSearchLanguage"]:
        """
        Returns a mapping of extensions to their corresponding SemanticSearchLanguage.
        """
        return MappingProxyType({
            ext: lang for lang in cls.members() for ext in lang.extensions if ext in lang.extensions
        })

    @classmethod
    def all_config_paths(cls) -> tuple[Path, ...]:
        """
        Returns a mapping of all config file paths to their corresponding LanguageConfigFile.
        """
        all_paths = []
        for lang in cls.members():
            if lang.config_files:
                all_paths.extend(config_file.path for config_file in lang.config_files if config_file)
        return tuple(all_paths)

    @classmethod
    def language_from_config_file(cls, config_file: Path) -> "SemanticSearchLanguage | None":
        """
        Returns the SemanticSearchLanguage for a given configuration file path.

        Args:
            config_file: The path to the configuration file.

        Returns:
            The corresponding SemanticSearchLanguage, or None if not found.
        """
        normalized_path = PROJECT_ROOT / config_file.name
        if not normalized_path.exists() or all(
            str(normalized_path) not in str(p) for p in cls.all_config_paths()
        ):
            return None
        if config_file.name in ("Makefile", "build.gradle.kts"):
            # there's language ambiguity here. TODO: Add check to resolve this ambiguity
            # for now, we make an educated guess
            if config_file.name == "Makefile":
                # C++ is more popular... no other reasoning here
                return SemanticSearchLanguage.C_PLUS_PLUS
            # Java more common than Kotlin, but Kotlin is more likely to use 'build.gradle.kts' ... I think. 🤷‍♂️
            return SemanticSearchLanguage.KOTLIN
        return next(
            (
                lang
                for lang in cls.members()
                if any(cfg.name for cfg in lang.config_files if cfg.name == config_file.name)
            ),
            None,
        )

    @classmethod
    def lang_from_ext(cls, ext: str) -> "SemanticSearchLanguage | None":
        """
        Returns the SemanticSearchLanguage for a given file extension.

        Args:
            ext: The file extension to look up.

        Returns:
            The corresponding SemanticSearchLanguage, or None if not found.
        """
        return cls.ext_map().get(ext.lstrip(".").lower(), None)


# Helper functions

def find_config_files() -> tuple[Path, ...] | None:
    """
    Finds all configuration files in the project root directory.

    Returns:
        A tuple of Path objects representing the configuration files, or None if no config files are found.
    """
    config_files = tuple(p for p in SemanticSearchLanguage.all_config_paths() if p.exists())
    return config_files or None


def languages_present_from_configs() -> tuple[SemanticSearchLanguage, ...]:
    """
    Returns a tuple of SemanticSearchLanguage for all languages present in the configuration files.

    Returns:
        A tuple of SemanticSearchLanguage objects.

    TODO: Integrate into indexing and search services to use these languages.
    """
    if config_files := find_config_files():
        return tuple(
            SemanticSearchLanguage.language_from_config_file(config_file) for config_file in config_files if config_file
        )
    return ()
