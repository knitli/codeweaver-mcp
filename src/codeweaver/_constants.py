# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Constants used throughout the CodeWeaver project, primarily for default configurations.
"""

from collections.abc import Generator
from typing import Literal, LiteralString, NamedTuple


class ExtLangPair(NamedTuple):
    """
    Mapping of file extensions to their corresponding programming languages.

    Not all 'extensions' are actually file extensions, some are file names or special cases, like `Makefile` or `Dockerfile`.
    """

    ext: LiteralString
    """The file extension, including leading dot if it's a file extension."""

    language: LiteralString
    """The programming or config language associated with the file extension."""

    @property
    def is_actual_ext(self) -> bool:
        """Check if the extension is a valid file extension."""
        return self.ext.startswith(".")

    @property
    def is_file_name(self) -> bool:
        """Check if the extension is a file name."""
        return not self.ext.startswith(".")

    @property
    def is_config(self) -> bool:
        """Check if the extension is a configuration file."""
        return self.language in CONFIG_FILE_LANGUAGES

    @property
    def is_doc(self) -> bool:
        """Check if the extension is a documentation file."""
        return next((True for doc_ext in DOC_FILES_EXTENSIONS if doc_ext.ext == self.ext), False)

    @property
    def is_code(self) -> bool:
        """Check if the extension is a code file."""
        return not self.is_config and not self.is_doc and not self.is_file_name

    @property
    def category(self) -> Literal["code", "docs", "config"]:
        """Return the language of file based on its extension."""
        if self.is_code:
            return "code"
        if self.is_doc:
            return "docs"
        if self.is_config:
            return "config"
        raise ValueError(f"Unknown category for {self.ext}")

    def is_same(self, filename: str) -> bool:
        """Check if the given filename is the same filetype as the extension."""
        if self.is_actual_ext:
            return filename.endswith(self.ext)
        return filename == self.ext if self.is_file_name else False


DEFAULT_EXCLUDED_DIRS: frozenset[LiteralString] = frozenset({
    ".DS_Store",
    ".cache",
    ".claude",
    ".eslintcache",
    ".git",
    ".hg",
    ".history",
    ".idea",
    ".jj",
    ".next",
    ".nuxt",
    ".roo",
    ".ruff_cache",
    ".svn",
    ".temp",
    ".tmp",
    ".tsbuildinfo",
    ".venv",
    ".vs",
    ".vscode",
    "Debug",
    "Release",
    "Releases",
    "Thumbs.db",
    "__pycache__",
    "__pytest_cache__",
    "aarch64",
    "arm",
    "arm64",
    "bin",
    "bld",
    "bower_components",
    "build",
    "debug",
    "dist",
    "htmlcov",
    "li64",
    "log",
    "logs",
    "node_modules",
    "obj",
    "out",
    "release",
    "releases",
    "remote-debug-profile",
    "site",
    "target",
    "temp",
    "venv",
    "win32",
    "win64",
    "x64",
    "x86",
})

DEFAULT_EXCLUDED_EXTENSIONS: frozenset[LiteralString] = frozenset({
    ".7z",
    ".avif",
    ".bmp",
    ".builds",
    ".cache",
    ".class",
    ".code-workspace",
    ".coverage",
    ".coverage.xml",
    ".dll",
    ".dmg",
    ".env",
    ".exe",
    ".gif",
    ".gz",
    ".iobj",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lcov",
    ".local",
    ".lock",
    ".log",
    ".meta",
    ".mov",
    ".mp3",
    ".mp4",
    ".msi",
    ".o",
    ".obj",
    ".pch",
    ".pdb",
    ".pgc",
    ".pgd",
    ".png",
    ".pyc",
    ".pyo",
    ".rar",
    ".rsp",
    ".scc",
    ".sig",
    ".snk",
    ".so",
    ".svclog",
    ".svg",
    ".swo",
    ".swp",
    ".tar",
    ".temp",
    ".tlb",
    ".tlog",
    ".tmp",
    ".tmp_proj",
    ".vspec",
    ".vssscc",
    ".wav",
    ".webm",
    ".webp",
    ".zip",
})

DOC_FILES_EXTENSIONS: tuple[ExtLangPair, ...] = (
    ExtLangPair(ext=".1", language="man"),
    ExtLangPair(ext=".2", language="man"),
    ExtLangPair(ext=".3", language="man"),
    ExtLangPair(ext=".4", language="man"),
    ExtLangPair(ext=".5", language="man"),
    ExtLangPair(ext=".6", language="man"),
    ExtLangPair(ext=".7", language="man"),
    ExtLangPair(ext=".8", language="man"),
    ExtLangPair(ext=".9", language="man"),
    ExtLangPair(ext=".Rmd", language="rmarkdown"),
    ExtLangPair(ext=".adoc", language="asciidoc"),
    ExtLangPair(ext=".asc", language="asciidoc"),
    ExtLangPair(ext=".asciidoc", language="asciidoc"),
    ExtLangPair(ext=".confluence", language="confluence"),
    ExtLangPair(ext=".creole", language="creole"),
    ExtLangPair(ext=".dita", language="dita"),
    ExtLangPair(ext=".docbook", language="docbook"),
    ExtLangPair(ext=".help", language="help"),
    ExtLangPair(ext=".hlp", language="help"),
    ExtLangPair(ext=".info", language="info"),
    ExtLangPair(ext=".ipynb", language="jupyter"),
    ExtLangPair(ext=".lagda", language="lagda"),
    ExtLangPair(ext=".latex", language="latex"),
    ExtLangPair(ext=".lhs", language="lhs"),
    ExtLangPair(ext=".man", language="man"),
    ExtLangPair(ext=".manpage", language="man"),
    ExtLangPair(ext=".markdown", language="markdown"),
    ExtLangPair(ext=".md", language="markdown"),
    ExtLangPair(ext=".mdown", language="markdown"),
    ExtLangPair(ext=".mdx", language="markdown"),
    ExtLangPair(ext=".mediawiki", language="mediawiki"),
    ExtLangPair(ext=".mkd", language="markdown"),
    ExtLangPair(ext=".mkdn", language="markdown"),
    ExtLangPair(ext=".nw", language="nw"),
    ExtLangPair(ext=".org", language="org"),
    ExtLangPair(ext=".pmd", language="pmd"),
    ExtLangPair(ext=".pod", language="pod"),
    ExtLangPair(ext=".pyx", language="cython"),
    ExtLangPair(ext=".rdoc", language="rdoc"),
    ExtLangPair(ext=".rest", language="restructuredtext"),
    ExtLangPair(ext=".rmd", language="rmd"),
    ExtLangPair(ext=".rnw", language="rnw"),
    ExtLangPair(ext=".rst", language="restructuredtext"),
    ExtLangPair(ext=".rtf", language="rtf"),
    ExtLangPair(ext=".tex", language="latex"),
    ExtLangPair(ext=".texi", language="texinfo"),
    ExtLangPair(ext=".texinfo", language="texinfo"),
    ExtLangPair(ext=".text", language="text"),
    ExtLangPair(ext=".textile", language="textile"),
    ExtLangPair(ext=".txt", language="text"),
    ExtLangPair(ext=".wiki", language="wiki"),
    ExtLangPair(ext=".xml", language="xml"),
    ExtLangPair(ext=".yard", language="yard"),
)
"""A tuple of `ExtLangPair` for documentation files."""

# spellchecker:off
CODE_FILES_EXTENSIONS: tuple[ExtLangPair, ...] = (
    ExtLangPair(".R", "r"),
    ExtLangPair(".Rprofile", "r"),
    ExtLangPair(".app.src", "erlang"),
    ExtLangPair(".as", "assemblyscript"),
    ExtLangPair(".asd", "lisp"),
    ExtLangPair(".aux", "latex"),
    ExtLangPair(".bat", "batch"),
    ExtLangPair(".bb", "clojure"),
    ExtLangPair(".beef", "beef"),
    ExtLangPair(".boot", "clojure"),
    ExtLangPair(".carbon", "carbon"),
    ExtLangPair(".chapel", "chapel"),
    ExtLangPair(".clj", "clojure"),
    ExtLangPair(".cljc", "clojure"),
    ExtLangPair(".cljs", "clojure"),
    ExtLangPair(".cljx", "clojure"),
    ExtLangPair(".cls", "latex"),
    ExtLangPair(".cmake", "cmake"),
    ExtLangPair(".coffee", "coffeescript"),
    ExtLangPair(".cr", "crystal"),
    ExtLangPair(".cue", "cue"),
    ExtLangPair(".d", "dlang"),
    ExtLangPair(".dart", "dart"),
    ExtLangPair(".dfm", "pascal"),
    ExtLangPair(".dlang", "dlang"),
    ExtLangPair(".dpr", "pascal"),
    ExtLangPair(".dts", "devicetree"),
    ExtLangPair(".dtsi", "devicetree"),
    ExtLangPair(".dtso", "devicetree"),
    ExtLangPair(".edn", "clojure"),
    ExtLangPair(".el", "emacs"),
    ExtLangPair(".elm", "elm"),
    ExtLangPair(".elv", "elvish"),
    ExtLangPair(".emacs", "emacs"),
    ExtLangPair(".erl", "erlang"),
    ExtLangPair(".es", "erlang"),
    ExtLangPair(".escript", "erlang"),
    ExtLangPair(".eta", "eta"),
    ExtLangPair(".factor", "factor"),
    ExtLangPair(".fr", "frege"),
    ExtLangPair(".fs", "fsharp"),
    ExtLangPair(".fsi", "fsharp"),
    ExtLangPair(".fsx", "fsharp"),
    ExtLangPair(".gleam", "gleam"),
    ExtLangPair(".gql", "graphql"),
    ExtLangPair(".graphql", "graphql"),
    ExtLangPair(".graphqls", "graphql"),
    ExtLangPair(".groovy", "groovy"),
    ExtLangPair(".gs", "gosu"),
    ExtLangPair(".hack", "hack"),
    ExtLangPair(".hck", "hack"),
    ExtLangPair(".hcl", "hcl"),
    ExtLangPair(".hhi", "hack"),
    ExtLangPair(".hjson", "hjson"),
    ExtLangPair(".hlsl", "hlsl"),
    ExtLangPair(".hrl", "erlang"),
    ExtLangPair(".hrl", "erlang"),
    ExtLangPair(".idr", "idris"),
    ExtLangPair(".imba", "imba"),
    ExtLangPair(".io", "io"),
    ExtLangPair(".its", "devicetree"),
    ExtLangPair(".janet", "janet"),
    ExtLangPair(".jdn", "janet"),
    ExtLangPair(".jelly", "jelly"),
    ExtLangPair(".jl", "julia"),
    ExtLangPair(".joke", "clojure"),
    ExtLangPair(".joker", "clojure"),
    ExtLangPair(".jule", "jule"),
    ExtLangPair(".less", "less"),
    ExtLangPair(".lidr", "idris"),
    ExtLangPair(".lisp", "lisp"),
    ExtLangPair(".lpr", "pascal"),
    ExtLangPair(".ls", "livescript"),
    ExtLangPair(".lsc", "lisp"),
    ExtLangPair(".lsp", "lisp"),
    ExtLangPair(".lucee", "lucee"),
    ExtLangPair(".m", "objective-c"),
    ExtLangPair(".mak", "make"),
    ExtLangPair(".makefile", "make"),
    ExtLangPair(".mk", "make"),
    ExtLangPair(".ml", "ocaml"),
    ExtLangPair(".mli", "ocaml"),
    ExtLangPair(".mm", "objective-c"),
    ExtLangPair(".mojo", "mojo"),
    ExtLangPair(".nh", "newick"),
    ExtLangPair(".nhx", "newick"),
    ExtLangPair(".nim", "nim"),
    ExtLangPair(".nim.cfg", "nimble"),
    ExtLangPair(".nim.cfg", "nimble"),
    ExtLangPair(".nimble", "nimble"),
    ExtLangPair(".nimble.cfg", "nimble"),
    ExtLangPair(".nimble.json", "nimble"),
    ExtLangPair(".nimble.toml", "nimble"),
    ExtLangPair(".nomad", "hcl"),
    ExtLangPair(".nu", "nushell"),
    ExtLangPair(".nushell", "nushell"),
    ExtLangPair(".nwk", "newick"),
    ExtLangPair(".odin", "odin"),
    ExtLangPair(".pas", "pascal"),
    ExtLangPair(".pascal", "pascal"),
    ExtLangPair(".pgsql", "sql"),
    ExtLangPair(".pharo", "pharo"),
    ExtLangPair(".pl", "perl"),
    ExtLangPair(".pm", "perl"),
    ExtLangPair(".pony", "pony"),
    ExtLangPair(".pp", "pascal"),
    ExtLangPair(".ps1", "powershell"),
    ExtLangPair(".purs", "purescript"),
    ExtLangPair(".pxd", "cython"),
    ExtLangPair(".pyx", "cython"),
    ExtLangPair(".qb64", "qb64"),
    ExtLangPair(".qml", "qml"),
    ExtLangPair(".r", "r"),
    ExtLangPair(".raku", "raku"),
    ExtLangPair(".rakudoc", "raku"),
    ExtLangPair(".rakudoc", "rakudo"),
    ExtLangPair(".rd", "r"),
    ExtLangPair(".red", "red"),
    ExtLangPair(".reds", "red"),
    ExtLangPair(".ring", "ring"),
    ExtLangPair(".rkt", "racket"),
    ExtLangPair(".rktd", "racket"),
    ExtLangPair(".rktl", "racket"),
    ExtLangPair(".rsx", "r"),
    ExtLangPair(".sc", "scheme"),
    ExtLangPair(".sch", "scheme"),
    ExtLangPair(".scheme", "scheme"),
    ExtLangPair(".scm", "scheme"),
    ExtLangPair(".scss", "scss"),
    ExtLangPair(".sld", "scheme"),
    ExtLangPair(".smali", "smali"),
    ExtLangPair(".sql", "sql"),
    ExtLangPair(".sqlite", "sql"),
    ExtLangPair(".sqlite3", "sql"),
    ExtLangPair(".sty", "latex"),
    ExtLangPair(".sv", "verilog"),
    ExtLangPair(".svelte", "svelte"),
    ExtLangPair(".tex", "latex"),
    ExtLangPair(".tf", "hcl"),
    ExtLangPair(".tfvars", "hcl"),
    ExtLangPair(".v", "v"),
    ExtLangPair(".v", "verilog"),
    ExtLangPair(".vale", "vale"),
    ExtLangPair(".vhd", "vhdl"),
    ExtLangPair(".vhdl", "vhdl"),
    ExtLangPair(".vue", "vue"),
    ExtLangPair(".workflow", "hcl"),
    ExtLangPair(".xhtml", "xml"),
    ExtLangPair(".xlf", "xml"),
    ExtLangPair(".xml", "xml"),
    ExtLangPair(".xrl", "erlang"),
    ExtLangPair(".xsd", "xml"),
    ExtLangPair(".xsl", "xml"),
    ExtLangPair(".yrl", "erlang"),
    ExtLangPair(".zig", "zig"),
    ExtLangPair("BSDmakefile", "make"),
    ExtLangPair("CMakefile", "cmake"),
    ExtLangPair("Cask", "emacs"),
    ExtLangPair("Dockerfile", "docker"),
    ExtLangPair("Emakefile", "erlang"),
    ExtLangPair("GNUmakefile", "make"),
    ExtLangPair("Justfile", "just"),
    ExtLangPair("Kbuild", "make"),
    ExtLangPair("Makefile", "make"),
    ExtLangPair("Makefile.am", "make"),
    ExtLangPair("Makefile.boot", "make"),
    ExtLangPair("Makefile.in", "make"),
    ExtLangPair("Makefile.inc", "make"),
    ExtLangPair("Makefile.wat", "make"),
    ExtLangPair("Rakefile", "rake"),
    ExtLangPair("_emacs", "emacs"),
    ExtLangPair("makefile", "make"),
    ExtLangPair("makefile.sco", "make"),
    ExtLangPair("mkfile", "make"),
    ExtLangPair("rebar.config", "erlang"),
)
# spellchecker:on
"""A tuple of `ExtLangPair`."""

CONFIG_FILE_LANGUAGES = frozenset({
    "bash",
    "yaml",
    "json",
    "toml",
    "ini",
    "make",
    "cmake",
    "xml",
    "cfg",
    "properties",
})


def get_ext_lang_pairs() -> Generator[ExtLangPair]:
    """Yield all `ExtLangPair` instances for code, config, and docs files."""
    yield from (*CODE_FILES_EXTENSIONS, *DOC_FILES_EXTENSIONS)
