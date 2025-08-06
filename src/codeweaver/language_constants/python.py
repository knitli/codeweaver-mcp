# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Default language constants for Python."""

from types import MappingProxyType


LANGUAGE = "python"


DEFAULT_PYTHON_AST_GREP_PATTERNS = (
    ("import_declaration", "import"),
    ("import_from_declaration", "import_from"),
    ("function_definition", "function"),
    ("class_definition", "class"),
    ("async_function_definition", "async_function"),
    ("type_expression", "type"),
    ("type_alias", "type_alias"),
    ("return_type", "return_type"),
)


DEFAULT_PYTHON_NER_PATTERNS = (
    MappingProxyType({
        "label": "LANGUAGE",
        "pattern": [{"LOWER": {"IN": ["python", "py", "python3", "py3"]}}],
    }),
    MappingProxyType({
        "label": "CODE_ELEMENT",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "abc",
                        "abstractbaseclass",
                        "class",
                        "component",
                        "constant",
                        "dataclass",
                        "decorator",
                        "endpoint",
                        "function",
                        "generic",
                        "generator",
                        "interface",
                        "lambda",
                        "method",
                        "model",
                        "module",
                        "package",
                        "protocol",
                        "route",
                        "service",
                        "variable",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "FRAMEWORK",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "aiohttp",
                        "bottle",
                        "dash",
                        "django",
                        "falcon",
                        "fastapi",
                        "fastmcp",
                        "flask",
                        "jina",
                        "jinja",
                        "matplotlib",
                        "mkdocs",
                        "numpy",
                        "pandas",
                        "polars",
                        "pydantic",
                        "pyramid",
                        "pytorch",
                        "sanic",
                        "scikit-learn",
                        "tensorflow",
                        "tornado",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "DATABASE",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        # keep this to python related patterns
                        "docarray",
                        "sqlalchemy",
                        "sqlite",
                        "sqlmodel",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "FILE_TYPE",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "ipynb",
                        "jupyter",
                        "jupyter-notebook",
                        "py",
                        "py3",
                        "pyproject",
                        "python",
                        "requirements",
                        "setup",
                        "setup-py",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "TOOL",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "bandit",
                        "black",
                        "codeweaver",  # hey, we get to be in our own list!
                        "coverage",
                        "conda",
                        "flake8",
                        "gunicorn",
                        "hatch",
                        "isort",
                        "mypy",
                        "pip",
                        "pipenv",
                        "poetry",
                        "pre-commit",
                        "pycharm",
                        "pylance",
                        "pylint",
                        "pytest",
                        "ruff",
                        "rye",
                        "sphinx",
                        "starlette",
                        "uv",
                        "uv-pip",
                        "uvicorn",
                        "venv",
                        "virtualenv",
                    ]
                }
            }
        ],
    }),
)
