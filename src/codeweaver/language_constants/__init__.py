"""
The language constants module provides default ast-grep and NER patterns for various programming languages. Users can override or extend these in the configuration. Contributions for new languages are welcome!

## Consistency Guidelines

- One module per language
- general code domain patterns in `codeweaver/language_constants/general.py`
- Use lowercase for module names
- constants per language (implementing all encouraged but anything is better than nothing):
  - `DEFAULT_{language}_AST_GREP_PATTERNS`
  - `DEFAULT_{language}_NER_PATTERNS`
    - Should only include patterns that are *specific* to the language.
    - General patterns should be in `codeweaver/language_constants/general.py`.

  - Others allowed as needed, please don't duplicate information available in `cw_types/language.py`

### Typing and Data Structures

- If a value is a sequence, use a tuple for immutability unless the sequence is expected to change at runtime.
  - If it's a mapping, use `types.MappingProxyType` for immutability (dict if it needs to be mutable).
- Use `__all__` to define public API for each module
- Use `__init__.py` to import all constants for easy access
- Use docstrings to describe the purpose of each constant and module
"""

from codeweaver.language_constants.general import DEFAULT_AST_GREP_PATTERNS, DEFAULT_NER_PATTERNS
from codeweaver.language_constants.javascript_family import (
    DEFAULT_JAVASCRIPT_AST_GREP_PATTERNS,
    DEFAULT_JAVASCRIPT_NER_PATTERNS,
)
from codeweaver.language_constants.python import (
    DEFAULT_PYTHON_AST_GREP_PATTERNS,
    DEFAULT_PYTHON_NER_PATTERNS,
)


ALL_NER_PATTERNS = (
    *DEFAULT_NER_PATTERNS,
    *DEFAULT_PYTHON_NER_PATTERNS,
    *DEFAULT_JAVASCRIPT_NER_PATTERNS,
)

ALL_AST_GREP_PATTERNS = (
    *DEFAULT_AST_GREP_PATTERNS,
    *DEFAULT_PYTHON_AST_GREP_PATTERNS,
    *DEFAULT_JAVASCRIPT_AST_GREP_PATTERNS,
)

__all__ = (
    "ALL_AST_GREP_PATTERNS",
    "ALL_NER_PATTERNS",
    "DEFAULT_AST_GREP_PATTERNS",
    "DEFAULT_JAVASCRIPT_AST_GREP_PATTERNS",
    "DEFAULT_JAVASCRIPT_NER_PATTERNS",
    "DEFAULT_NER_PATTERNS",
    "DEFAULT_PYTHON_AST_GREP_PATTERNS",
    "DEFAULT_PYTHON_NER_PATTERNS",
)
