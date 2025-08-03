"""
The language constants module provides default ast-grep and NER patterns for various programming languages. Users can override or extend these in the configuration. Contributions for new languages are welcome!

## Consistency Guidelines

- One module per language
- general code domain patterns in `codeweaver/language_constants/general.py`
- Use lowercase for module names
- constants per language (implementing all encouraged but anything is better than nothing):
  - `LANGUAGE`: Language name in lowercase (e.g. `python`)
  - `DEFAULT_{language_name}AST_GREP_PATTERNS`: Default ast-grep pattern for the language (e.g. `DEFAULT_PYTHON_AST_GREP_PATTERNS`)
  - `DEFAULT_{language_name}_NER_PATTERNS`: Default NER pattern for the language
  - `DEFAULT_EXTENSIONS`: Default file extensions for the language (a tuple of strings without leading dot)

  - Others allowed as needed.

### Typing and Data Structures

- If a value is a sequence, use a tuple for immutability unless the sequence is expected to change at runtime.
  - If it's a mapping, use `types.MappingProxyType` for immutability (dict if it needs to be mutable).
- Use `__all__` to define public API for each module
- Use `__init__.py` to import all constants for easy access
- Use docstrings to describe the purpose of each constant and module
"""

from codeweaver.language_constants.general import DEFAULT_NER_PATTERNS


ALL_NER_PATTERNS = (*DEFAULT_NER_PATTERNS,)

__all__ = (
    "ALL_NER_PATTERNS",
)
