# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Decorators used across the CodeWeaver project.
"""

import functools
import logging

from typing import Any

from codeweaver.cw_types import T


logger = logging.getLogger(__name__)


def _find_class_in_bases(bases: tuple[type, ...]) -> type[T] | None:
    """
    Find a class in the bases of the given class by name.
    """
    config_classes = {"DataSource", "VectorBackend", "EmbeddingProvider", "ServiceProvider"}
    return next(
        (
            base
            for base in bases
            if (isinstance(base, type) and base.__name__ in config_classes)
            or any(
                name
                for name in config_classes
                if name in base.__name__ and not base.__name__.startswith("_")
            )
        ),
        None,
    )


def _get_class_to_subclass(cls: type[T]) -> type[T] | None:
    """
    Get the name of the class to subclass to implement the not-implemented class.
    """
    if (bases := cls.__bases__) and (found_class := _find_class_in_bases(bases)):
        return found_class
    return None


class NotImplementedClassError(Exception):
    """Enhanced exception for classes marked with @not_implemented decorator."""

    def __init__(
        self, message: str, class_name: str | None = None, suggestions: list[str] | None = None
    ):
        """Initialize with enhanced context.

        Args:
            message: Error message
            class_name: Name of the class that is not implemented
            suggestions: List of suggested alternatives or next steps
        """
        self.class_name = class_name
        self.suggestions = suggestions or []

        full_message = f"{self.class_name}: {message}" if self.class_name else message
        if self.suggestions:
            suggestions_text = "\n".join(f"  - {suggestion}" for suggestion in self.suggestions)
            full_message += f"\n\nSuggestions:\n{suggestions_text}"

        super().__init__(full_message)


def not_implemented(
    message: str | None = None,
    suggestions: list[str] | None = None,
    *,
    allow_subclassing: bool = True,
) -> callable:
    """Class decorator that prevents instantiation of placeholder implementations.

    # @not_implemented Decorator

    A class decorator for marking placeholder implementations that provides clarity and prevents accidental instantiation.

    ## Overview

    The `@not_implemented` decorator is designed for classes that are planned but not yet implemented. It:

    - ✅ **Prevents instantiation** with clear error messages
    - ✅ **Preserves class structure** for documentation and type checking
    - ✅ **Allows subclassing** by default (configurable)
    - ✅ **Provides helpful suggestions** for alternatives
    - ✅ **Maintains method signatures** for IDE support

    ## Usage

    ### Basic Usage

    The default message and suggestions are very clear and informative out of the box, so using it can be as simple as:
    ```python
    @not_implemented
    class MyPlaceholderClass:
        def my_method(self):
            pass  # Implementation details for documentation
    ```

    ### Advanced Usage with Custom Messages

    You can also provide a custom message and suggestions to guide users on what to do next:

    ```python
    @not_implemented(
        message="Git integration requires GitPython dependency",
        suggestions=[
            "Install GitPython: pip install GitPython",
            "Install pygit2: pip install pygit2",
            "Use filesystem source as alternative",
            "Implement using subprocess calls",
        ],
    )
    class GitRepositorySourceProvider(AbstractDataSource):
        def discover_content(self, config):
            pass  # Implementation details for documentation
    ```

    ### Preventing Subclassing

    ## Features

    ### 1. Instantiation Prevention

    The decorator prevents instantiation of the class directly by wrapping the `__new__()` method, raising a `NotImplementedClassError` with a clear message and suggestions.

    ### 2. Implementation Status Checking

    ```python
    # Check if a class is implemented
    print(GitRepositorySourceProvider.is_implemented())  # False

    # Get detailed implementation info
    info = GitRepositorySourceProvider.get_implementation_info()
    print(info["message"])  # Custom error message
    print(info["suggestions"])  # List of suggestions
    print(info["implemented"])  # False
    ```

    ### 3. Subclassing Support

    ```python
    class MyGitImplementation(GitRepositorySourceProvider):
        def __init__(self):
            super().__init__("git", "my-implementation")

        async def discover_content(self, config):
            return []  # Your implementation here


    # This works - subclasses can be instantiated
    my_source = MyGitImplementation()
    ```

    ### 4. Method Preservation

    The decorator preserves all method signatures, so:
    - IDE autocompletion works
    - Type checking works
    - Documentation tools work
    - Methods can be inspected

    ```python
    # All methods are accessible for inspection
    print(hasattr(GitRepositorySourceProvider, "discover_content"))  # True
    print(GitRepositorySourceProvider.discover_content.__doc__)  # Method docstring
    ```

    ## Applied Examples

    ### Git Repository Source

    ```python
    @not_implemented(
        message="We haven't implemented Git repository source yet",
        suggestions=[
            "You can use the Filesystem source for local files",
            "Implement your own using the `SourceData` protocol",
            "Consider contributing to the project to help us implement this!",
            "Or open an issue to discuss your use case"
        ]
    )
    class GitRepositorySourceProvider(AbstractDataSource):
        # Class implementation with full method signatures...
    ```

    ## Error Handling

    The decorator raises `NotImplementedClassError` (not Python's built-in `NotImplementedError`) to avoid conflicts:

    ```python
    from codeweaver.sources.decorators import NotImplementedClassError

    try:
        GitRepositorySourceProvider()
    except NotImplementedClassError as e:
        print(f"Class not implemented: {e.class_name}")
        print(f"Message: {e}")
        print(f"Suggestions: {e.suggestions}")
    ```
    """

    def decorator(cls: type[T]) -> type[T]:
        # Store original class information
        original_name = cls.__name__
        original_doc = cls.__doc__ or ""
        implementing_class = _get_class_to_subclass(cls)

        # Create default message if none provided
        default_message = (
            "Sorry! :open_hands: We haven't implemented this integration yet. ☹️\n"
            "If this were 1995, we would have a construction gif here. :construction:\n\n"
            f"Adding {original_name} is on our roadmap, but we need your help to prioritize it!\n\n"
            "Thank you for your understanding! :pray:"
        )
        error_message = message or default_message

        default_suggestions = [
            "Contribute to codeweaver to help implement this feature.",
            "[Open an issue](https://github.com/knitli/codeweaver-mcp/issues/) to discuss your use case.",
            "Use your another source or provider that's already implemented.",
            f"Implement your own by using the `{implementing_class}` protocol.",
            "Throwing money at us may help too, but we prefer code contributions! :money_with_wings:",
        ]

        # Add not-implemented marker to docstring
        if (
            note
            := f"Note: This is a placeholder! We haven't implemented this yet, but you can by implementing `{implementing_class}`"
        ) not in original_doc:
            implementation_note = f"\n\n{note} "
            cls.__doc__ = (original_doc + implementation_note).strip()

        # Store metadata about the not-implemented status
        cls.__not_implemented__ = True
        cls.__not_implemented_message__ = error_message
        cls.__not_implemented_suggestions__ = suggestions or default_suggestions

        # Override __new__ to prevent instantiation
        original_new = cls.__new__

        @functools.wraps(original_new)
        def __new__(cls_inner, *args: Any, **kwargs: dict[str, Any]):  # noqa: N807
            """Wraps the original __new__ method to prevent instantiation."""
            # Check if this is direct instantiation or subclass instantiation
            if cls_inner is cls and not allow_subclassing:
                raise NotImplementedClassError(
                    error_message, class_name=original_name, suggestions=suggestions
                )
            if cls_inner is cls:
                # Direct instantiation of the not-implemented class
                raise NotImplementedClassError(
                    error_message, class_name=original_name, suggestions=suggestions
                )
                # Subclass instantiation - allow if allow_subclassing is True
            if allow_subclassing:
                logger.info(
                    "Creating subclass instance of not-implemented class %s: %s",
                    original_name,
                    cls_inner.__name__,
                )
                # Call the original __new__ method
                return (
                    original_new(cls_inner)
                    if original_new is object.__new__
                    else original_new(cls_inner, *args, **kwargs)
                )
            raise NotImplementedClassError(
                f"Sorry! :open_hands:\n You can't subclass {original_name} because it is not implemented yet. If you want to implement it, please open a PR or issue on our GitHub repository. You can implement it by implementing the protocol {implementing_class}.",
                class_name=original_name,
                suggestions=suggestions,
            )

        cls.__new__ = staticmethod(__new__)

        # Add class method to check implementation status
        @classmethod
        def is_implemented(cls_inner: type) -> bool:
            """Check if this class is implemented.

            Returns:
                False for classes marked with @not_implemented
            """
            return not getattr(cls_inner, "__not_implemented__", False)

        cls.is_implemented = is_implemented

        # Add class method to get implementation info
        @classmethod
        def get_implementation_info(cls_inner: type) -> dict[str, Any]:
            """Get information about the implementation status.

            Returns:
                Dictionary with implementation status and details
            """
            return {
                "implemented": cls_inner.is_implemented(),
                "class_name": original_name,
                "message": getattr(cls_inner, "__not_implemented_message__", None),
                "suggestions": getattr(cls_inner, "__not_implemented_suggestions__", []),
                "allow_subclassing": allow_subclassing,
            }

        cls.get_implementation_info = get_implementation_info

        # Log the decoration
        logger.debug(
            "Applied @not_implemented decorator to %s (allow_subclassing=%s)",
            original_name,
            allow_subclassing,
        )

        return cls

    return decorator


def require_implementation(*methods: str) -> callable:
    """Method decorator that ensures specific methods are implemented in subclasses.

    This is useful for abstract base classes where certain methods must be
    overridden by concrete implementations.

    Args:
        *methods: Names of methods that must be implemented

    Returns:
        Method decorator

    Example:
        ```python
        class BaseSource:
            @require_implementation("discover_content", "read_content")
            def __init__(self):
                pass
        ```
    """

    def decorator(func: callable) -> callable:
        @functools.wraps(func)
        def wrapper(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
            class_name = self.__class__.__name__
            base_class = func.__qualname__.split(".")[0]

            # Check if required methods are implemented
            missing_methods = []
            for method_name in methods:
                method = getattr(self.__class__, method_name, None)
                if method is None:
                    missing_methods.append(method_name)
                    continue

                # Check if method is just the abstract version
                if hasattr(method, "__qualname__"):
                    method_class = method.__qualname__.split(".")[0]
                    if method_class == base_class:
                        missing_methods.append(method_name)

            if missing_methods:
                raise NotImplementedError(
                    f"The following methods must be implemented in {class_name}: {', '.join(missing_methods)}",
                    class_name=class_name,
                    suggestions=[
                        f"Override {method}() in your {class_name} implementation"
                        for method in missing_methods
                    ],
                )

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class FeatureNotEnabledError(Exception):
    """Exception raised when a feature flag is required but not enabled."""

    def __init__(self, feature_name: str, msg: str | None = None):
        """Initialize with the feature name."""
        if msg:
            super().__init__(
                f"{msg} \n\nYou can enable it by running: `[uv] pip install codeweaver[{feature_name}]`"
            )
        super().__init__(f"Sorry! :open_hands: The feature '{feature_name}' is not enabled. ")


def feature_flag_required(feature_name: str, dependencies: tuple[str] | None) -> callable:
    """Decorator to mark classes that require a specific feature flag to be enabled.

    Args:
        feature_name: Name of the feature flag required for this method
        dependencies: Optional tuple of dependencies that need to be installed for this feature

    Returns:
        Method decorator that checks if the feature flag is enabled
    """

    def decorator(cls: type[T]) -> type[T]:
        """Decorator that checks if the feature flag is enabled before instantiating the class."""
        class_name = cls.__name__
        original_init = cls.__init__
        original_doc = cls.__doc__ or ""
        import importlib.util

        if dependencies and all(importlib.util.find_spec(dep) is not None for dep in dependencies):
            cls._dependencies_met = True
        else:
            cls._dependencies_met = False
            if (
                note
                := f"\n\nNote: {class_name} only works with '{feature_name}' enabled.\n\n Install with `[uv] pip install codeweaver[{feature_name}]`\n"
            ) not in original_doc:
                cls.__doc__ = (original_doc + note).strip()
        cls._feature_flag_required = feature_name

        @functools.wraps(original_init)
        def wrapper(self, *args: Any, **kwargs: dict[str, Any]) -> Any:
            if self._dependencies_met:
                return original_init(self, *args, **kwargs)
            raise FeatureNotEnabledError(
                feature_name,
                msg=f"{class_name} requires the '{feature_name}' feature flag to be enabled.",
            )

        cls.__init__ = wrapper
        return cls

    return decorator
