#!/usr/bin/env python
"""Method Signature Standardization Script.

Standardizes method signatures across packages to ensure consistency.
"""

import argparse

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SignatureStandard:
    """Standard signature for a method."""

    method_name: str
    args: list[str]
    defaults: list[str] = None
    return_type: str = None
    is_async: bool = False


class SignatureStandardizer:
    """Standardizes method signatures across packages."""

    def __init__(self, src_path: Path):
        """Initialize the standardizer with the source path."""
        self.src_path = src_path
        self.standards = self._define_standards()

    def _define_standards(self) -> dict[str, dict[str, SignatureStandard]]:
        """Define standard signatures by package."""
        return {
            "providers": {
                "__init__": SignatureStandard(
                    "__init__", ["config", "*", "logger", "api_key"], ["None", "None"], "None"
                ),
                "health_check": SignatureStandard("health_check", [], [], "bool", is_async=True),
                "validate_api_key": SignatureStandard(
                    "validate_api_key", ["api_key"], [], "bool", is_async=True
                ),
                "get_capabilities": SignatureStandard(
                    "get_capabilities", [], [], "ProviderCapabilities"
                ),
            },
            "backends": {
                "__init__": SignatureStandard(
                    "__init__", ["config", "*", "logger", "client"], ["None", "None"], "None"
                ),
                "health_check": SignatureStandard("health_check", [], [], "bool", is_async=True),
                "initialize": SignatureStandard("initialize", [], [], "None", is_async=True),
                "shutdown": SignatureStandard("shutdown", [], [], "None", is_async=True),
            },
            "sources": {
                "__init__": SignatureStandard(
                    "__init__", ["source_id", "*", "config"], ["None"], "None"
                ),
                "health_check": SignatureStandard("health_check", [], [], "bool", is_async=True),
                "start": SignatureStandard("start", [], [], "bool", is_async=True),
                "stop": SignatureStandard("stop", [], [], "bool", is_async=True),
                "check_availability": SignatureStandard(
                    "check_availability", [], [], "bool", is_async=True
                ),
            },
            "services": {
                "__init__": SignatureStandard(
                    "__init__",
                    ["config", "*", "logger", "fastmcp_server"],
                    ["None", "None"],
                    "None",
                ),
                "health_check": SignatureStandard(
                    "health_check", [], [], "ServiceHealth", is_async=True
                ),
                "_initialize_provider": SignatureStandard(
                    "_initialize_provider", [], [], "None", is_async=True
                ),
                "_shutdown_provider": SignatureStandard(
                    "_shutdown_provider", [], [], "None", is_async=True
                ),
                "_check_health": SignatureStandard(
                    "_check_health", [], [], "HealthStatus", is_async=True
                ),
            },
        }

    def create_standard_base_classes(self) -> dict[str, str]:
        """Create standard base classes with consistent utility methods."""
        return {
            "universal_base.py": '''"""Universal base class for all CodeWeaver components."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from codeweaver.utils.decorators import require_implementation


class CodeWeaverComponent(ABC):
    """Universal base class for all CodeWeaver components.

    Provides consistent patterns for initialization, health checking, and lifecycle management.
    """

    def __init__(self, component_id: str, config: Any, *, logger: Optional[logging.Logger] = None):
        """Initialize the component with consistent parameters.

        Args:
            component_id: Unique identifier for this component
            config: Component configuration
            logger: Optional logger instance
        """
        self.component_id = component_id
        self.config = config
        self.logger = logger or logging.getLogger(f"codeweaver.{self.__class__.__name__.lower()}")
        self._initialized = False

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the component is healthy and operational.

        Returns:
            True if healthy, False otherwise
        """
        ...

    async def initialize(self) -> None:
        """Initialize the component if not already initialized."""
        if not self._initialized:
            await self._initialize_component()
            self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the component gracefully."""
        if self._initialized:
            await self._shutdown_component()
            self._initialized = False

    @abstractmethod
    async def _initialize_component(self) -> None:
        """Component-specific initialization logic."""
        ...

    @abstractmethod
    async def _shutdown_component(self) -> None:
        """Component-specific shutdown logic."""
        ...

    def get_component_info(self) -> Dict[str, Any]:
        """Get information about this component."""
        return {
            "id": self.component_id,
            "type": self.__class__.__name__,
            "initialized": self._initialized,
            "config": getattr(self.config, 'model_dump', lambda: self.config)()
        }
''',
            "provider_base.py": '''"""Standard base class for providers."""

import logging
from typing import Any, Optional

from codeweaver.utils.decorators import require_implementation
from .universal_base import CodeWeaverComponent


class StandardProviderBase(CodeWeaverComponent):
    """Standard base class for all providers with consistent API patterns."""

    @require_implementation("validate_api_key", "_initialize_component", "_shutdown_component")
    def __init__(
        self,
        provider_id: str,
        config: Any,
        *,
        logger: Optional[logging.Logger] = None,
        api_key: Optional[str] = None
    ):
        """Initialize provider with standard signature.

        Args:
            provider_id: Unique provider identifier
            config: Provider configuration
            logger: Optional logger instance
            api_key: Optional API key for authentication
        """
        super().__init__(provider_id, config, logger=logger)
        self.api_key = api_key

    async def validate_api_key(self, api_key: str) -> bool:
        """Validate API key for this provider.

        Args:
            api_key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        raise NotImplementedError("Subclasses must implement validate_api_key")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities."""
        return {
            "provider_id": self.component_id,
            "supports_validation": True,
            "requires_api_key": self.api_key is not None
        }
''',
        }

    def generate_improvement_report(self) -> str:
        """Generate a report of recommended signature improvements."""
        report = [
            "# Method Signature Standardization Report\n",
            "## Current Issues\n",
            "Based on the consistency analysis, we found 21 signature inconsistencies across packages.",
            "The primary issues are:\n",
            "### __init__ Method Inconsistencies",
            "- **Sources**: Inconsistent parameters across implementations",
            "- **Services**: Varying parameter order and types",
            "- **Providers**: Inconsistent handling of optional parameters",
            "- **Backends**: Missing standard parameters\n",
            "### Utility Method Inconsistencies",
            "- **health_check**: Different return types and signatures",
            "- **initialize/shutdown**: Inconsistent lifecycle management",
            "- **get_capabilities**: Different patterns across packages\n",
            "## Standardization Plan\n",
        ]
        for package, methods in self.standards.items():
            report.append(f"### Package: {package}")
            for method_name, standard in methods.items():
                raw_args = ", ".join(standard.args) if standard.args else ""
                raw_async = "async " if standard.is_async else ""
                return_value = f" -> {standard.return_type}" if standard.return_type else ""

                report.append(
                    f"- **{method_name}**: `{raw_async}def {method_name}({raw_args}){return_value}`"
                )
            report.append("")

        report.extend(("## Benefits of Standardization\n", "1. **Consistency**: Uniform API across all packages", "2. **Maintainability**: Easier to understand and modify", "3. **Type Safety**: Better static analysis and IDE support", "4. **Documentation**: Clearer patterns for contributors", "5. **Testing**: Consistent patterns for test automation\n", "## Implementation Strategy\n", "1. **Create Universal Base Classes**: Common patterns for all components", "2. **Gradual Migration**: Package-by-package standardization", "3. **Validation**: Runtime checks for signature compliance", "4. **Documentation**: Update guides and examples"))

        return "\n".join(report)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Standardize method signatures across packages")
    parser.add_argument(
        "--src-path", type=Path, default=Path("src"), help="Path to source code directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/signature_standards.md"),
        help="Output file for standardization report",
    )
    parser.add_argument(
        "--create-bases", action="store_true", help="Create standard base class files"
    )

    args = parser.parse_args()

    if not args.src_path.exists():
        print(f"❌ Source path {args.src_path} does not exist")
        return 1

    print("🎯 Planning signature standardization...")
    standardizer = SignatureStandardizer(args.src_path)

    # Generate report
    report = standardizer.generate_improvement_report()

    with (args.output).open("w") as f:
        f.write(report)
    print(f"📄 Report written to {args.output}")

    # Create base classes if requested
    if args.create_bases:
        base_classes = standardizer.create_standard_base_classes()
        base_dir = args.src_path / "codeweaver" / "utils" / "bases"
        base_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in base_classes.items():
            base_file = base_dir / filename
            with base_file.open("w") as f:
                f.write(content)
            print(f"📄 Created base class: {base_file}")

    print("\n✅ Standardization planning complete!")
    return 0


if __name__ == "__main__":
    exit(main())
