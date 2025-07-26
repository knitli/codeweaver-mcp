#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Command-line interface for CodeWeaver configuration management.

Provides tools for validating and generating configuration files.
"""

import argparse
import logging
import sys

from pathlib import Path
from typing import NoReturn

from codeweaver.config import (
    CodeWeaverConfig,
    ConfigManager,
    ConfigurationError,
    ConfigValidator,
    setup_development_config,
    setup_production_config,
    setup_testing_config,
)


def setup_logging(*, verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a configuration file."""
    try:
        _validate_cmd(args)
    except ConfigurationError as e:
        print(f"❌ Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def _validate_cmd(args: argparse.Namespace) -> None:
    """Internal command to validate a configuration file."""
    # Load configuration using new system
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config_file)

    # Validate configuration
    validator = ConfigValidator()
    result = validator.validate_configuration(config)

    print(f"✅ Configuration file: {args.config_file}")
    print(f"📋 Format: New (v{config.config_version})")
    print(f"🏗️  Backend: {config.backend.provider}")
    print(f"🤖 Provider: {config.providers.provider}")

    if result.errors:
        print(f"\n❌ Validation Errors ({len(result.errors)}):")
        for i, error in enumerate(result.errors, 1):
            print(f"  {i}. {error}")

    if result.warnings:
        print(f"\n⚠️  Validation Warnings ({len(result.warnings)}):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print("\n✅ No validation warnings found")

    if args.show_effective_config:
        print("\n📄 Effective Configuration:")
        import json

        config_dict = config.model_dump(exclude_unset=True)
        print(json.dumps(config_dict, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate example configurations."""
    from codeweaver.config import ConfigSchema

    # Available scenarios
    scenarios = {
        "development": lambda: setup_development_config(),
        "production": lambda: setup_production_config(
            backend_url="http://localhost:6333",
            embedding_api_key="your-api-key",
            source_paths=["./src"],
        ),
        "testing": lambda: setup_testing_config(),
        "example": lambda: ConfigSchema.generate_example_config("toml"),
    }

    if args.scenario == "all":
        print("📚 Available deployment scenarios:")
        for name in scenarios:
            print(f"  • {name}")

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for name, config_generator in scenarios.items():
                output_file = output_dir / f"{name}.toml"

                if name == "example":
                    content = config_generator()
                else:
                    config = config_generator()
                    manager = ConfigManager(config)
                    content = manager.export_config("toml", include_defaults=True)

                with output_file.open("w") as f:
                    f.write(content)
                print(f"📁 Generated: {output_file}")

    elif args.scenario in scenarios:
        config_generator = scenarios[args.scenario]

        if args.scenario == "example":
            content = config_generator()
        else:
            config = config_generator()
            manager = ConfigManager(config)
            content = manager.export_config("toml", include_defaults=True)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                f.write(content)
            print(f"✅ Generated configuration: {output_path}")
        else:
            print(f"📄 Configuration for '{args.scenario}':")
            print(content)

    else:
        print(f"❌ Unknown scenario: {args.scenario}")
        print(f"Available scenarios: {', '.join(scenarios.keys())}")
        sys.exit(1)


def cmd_check_compatibility(args: argparse.Namespace) -> None:
    """Check backend and provider compatibility."""
    from codeweaver.config import BackendConfigBuilder, ConfigValidator, ProviderConfigBuilder

    # Create test configurations
    backend_config = BackendConfigBuilder(args.backend).build()
    provider_config = ProviderConfigBuilder(args.provider).build()

    # Create a test configuration for validation
    test_config = CodeWeaverConfig(backend=backend_config, providers=provider_config)

    # Validate
    validator = ConfigValidator()
    result = validator.validate_configuration(test_config)

    print("🔍 Compatibility Check:")
    print(f"  Backend: {args.backend}")
    print(f"  Provider: {args.provider}")

    if result.is_valid:
        print("✅ Compatible combination")
    else:
        print("⚠️  Compatibility issues found:")
        for error in result.errors:
            print(f"  • {error}")

    if result.warnings:
        print("⚠️  Warnings:")
        for warning in result.warnings:
            print(f"  • {warning}")


def cmd_info(args: argparse.Namespace) -> None:
    """Show configuration system information."""
    print("📊 CodeWeaver Configuration System")
    print("==================================")
    print("🏗️  Supported Backends:")

    # Hardcode the supported backends from our new system
    backends = ["qdrant", "pinecone", "weaviate"]
    for backend in backends:
        print(f"  • {backend}")
    print()

    # Supported providers
    print("🤖 Supported Embedding Providers:")
    providers = ["voyage-ai", "openai", "cohere", "huggingface", "sentence-transformers"]
    for provider in providers:
        print(f"  • {provider}")
    print()

    # Configuration locations
    print("📁 Configuration File Locations (precedence order):")
    print("  1. .local.codeweaver.toml (workspace local)")
    print("  2. .codeweaver.toml (repository)")
    print("  3. ~/.config/codeweaver/config.toml (user)")
    print()
    print("🌍 Key Environment Variables:")
    env_vars = [
        ("CW_VECTOR_BACKEND_PROVIDER", "Backend provider (qdrant, pinecone, etc.)"),
        ("CW_VECTOR_BACKEND_URL", "Backend URL"),
        ("CW_VECTOR_BACKEND_API_KEY", "Backend API key"),
        ("embedding_provider", "Embedding provider (voyage, openai, etc.)"),
        ("CW_EMBEDDING_API_KEY", "Embedding API key"),
        ("CW_VOYAGE_API_KEY", "Voyage AI API key"),
        ("CW_OPENAI_API_KEY", "OpenAI API key"),
        ("CW_VECTOR_BACKEND_COLLECTION", "Vector database collection name"),
        ("CW_ENABLE_HYBRID_SEARCH", "Enable hybrid search (true/false)"),
        ("CW_USE_LOCAL_MODELS", "Use local models (true/false)"),
    ]

    for var, description in env_vars:
        print(f"  • {var:25} - {description}")


def main() -> NoReturn:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CodeWeaver Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a configuration file
  python -m codeweaver.config_cli validate config.toml

  # Generate production configuration example
  python -m codeweaver.config_cli generate production -o production.toml

  # Check backend/provider compatibility
  python -m codeweaver.config_cli check-compatibility qdrant voyage --enable-hybrid

  # Show configuration system information
  python -m codeweaver.config_cli info
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a configuration file")
    validate_parser.add_argument("config_file", help="Path to configuration file")
    validate_parser.add_argument(
        "--show-effective-config",
        action="store_true",
        help="Show the effective configuration after merging",
    )

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate example configurations")
    generate_parser.add_argument(
        "scenario",
        choices=["all", "development", "production", "testing", "example"],
        help="Configuration scenario to generate",
    )
    generate_parser.add_argument("-o", "--output", help="Output file path (for single scenario)")
    generate_parser.add_argument("--output-dir", help="Output directory (for all scenarios)")

    # Check compatibility command
    compat_parser = subparsers.add_parser(
        "check-compatibility", help="Check backend and provider compatibility"
    )
    compat_parser.add_argument("backend", help="Backend provider name")
    compat_parser.add_argument("provider", help="Embedding provider name")
    compat_parser.add_argument(
        "--enable-hybrid", action="store_true", help="Check hybrid search compatibility"
    )

    # Info command
    subparsers.add_parser("info", help="Show configuration system information")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging(args.verbose)

    # Route to appropriate command handler
    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "check-compatibility":
        cmd_check_compatibility(args)
    elif args.command == "info":
        cmd_info(args)

    sys.exit(0)


if __name__ == "__main__":
    main()
