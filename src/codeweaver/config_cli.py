#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Command-line interface for CodeWeaver configuration management.

Provides tools for validating, migrating, and generating configuration files.
"""

import argparse
import logging
import sys

from pathlib import Path
from typing import NoReturn

from codeweaver.config_migration import (
    ConfigMigrationError,
    ConfigValidator,
    check_backend_compatibility,
    generate_deployment_configs,
    validate_configuration_file,
)


def setup_logging(*, verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a configuration file."""
    try:
        _validate_cmd(args)
    except ConfigMigrationError as e:
        print(f"❌ Validation failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def _validate_cmd(args: argparse.Namespace) -> None:
    """Internal command to validate a configuration file."""
    config, warnings = validate_configuration_file(args.config_file)

    print(f"✅ Configuration file: {args.config_file}")
    print(f"📋 Format: {'Legacy' if config.is_legacy_config() else 'New'}")
    print(f"🏗️  Backend: {config.get_effective_backend_provider()}")
    print(f"🤖 Provider: {config.get_effective_CW_EMBEDDING_PROVIDER()}")

    if warnings:
        print(f"\n⚠️  Validation Warnings ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print("\n✅ No validation warnings found")

    if args.show_effective_config:
        print("\n📄 Effective Configuration:")
        effective_config = config.to_new_format_dict()
        import json

        print(json.dumps(effective_config, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate example configurations."""
    configs = generate_deployment_configs()

    if args.scenario == "all":
        print("📚 Available deployment scenarios:")
        for name in configs:
            print(f"  • {name}")

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for name, content in configs.items():
                output_file = output_dir / f"{name}.toml"
                with output_file.open("w") as f:
                    f.write(content)
                print(f"📁 Generated: {output_file}")

    elif args.scenario in configs:
        content = configs[args.scenario]

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
        print(f"Available scenarios: {', '.join(configs.keys())}")
        sys.exit(1)


def cmd_check_compatibility(args: argparse.Namespace) -> None:
    """Check backend and provider compatibility."""
    compatible = check_backend_compatibility(args.backend, args.provider)
    warnings = ConfigValidator.validate_backend_provider_combination(args.backend, args.provider)

    print("🔍 Compatibility Check:")
    print(f"  Backend: {args.backend}")
    print(f"  Provider: {args.provider}")

    if compatible:
        print("✅ Compatible combination")
    else:
        print("⚠️  Potential compatibility issues:")
        for warning in warnings:
            print(f"  • {warning}")

    # Additional checks
    if args.enable_hybrid:
        if hybrid_warnings := ConfigValidator.validate_hybrid_search_config(
            args.backend, True, True
        ):
            print("\n🔀 Hybrid Search Warnings:")
            for warning in hybrid_warnings:
                print(f"  • {warning}")
        else:
            print("\n✅ Hybrid search is supported")


def cmd_info(args: argparse.Namespace) -> None:
    """Show configuration system information."""
    _print_args(
        "📊 CodeWeaver Configuration System",
        "==================================",
        "🏗️  Supported Backends:",
    )
    validator = ConfigValidator()
    for backend in validator.backend_embedding_compatibility:
        hybrid_support = "✅" if backend in validator.hybrid_search_backends else "❌"
        print(f"  • {backend:15} (Hybrid: {hybrid_support})")
    print()

    # Supported providers
    print("🤖 Supported Embedding Providers:")
    all_providers = set()
    for providers in validator.backend_embedding_compatibility.values():
        all_providers.update(providers)

    for provider in sorted(all_providers):
        local_support = "✅" if provider in ConfigValidator.LOCAL_PROVIDERS else "❌"
        rerank_support = "✅" if provider in ConfigValidator.RERANKING_PROVIDERS else "❌"
        print(f"  • {provider:20} (Local: {local_support}, Rerank: {rerank_support})")
    print()

    # Configuration locations
    print("📁 Configuration File Locations (precedence order):")
    print("  1. .local.code-weaver.toml (workspace local)")
    _print_args(
        "  2. .code-weaver.toml (repository)",
        "  3. ~/.config/code-weaver/config.toml (user)",
        "🌍 Key Environment Variables:",
    )
    env_vars = [
        ("CW_VECTOR_BACKEND_PROVIDER", "Backend provider (qdrant, pinecone, etc.)"),
        ("CW_VECTOR_BACKEND_URL", "Backend URL"),
        ("CW_VECTOR_BACKEND_API_KEY", "Backend API key"),
        ("CW_EMBEDDING_PROVIDER", "Embedding provider (voyage, openai, etc.)"),
        ("CW_EMBEDDING_API_KEY", "Embedding API key"),
        ("CW_VOYAGE_API_KEY", "Voyage AI API key"),
        ("CW_OPENAI_API_KEY", "OpenAI API key"),
        ("CW_VECTOR_BACKEND_COLLECTION", "Vector database collection name"),
        ("CW_ENABLE_HYBRID_SEARCH", "Enable hybrid search (true/false)"),
        ("CW_USE_LOCAL_MODELS", "Use local models (true/false)"),
    ]

    for var, description in env_vars:
        print(f"  • {var:25} - {description}")


# TODO Rename this here and in `cmd_info`
def _print_args(arg0, arg1, arg2):
    print(arg0)
    print(arg1)
    print()

    # Supported backends
    print(arg2)


def main() -> NoReturn:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CodeWeaver Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a configuration file
  python -m codeweaver.config_cli validate config.toml

  # Migrate legacy configuration to new format
  python -m codeweaver.config_cli migrate legacy-config.toml -o new-config.toml

  # Generate production configuration example
  python -m codeweaver.config_cli generate production_cloud -o production.toml

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

    # Migrate command
    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate configuration from legacy to new format"
    )
    migrate_parser.add_argument("config_file", help="Path to legacy configuration file")
    migrate_parser.add_argument("-o", "--output", help="Output path for migrated configuration")
    migrate_parser.add_argument(
        "--show-diff", action="store_true", help="Show the migrated configuration content"
    )

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate example configurations")
    generate_parser.add_argument(
        "scenario",
        choices=[
            "all",
            "local_development",
            "production_cloud",
            "enterprise_multi_source",
            "pinecone_setup",
            "weaviate_hybrid",
        ],
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
