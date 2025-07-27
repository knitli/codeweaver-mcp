# sourcery skip: do-not-use-staticmethod
# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Configuration migration utilities for backward compatibility."""

import logging

from typing import Any


logger = logging.getLogger(__name__)


class ConfigMigration:
    """Handle migration of existing configurations to include services."""

    @staticmethod
    def migrate_server_config_to_services(config: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
        # sourcery skip: no-long-functions
        """Migrate existing server-level middleware config to services config."""
        migrated = config.copy()

        # Create services section if it doesn't exist
        if "services" not in migrated:
            migrated["services"] = {}

        services = migrated["services"]

        # Migrate existing server logging settings to services.logging
        if "server" in migrated:
            server_config = migrated["server"]

            if "logging" not in services:
                services["logging"] = {"enabled": True, "provider": "fastmcp_logging"}

            # Map server log_level to services.logging.log_level
            if "log_level" in server_config:
                services["logging"]["log_level"] = server_config["log_level"]
                logger.info("Migrated server.log_level to services.logging.log_level")

            # Map enable_request_logging to include_payloads
            if "enable_request_logging" in server_config:
                services["logging"]["include_payloads"] = server_config["enable_request_logging"]
                logger.info("Migrated server.enable_request_logging to services.logging.include_payloads")

        # Migrate chunking config to services.chunking
        if "chunking" in migrated:
            chunking_config = migrated["chunking"]

            if "chunking" not in services:
                services["chunking"] = {"enabled": True, "provider": "fastmcp_chunking"}

            # Copy chunking settings
            for key, value in chunking_config.items():
                if key not in ["enabled", "provider"]:  # Don't override service-specific fields
                    services["chunking"][key] = value

            # Ensure provider is set
            if "provider" not in services["chunking"]:
                services["chunking"]["provider"] = "fastmcp_chunking"

            logger.info("Migrated chunking configuration to services.chunking")

        # Migrate indexing config to services.filtering
        if "indexing" in migrated:
            indexing_config = migrated["indexing"]

            if "filtering" not in services:
                services["filtering"] = {"enabled": True, "provider": "fastmcp_filtering"}

            # Map indexing settings to filtering
            mapping = {
                "use_gitignore": "use_gitignore",
                "additional_ignore_patterns": "ignore_directories",
            }

            for old_key, new_key in mapping.items():
                if old_key in indexing_config:
                    services["filtering"][new_key] = indexing_config[old_key]

            # Ensure provider is set
            if "provider" not in services["filtering"]:
                services["filtering"]["provider"] = "fastmcp_filtering"

            logger.info("Migrated indexing configuration to services.filtering")

        # Set default middleware services if not present
        middleware_defaults = {
            "timing": {"enabled": True, "provider": "fastmcp_timing"},
            "error_handling": {"enabled": True, "provider": "fastmcp_error_handling"},
            "rate_limiting": {"enabled": True, "provider": "fastmcp_rate_limiting"},
        }

        for service_name, default_config in middleware_defaults.items():
            if service_name not in services:
                services[service_name] = default_config
                logger.info("Added default %s service configuration", service_name)

        # Ensure global service settings are present
        if "middleware_auto_registration" not in services:
            services["middleware_auto_registration"] = True

        if "middleware_initialization_order" not in services:
            services["middleware_initialization_order"] = [
                "error_handling", "rate_limiting", "logging", "timing"
            ]

        return migrated

    @staticmethod
    def validate_migrated_config(config: dict[str, Any]) -> list[str]:
        """Validate migrated configuration and return warnings."""
        warnings = []

        if "services" not in config:
            warnings.append("No services configuration found after migration")
            return warnings

        services = config["services"]

        # Check core services
        core_services = ["chunking", "filtering"]
        for service in core_services:
            if service not in services:
                warnings.append(f"Core service '{service}' not configured")
            elif not services[service].get("enabled", True):
                warnings.append(f"Core service '{service}' is disabled")

        # Check middleware services
        middleware_services = ["logging", "timing", "error_handling", "rate_limiting"]
        warnings.extend(
            f"Middleware service '{service}' not configured"
            for service in middleware_services
            if service not in services
        )
        # Validate middleware initialization order
        if "middleware_initialization_order" in services:
            order = services["middleware_initialization_order"]
            if not isinstance(order, list):
                warnings.append("middleware_initialization_order should be a list")
            else:
                valid_services = set(middleware_services)
                warnings.extend(
                    f"Invalid service in initialization order: {service_name}"
                    for service_name in order
                    if service_name not in valid_services
                )
        return warnings

    @staticmethod
    def apply_compatibility_fixes(config: dict[str, Any]) -> dict[str, Any]:
        """Apply compatibility fixes for common configuration issues."""
        fixed = config.copy()

        # Fix rate limiting configuration if using old format
        if "services" in fixed and "rate_limiting" in fixed["services"]:
            rate_config = fixed["services"]["rate_limiting"]

            # Convert old requests_per_minute to requests_per_second
            if "requests_per_minute" in rate_config:
                requests_per_minute = rate_config.pop("requests_per_minute")
                rate_config["max_requests_per_second"] = requests_per_minute / 60.0
                logger.info("Converted requests_per_minute to max_requests_per_second")

        # Fix logging configuration
        if "services" in fixed and "logging" in fixed["services"]:
            log_config = fixed["services"]["logging"]

            # Ensure log_level is uppercase
            if "log_level" in log_config:
                log_config["log_level"] = log_config["log_level"].upper()

        # Fix file size configurations (convert to bytes if needed)
        if "services" in fixed and "filtering" in fixed["services"]:
            filter_config = fixed["services"]["filtering"]

            if "max_file_size" in filter_config:
                size_value = filter_config["max_file_size"]
                if isinstance(size_value, str) and size_value.endswith("MB"):
                    # Convert "1MB" to bytes
                    mb_value = float(size_value[:-2])
                    filter_config["max_file_size"] = int(mb_value * 1024 * 1024)
                    logger.info("Converted max_file_size from MB to bytes")

        return fixed


def migrate_config(config_data: dict[str, Any]) -> dict[str, Any]:
    """Main function to migrate configuration with all fixes applied."""
    logger.info("Starting configuration migration")

    # Apply migration
    migrated = ConfigMigration.migrate_server_config_to_services(config_data)

    # Apply compatibility fixes
    fixed = ConfigMigration.apply_compatibility_fixes(migrated)

    # Validate and warn about issues
    warnings = ConfigMigration.validate_migrated_config(fixed)
    for warning in warnings:
        logger.warning("Configuration migration warning: %s", warning)

    logger.info("Configuration migration completed")
    return fixed
