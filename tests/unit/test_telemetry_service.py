# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Unit tests for telemetry service."""

import os

from unittest.mock import patch

import pytest

from pydantic import ValidationError

from codeweaver.services.providers.telemetry import PostHogTelemetryProvider
from codeweaver.types import TelemetryServiceConfig


@pytest.mark.unit
@pytest.mark.telemetry
@pytest.mark.mock_only
class TestPostHogTelemetryProvider:
    """Unit tests for PostHog telemetry provider."""

    @pytest.fixture
    def default_config(self) -> TelemetryServiceConfig:
        """Create a default telemetry configuration."""
        return TelemetryServiceConfig(enabled=True, mock_mode=True, anonymous_tracking=True)

    @pytest.fixture
    def telemetry_provider(
        self, default_config: TelemetryServiceConfig
    ) -> PostHogTelemetryProvider:
        """Create a telemetry provider instance."""
        return PostHogTelemetryProvider(default_config)

    @pytest.mark.env_vars
    def test_opt_out_environment_variable(self) -> None:
        """Test opt-out via CW_TELEMETRY_ENABLED environment variable."""
        config = TelemetryServiceConfig(enabled=True)

        with patch.dict(os.environ, {"CW_TELEMETRY_ENABLED": "false"}):
            provider = PostHogTelemetryProvider(config)
            assert not provider._enabled

    @pytest.mark.env_vars
    def test_opt_out_no_telemetry_variable(self) -> None:
        """Test opt-out via CW_NO_TELEMETRY environment variable."""
        config = TelemetryServiceConfig(enabled=True)

        with patch.dict(os.environ, {"CW_NO_TELEMETRY": "true"}):
            provider = PostHogTelemetryProvider(config)
            assert not provider._enabled

    @pytest.mark.config
    def test_config_based_opt_out(self) -> None:
        """Test opt-out via configuration."""
        config = TelemetryServiceConfig(enabled=False)
        provider = PostHogTelemetryProvider(config)
        assert not provider._enabled

    def test_runtime_opt_out(self, telemetry_provider: PostHogTelemetryProvider) -> None:
        """Test runtime opt-out functionality."""
        assert telemetry_provider._enabled

        telemetry_provider.set_enabled(enabled=False)
        assert not telemetry_provider._enabled

    def test_anonymous_mode_toggle(self, telemetry_provider: PostHogTelemetryProvider) -> None:
        """Test anonymous mode toggle functionality."""
        original_user_id = telemetry_provider._user_id

        telemetry_provider.set_anonymous(anonymous=False)
        assert not telemetry_provider._anonymous
        assert telemetry_provider._user_id != original_user_id

    def test_path_sanitization(self, telemetry_provider: PostHogTelemetryProvider) -> None:
        """Test file path sanitization."""
        sensitive_path = "/home/user/secret-project/important-file.py"
        sanitized = telemetry_provider._sanitize_path(sensitive_path)

        assert sensitive_path != sanitized
        assert "secret-project" not in sanitized
        assert "important-file" not in sanitized
        assert sanitized.startswith("path_")
        assert ".py_" in sanitized  # Extension is included in the format

    def test_repository_name_sanitization(
        self, telemetry_provider: PostHogTelemetryProvider
    ) -> None:
        """Test repository name sanitization."""
        sensitive_repo = "company/proprietary-software"
        sanitized = telemetry_provider._sanitize_repository_name(sensitive_repo)

        assert sensitive_repo != sanitized
        assert "company" not in sanitized
        assert "proprietary-software" not in sanitized
        assert sanitized.startswith("repo_")

    def test_query_sanitization(self, telemetry_provider: PostHogTelemetryProvider) -> None:
        """Test query sanitization."""
        sensitive_query = "find password in authentication.py"
        sanitized = telemetry_provider._sanitize_query(sensitive_query)

        assert "raw_query" not in sanitized  # Should be sanitized
        assert "query_length" in sanitized
        assert "word_count" in sanitized
        assert sanitized["query_length"] == len(sensitive_query)

    def test_privacy_info(self, telemetry_provider: PostHogTelemetryProvider) -> None:
        """Test privacy information reporting."""
        privacy_info = telemetry_provider.get_privacy_info()

        assert "telemetry_enabled" in privacy_info
        assert "anonymous_tracking" in privacy_info
        assert "data_collection" in privacy_info
        assert "privacy_measures" in privacy_info
        assert "opt_out_methods" in privacy_info

        # Verify opt-out methods are documented
        opt_out_methods = privacy_info["opt_out_methods"]
        assert any("CW_TELEMETRY_ENABLED=false" in method for method in opt_out_methods)
        assert any("CW_NO_TELEMETRY=true" in method for method in opt_out_methods)

    @pytest.mark.asyncio
    async def test_event_tracking_disabled(self) -> None:
        """Test that events are not tracked when telemetry is disabled."""
        config = TelemetryServiceConfig(enabled=False)
        provider = PostHogTelemetryProvider(config)

        await provider.track_event("test_event", {"key": "value"})

        stats = await provider.get_telemetry_stats()
        assert stats["events_tracked"] == 0

    @pytest.mark.asyncio
    async def test_health_check(self, telemetry_provider: PostHogTelemetryProvider) -> None:
        """Test health check functionality."""
        await telemetry_provider.initialize()

        is_healthy = await telemetry_provider._check_health()
        assert is_healthy  # Should be healthy when enabled and initialized

    @pytest.mark.asyncio
    async def test_batch_flushing(self, telemetry_provider: PostHogTelemetryProvider) -> None:
        """Test event batching and flushing."""
        await telemetry_provider.initialize()

        # Track multiple events
        for i in range(5):
            await telemetry_provider.track_event(f"test_event_{i}", {"index": i})

        stats_before_flush = await telemetry_provider.get_telemetry_stats()
        assert stats_before_flush["events_tracked"] == 5
        assert stats_before_flush["queue_size"] > 0

        # Flush events
        await telemetry_provider.flush()

        stats_after_flush = await telemetry_provider.get_telemetry_stats()
        assert stats_after_flush["events_sent"] == 5
        assert stats_after_flush["queue_size"] == 0

    def test_config_validation(self) -> None:
        """Test configuration validation."""
        # Test valid configuration
        valid_config = TelemetryServiceConfig(
            enabled=True, batch_size=50, flush_interval=30.0, max_queue_size=1000
        )
        provider = PostHogTelemetryProvider(valid_config)
        assert provider._config.batch_size == 50

        # Test invalid batch size (should be constrained by Pydantic)
        with pytest.raises(ValidationError):
            TelemetryServiceConfig(batch_size=0)  # Below minimum

        with pytest.raises(ValidationError):
            TelemetryServiceConfig(batch_size=2000)  # Above maximum
