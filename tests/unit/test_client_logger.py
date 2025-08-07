#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

# sourcery skip: avoid-global-variables
"""Test script to verify the CodeWeaver logging fixes."""

import logging
import sys

from pathlib import Path

import pytest


# Add the src directory to the path so we can import codeweaver modules
sys.path.insert(0, Path(__file__).parent.parent.parent / 'src')

from codeweaver.cli.utils.client_logger import CodeWeaverLogMessage, get_handler, get_logger


@pytest.mark.async_test
@pytest.mark.unit
class TestLogger:
    """Custom logger for testing purposes."""

    @pytest.mark.asyncio
    async def test_client_logger() -> bool:
        """Test the logger functionality."""
        print("Testing CodeWeaver logging...")

        # Test 1: Create a CodeWeaverLogMessage
        try:
            message = CodeWeaverLogMessage(level=logging.INFO, logger="test", data="Test message")
            print(f"✓ CodeWeaverLogMessage created successfully: level={message.level}, logger={message.logger}, data={message.data}")
        except Exception as e:
            print(f"✗ Error creating CodeWeaverLogMessage: {e}")
            return False

        # Test 2: Test with string level
        try:
            message = CodeWeaverLogMessage(level="DEBUG", logger="test", data="Test with string level")
            print(f"✓ CodeWeaverLogMessage with string level created: level={message.level}")
        except Exception as e:
            print(f"✗ Error creating CodeWeaverLogMessage with string level: {e}")
            return False

        # Test 3: Test with None values
        try:
            message = CodeWeaverLogMessage()
            print(f"✓ CodeWeaverLogMessage with defaults created: level={message.level}, logger={message.logger}, data={message.data}")
        except Exception as e:
            print(f"✗ Error creating CodeWeaverLogMessage with defaults: {e}")
            return False

        # Test 4: Test get_logger
        try:
            logger = await get_logger()
            print(f"✓ Logger created successfully: {logger.name}")

            # Test logging
            logger.info("Test log message")
            print("✓ Log message sent successfully")
        except Exception as e:
            print(f"✗ Error with get_logger: {e}")
            return False

        # Test 5: Test get_handler
        try:
            handler = get_handler()
            print(f"✓ Handler retrieved successfully: {type(handler)}")
        except Exception as e:
            print(f"✗ Error with get_handler: {e}")
            return False

        print("\nAll tests passed! The logging system is working correctly.")
        return True
