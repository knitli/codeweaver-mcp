# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Error handling and recovery system for intent processing."""

from codeweaver.intent.recovery.fallback_handler import FallbackChain, IntentErrorHandler


__all__ = ("FallbackChain", "IntentErrorHandler")
