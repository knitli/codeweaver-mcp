# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Exceptions for intent processing in CodeWeaver."""
# Intent processing errors extending existing hierarchy
from codeweaver.types.exceptions import CodeWeaverError


class IntentError(CodeWeaverError):
    """Base class for intent layer errors."""


class IntentParsingError(IntentError):
    """Error in parsing user intent."""


class StrategyExecutionError(IntentError):
    """Error in strategy execution."""


class StrategySelectionError(IntentError):
    """Error in strategy selection."""


class ServiceIntegrationError(IntentError):
    """Error in service layer integration."""
