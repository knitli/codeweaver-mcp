# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Service layer for CodeWeaver - connecting middleware with factory patterns."""

from codeweaver.services.manager import ServicesManager
from codeweaver.services.providers.base_provider import BaseServiceProvider


__all__ = ["BaseServiceProvider", "ServicesManager"]
