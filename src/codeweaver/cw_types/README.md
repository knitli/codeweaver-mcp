<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# CodeWeaver's Type System

## Using the Type System - Maintainers

To keep circular dependencies in check, the type system is split into multiple packages and modules. Packages are defined by the part of the framework they belong to (e.g. `sources`, `backends`, `services`, etc.). Modules in the root-level types package define building-block type-types that are used across the framework.

### Importing Types

- Within the types package, use full paths to import types from other modules and packages in the types package, like:
    ```python
    from codeweaver.cw_types.sources.enums import ContentType
    ```
- **Within the rest of the framework**, **import only from `types`**. For example:
    ```python
    from codeweaver.cw_types import ContentType
    ```
