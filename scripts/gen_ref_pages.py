#!/usr/bin/env python
# sourcery skip: avoid-global-variables

# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files


root = Path(__file__).parent.parent
src = root / "src" / "codeweaver"  # (1)!

for path in sorted(src.rglob("*.py")):  # (2)!
    module_path = path.relative_to(src).with_suffix("")  # (3)!
    doc_path = path.relative_to(src).with_suffix(".md")  # (4)!
    full_doc_path = Path("api_reference", doc_path)  # (5)!

    parts = tuple(module_path.parts)

    if "__init__" in parts or "__main__" in parts:  # (6)!
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  # (7)!
        identifier = ".".join(parts)  # (8)!
        print(f"::: {identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))  # (10)!
