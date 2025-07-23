# SPDX-FileCopyrightText: 2025 Knitli Inc.
# SPDX-FileContributor: Adam Poulemanos <adam@knit.li>
#
# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Structural search functionality using ast-grep.

Provides direct ast-grep pattern matching capabilities for precise
code structure queries across multiple programming languages.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ast-grep for structural search
try:
    from ast_grep_py import SgRoot
    AST_GREP_AVAILABLE = True
except ImportError:
    AST_GREP_AVAILABLE = False
    logging.warning("ast-grep-py not available for structural search")


class AstGrepStructuralSearch:
    """Provides direct ast-grep structural search capabilities."""
    
    def __init__(self):
        self.available = AST_GREP_AVAILABLE
        if not self.available:
            logger.warning("ast-grep not available for structural search")
    
    async def structural_search(self, pattern: str, language: str, 
                               root_path: str) -> List[Dict[str, Any]]:
        """Perform structural search using ast-grep patterns."""
        if not self.available:
            raise ValueError("ast-grep not available, install with: pip install ast-grep-py")
        
        results = []
        root = Path(root_path)
        
        # Find files for the language
        extensions = self._get_extensions_for_language(language)
        files = []
        
        for ext in extensions:
            files.extend(root.rglob(f"*{ext}"))
        
        # Search each file
        for file_path in files[:100]:  # Limit to avoid overload
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse and search with ast-grep
                sg_root = SgRoot(content, language)
                tree = sg_root.root()
                
                matches = tree.find(pattern=pattern)
                if matches:
                    range_info = matches.range()
                    results.append({
                        "file_path": str(file_path),
                        "match_content": matches.text(),
                        "start_line": range_info.start.line + 1,
                        "end_line": range_info.end.line + 1,
                        "start_column": range_info.start.column + 1,
                        "end_column": range_info.end.column + 1,
                    })
                
            except Exception as e:
                logger.warning(f"Error searching {file_path}: {e}")
                continue
        
        return results
    
    def _get_extensions_for_language(self, language: str) -> List[str]:
        """Get file extensions for a language."""
        lang_map = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'rust': ['.rs'],
            'go': ['.go'],
            'java': ['.java'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.hpp', '.cc', '.cxx'],
            'html': ['.html', '.htm'],
            'css': ['.css', '.scss', '.sass'],
        }
        return lang_map.get(language, [f'.{language}'])