"""Language constants for JavaScript and its family of languages."""

from types import MappingProxyType


DEFAULT_JAVASCRIPT_AST_GREP_PATTERNS = (
    ("comment", "comment"),
    ("function_declaration", "function"),
    ("arguments", "argument"),
    ("formal_parameters", "parameter"),
    ("class_declaration", "class"),
    ("arrow_function", "arrow_function"),
    ("method_definition", "method"),
    ("interface_declaration", "interface"),
    ("type_alias", "type_alias"),
    ("type_annotation", "type_annotation"),
    ("type_parameter", "type_parameter"),
    ("return_type", "return_type"),
    ("variable_declaration", "variable"),
    ("import_statement", "import"),
    ("export_statement", "export"),
)

# TODO: JSX and TSX

DEFAULT_JAVASCRIPT_NER_PATTERNS = (
    MappingProxyType({
        "label": "LANGUAGE",
        "pattern": [{"LOWER": {"IN": ["javascript", "js", "typescript", "ts", "jsx", "tsx"]}}],
    }),
    MappingProxyType({
        "label": "CODE_ELEMENT",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "argument",
                        "attribute",
                        "block",
                        "class",
                        "constant",
                        "enum",
                        "function",
                        "interface",
                        "method",
                        "module",
                        "namespace",
                        "package",
                        "parameter",
                        "property",
                        "type",
                        "variable",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "FRAMEWORK",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "angular",
                        "astro",
                        "backbone",
                        "d3",
                        "ember",
                        "express",
                        "jquery",
                        "koa",
                        "lodash",
                        "nestjs",
                        "nextjs",
                        "nuxt",
                        "nuxtjs",
                        "qwik",
                        "react",
                        "remix",
                        "rxjs",
                        "solidjs",
                        "svelte",
                        "underscore",
                        "vue",
                        "zod",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "DATABASE",
        "pattern": [{"LOWER": {"IN": ["mongodb", "mongoose", "sequelize", "typeorm", "prisma"]}}],
    }),
    MappingProxyType({
        "label": "TEST_FRAMEWORK",
        "pattern": [{"LOWER": {"IN": ["ava", "chai", "cypress", "jest", "mocha", "vitest"]}}],
    }),
    MappingProxyType({
        "label": "TOOL",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "babel",
                        "biome",
                        "bun",
                        "bunx",
                        "esbuild",
                        "eslint",
                        "deno",
                        "npm",
                        "npx",
                        "nx",
                        "parcel",
                        "pnpm",
                        "prettier",
                        "rollup",
                        "tsc",
                        "vite",
                        "webpack",
                        "yarn",
                    ]
                }
            }
        ],
    }),
)
