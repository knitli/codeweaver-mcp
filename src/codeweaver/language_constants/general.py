"""General language constants for CodeWeaver."""

from types import MappingProxyType

from codeweaver.cw_types.language import SemanticSearchLanguage


DEFAULT_AST_GREP_PATTERNS = (("comment", "comment"),)

DEFAULT_NER_PATTERNS = (
    MappingProxyType({
        "label": "LANGUAGE",
        "pattern": [{"LOWER": {"IN": SemanticSearchLanguage.get_values()}}],
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
                        # only include *very* common frameworks
                        # more can be in the language-specific NER patterns
                        "react",
                        "vue",
                        "angular",
                        "django",
                        "flask",
                        "express",
                        "spring",
                        "fastapi",
                        "nextjs",
                        "nuxt",
                        "laravel",
                        "rails",
                        "dotnet",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "DATABASE",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        # keep ORMs and adapters out of this list
                        # those should be in the language-specific NER patterns
                        "bigquery",
                        "cassandra",
                        "cockroachdb",
                        "couchdb",
                        "dynamodb",
                        "elasticsearch",
                        "epsilla",
                        "faiss",
                        "firestore",
                        "hnsw",
                        "mariadb",
                        "milvus",
                        "mongodb",
                        "mongodb",
                        "mysql",
                        "neo4j",
                        "neptune",
                        "opensearch",
                        "oracle",
                        "pinecone",
                        "postgresql",
                        "qdrant",
                        "redis",
                        "redisearch",
                        "sqlite",
                        "vearch",
                        "vector",
                        "vector_db",
                        "vectordb",
                        "vectorsearch",
                        "weaviate",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "OPERATION",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        # general operations that can apply to any language
                        # We shouldn't need to define any language-specific operations...
                        "authentication",
                        "auth",
                        "authorization",
                        "backup",
                        "caching",
                        "building",
                        "caching",
                        "compiling",
                        "debugging",
                        "debugging",
                        "deployment",
                        "formatting",
                        "linting",
                        "logging",
                        "migration",
                        "monitoring",
                        "optimization",
                        "packaging",
                        "profiling",
                        "refactoring",
                        "testing",
                        "validation",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "FILE_TYPE",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "json",
                        "yaml",
                        "xml",
                        "csv",
                        "sql",
                        "dockerfile",
                        "makefile",
                        "config",
                        "env",
                        "properties",
                        "manifest",
                        "schema",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "TOOL",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        # only include general and very common tools here
                        # more can be in the language-specific NER patterns
                        "ansible",
                        "atom",
                        "babel",
                        "cargo",
                        "circleci",
                        "composer",
                        "docker",
                        "eclipse",
                        "emacs",
                        "eslint",
                        "git",
                        "github",
                        "githubactionsgitlab",
                        "gradle",
                        "intellij",
                        "jenkins",
                        "kubernetes",
                        "maven",
                        "mise",
                        "moon",
                        "netlify",
                        "netbeans",
                        "npm",
                        "nuget",
                        "nvim",
                        "sentry",
                        "sublime",
                        "terraform",
                        "travisci",
                        "vim",
                        "visualstudio",
                        "vscode",
                        "webpack",
                    ]
                }
            }
        ],
    }),
)


__all__ = ("DEFAULT_NER_PATTERNS",)
