"""General language constants for CodeWeaver."""

from types import MappingProxyType


DEFAULT_NER_PATTERNS = (
    MappingProxyType({
        "label": "LANGUAGE",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "python",
                        "javascript",
                        "typescript",
                        "java",
                        "go",
                        "rust",
                        "c++",
                        "csharp",
                        "ruby",
                        "php",
                        "swift",
                        "kotlin",
                        "scala",
                    ]
                }
            }
        ],
    }),
    MappingProxyType({
        "label": "CODE_ELEMENT",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "function",
                        "class",
                        "method",
                        "variable",
                        "constant",
                        "interface",
                        "module",
                        "package",
                        "component",
                        "service",
                        "endpoint",
                        "route",
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
                        "cassandra",
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
                        "githubactions"
                        "gitlab",
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
                        "pip",
                        "pycharm",
                        "pytest",
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


__all__ = (
    "DEFAULT_NER_PATTERNS",
)
