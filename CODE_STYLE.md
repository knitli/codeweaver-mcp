# Code Style and Design Principles

## Design Principles and Goals

1. CodeWeaver should be able to deliver sophisticated, tailored, and exact codebase context in response to AI agent requests, which they should be able to make in plain language.
2. The process and results should be transparent to developers who wish to see and understand it.
3. CodeWeaver should be extensible and modular while remaining simple and intuitive. Its modules and their functions and use should be obvious.
4. Use existing powerful and extensible generic types as much as possible. Import, copy, or imitate popular platforms that already exist. For CodeWeaver, that primarily means the pydantic ecosystem -- `pydantic`, `pydantic-settings`, `pydantic-ai`, `pydantic-eval`, `pydantic-graph`, `FastMCP`. We don't need to reinvent something if it will work for us out of the box.
5. These libraries provide powerful but simple and intuitive abstractions that we can often use without significant modification. When that won't work, we follow their example -- simple, powerful, generic but well-validated, annotated, and typed interfaces.