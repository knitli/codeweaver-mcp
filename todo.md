<!--
SPDX-FileCopyrightText: 2025 Knitli Inc.
SPDX-FileContributor: Adam Poulemanos <adam@knit.li>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# ToDo

## Integrate sampling

- FastMCP has a sampling feature that allows for out-of-band dialogue with the LLM to refine the results.
- The beautiful thing is that you can use the developer's LLM to have out-of-context conversations *about how to support the user's LLM*. (i.e. shape the context for the developer's session without polluting the context for the developer's session)
- This could be the backbone of the intent layer, where we can use the LLM to shape context ... for itself.
- I'm pretty sure there must be an appropriate movie reference for this... it feels a bit... Groundhog Day-ish... Inception-ish...

- Easy to implement. See https://gofastmcp.com/servers/sampling -- an import and our own tailored system prompt should be enough to get started.

## Prompting

- Setup proper prompting where needed for fastmcp

## Expand Baseline Functionality

- Add more NLP patterns for different coding languages
- Add semantic search patterns for all ast-grep supported languages
- Add support other languages (not English)
- Index and search code comments
- Add redis, milvus, weaviate, epsilla, elasticsearch, HnSw, InMemoryExactNNIndex for DocArray

## Performance and Scalability

- Add performance profiling and benchmarking
- Publish docker version


## Investigate Smells 🦨

- `cw_types/backends/capabilities.py` and `cw_types/backends/providers`: `BackendCapabilities` and `SERVICES_REGISTRY` are seemingly different versions of the same thing... unclear if one, both, or either are used anywhere
  - pyright also warns about neither defining all of the attributes of the other

## Auth Integration

- Add the `eunomia` and `permit.io` auth middleware from fastmcp (already penciled into `pyproject.toml`)
  - **WorkOS/AuthKit is already ready** out of the box, but we should: 1) Add options to the config and passthrough env vars (it can be enabled right now with the fastmcp env vars), 2) Document how to use it, 3) Add a test for it
  - Eunomia and permit.io could be added very quickly, at least for just authenticating the mcp server
  - ideally we'd integrate the auth middleware throughout to allow for more fine-grained access control, but that can be done later

## Fix ridiculous number of typing errors in the cli package

- One of our LLM friends got a bit overzealous in implementing the cli package, with no concern for typing
- I fixed the worst of it, but there are still *a lot* (like 400+) typing errors in the cli package, but I suspect it's kind of a chain-reaction deal where if we added some stronger top-level typing, it would fix a lot of the errors

## Root out literal strings

- There are a lot of literal strings in the codebase, which I'd understand given how new it is, except that we actually defined enums for the exact things that are being used as literal strings
- We should root out these literal strings and replace them with the appropriate enum values

## Remove aspirational code

- There's a fair amount of penciled in code for providers that don't exist yet and aren't high on the priority list

## Longer term

- We can probably consolidate types in a lot of places and try to bring the protocols, ABCs, and dataclasses, enums, and base models more in line with each other
  - Kind of a natural consequence of building out an enterprise scale codebase in 10 days... but we should try to clean it up a bit when we can breathe.



notes:
- pydantic-graph's persistence modules are a good example for the thread project -- we'd do it in rust, but the persistence modules are a good example of how to do it in python
