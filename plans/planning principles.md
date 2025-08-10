# Overall Product Plan

## High Level Concept

CodeWeaver (the package and github repo is codeweaver-mcp) is a next-generation [model context protocol](https://modelcontextprotocol.io) server (MCP) and extensible platform for fusing and discovering *codebase intelligence*. CodeWeaver is an open-source developer-focused component of Knitli's broader "exquisite context" strategy to deliver precise and complete context to AI agents to significantly reduce token waste, dramatically reduce time-to-desired-result, and improve agent outcomes -- use intelligent context delivery to make AI agents more efficient, effective, and much cheaper. 

### The MVP

CodeWeaver's MVP aims to deliver effective and targeted codebase context to developers' AI agents and deliver a platform that's easy for developers to expand, extend and grow.

#### Architecture

The functional vision: An intelligent codebase context layer that fuses data from anywhere to provide clear, actionable, and situation-specific context to the developer's agent. The goal is to consistently deliver the *exact* information the developers' agent needs to successfully execute its task in a way that completely abstracts the complexity of data retrieval and synthesis from the developer's agent. CodeWeaver exposes *a single tool* to the developer's agent: `**find_code**` with basic parameters (not yet fully designed, but possibly: `query` or `need` or `question` (str description of the desired information), `goal` or `intent` (str description of the developer's task), *maybe* something like an optional `focus` enum (i.e. familiarize, explain, implement, design, fix), `token_limit`: A limit on the tokens returned (a hard cap would be available in developer settings that would govern this -- an agent could say "50_000" but it won't get more than 10_000 if the developer set a cap at that level -- we're generally thinking 10_000 is a good default global cap)).

- Built on `FastMCP`, `pydantic`, `pydantic-settings`, `pydantic-ai`, `pydantic-evals` and `pydantic-graph`. CLI with `cyclopts`.
  - `FastMCP` provides the core server, generic services (logging, error-handling, timing, rate-limiting, optional auth)
    - Could also give a lot more if we choose that route (see dependency plan).
  - `pydantic-settings` provides powerful and simple configuration through json, toml, env
  - `pydantic-ai` brings out-of-the-box feature-flaggable support for every major AI model API through a common interface, along with advanced agent strategy and pipeline tools. We can use this support both internally with user provided BYOK, and also expose it for provider support. For example, we can expose cohere, openai, ollama, hugging-face and others are remote and local embedding generation and reranking model providers. It also lets us add support for `tavily` and `duckduckgo` as data source providers and exposes tool
    - rexports `pydantic-graph` and `pydantic-evals`
  - `pydantic-evals` provides a framework for automated strategy/pipeline evaluation
  - `pydantic-graph` provides a powerful scheme for designing and executing (graph-based) pipelines (which will form the core of our resolution pipeline)

**Other Capabilities for MVP**:

Search/Discovery:
  - Embedding generation and reranking with `voyage-ai` as a default embedding and reranking provider
  - `qdrant-client` for local or cloud embedding storage/retrieval, qdrant has robust hybrid search capabilities that we will leverage (payload based storage, robust query criteria, sparse indexes). Qdrant will be the default vector storage provider.
  - Possibly also implement `redis` primarily for caching requests/responses but can also be implemented as a vector store
  - Include `pydantic-ai`'s `tavily` (requires developer API key), fallback to `duckduckgo search` for no API key as default behavior
  - Ideally also incorporate the `context7` mcp tool into the data fusion capability

Supporting Services
  - `ast-grep-py` provides multilanguage semantic search and discovery powered by tree-sitter. We'll also have a naive parser as fallback on unsupported languages, and can expose custom pattern support 
    - core part of indexing strategy -- use semantic relationships and metadata (e.g. line numbers/module names as payloads, supported by documentation and config file payloads) 
  - `posthog` for opt-out privacy-friendly telemetry -- focus entirely on improving outcomes
  - `watchfiles` for monitoring filesystem changes; also supports adding hooks to execute when changes occur
  - `rignore` for file filtering/walking that respects gitignore and other ignorefiles (including custom ignores we can expose to user through config)
  - form a background indexing/watching service
  - **Multiple intelligence support services**
    - Using FastMCP `Context` to inject information to support resolving developer intent/goals
    - zero shot optimization/intelligence (in this case, that's zero-shot from the perspective of the developer's agent)
    - strategies for actionable results based on need


### The Flow

#### Main pipeline (a sort of multimodal rag on steroids)

Developer Task -> Developer's Agent -> `find_code` call -> Strategy-based intelligent intent/need resolution -> Optional `sampling` and/or `elicitation` request for clarification/refinement -> Fork data retrieval to available data sources (e.g. qdrant with relevant filters applied, tavily if relevant, context7 if relevant) -> rerank (combined provider and internal strategies) -> optional (but default) context agent review using `sampling` with optional fallback to developer supplied agent/api key or local agent

#### Secondary pipeline -- precontext generation

Essentially same as above except the trigger is a developer CLI or API call to generate precontext, and necessarily uses a developer-supplied API key or local agent instead of `sampling`. In this scenario there is no developer agent -- the developer takes the place of the developer agent. Sampling is only available in an MCP context (i.e. developer's agent request), so we can't piggyback on the client's agent and need our own.


> [!NOTE]
> See [the dependency plan](DEPENDENCY_PLAN.md) for more in depth discussion of all planned dependencies and how we'll use them.



## Architectural Strengths

### 1. Simplicity
- **Fewer Abstractions**: No factories, registries, service layers, middleware bridges
- **Clear Data Flow**: Graph structure shows exactly how data moves
- **Single Responsibility**: Each node does one thing well

### 2. Extensibility
- **Add Providers**: Implement interface, add to inference function
- **Add Nodes**: Create dataclass with run method
- **Add Flows**: Compose existing nodes in new ways
- **Add Intent Types**: Extend AnalyzeIntent logic

### 3. Performance
- **Natural Parallelization**: Independent operations run concurrently
- **Minimal Overhead**: Direct data flow without complex coordination
- **Efficient Resource Usage**: No unnecessary abstraction layers

### 4. Developer Experience
- **Self-Documenting**: Graph visualizations show system behavior
- **Easy Testing**: Each node is independently testable
- **Clear Debugging**: Graph execution shows exact failure points
- **Incremental Development**: Start simple, add complexity as needed

### 5. Maintainability
- **Explicit Dependencies**: Node return types define graph structure
- **Type Safety**: Pydantic models ensure data integrity
- **Version Control Friendly**: Changes are localized to specific nodes
- **Refactoring Safe**: Graph structure prevents breaking changes

## Dev Strategy

This architecture can be implemented incrementally:

1. **Phase 1**: Basic graph structure with simple intent analysis
2. **Phase 2**: Port existing functionality node by node
3. **Phase 3**: Add sophisticated intent analysis with NLP/LLM
4. **Phase 4**: Optimize with advanced parallelization and caching
5. **Phase 5**: Add conversational context and multi-turn queries


## Estimated Implementation Timeline

Development Timeline Breakdown

  Phase 1: Core Infrastructure (1-2 weeks)

  - Set up pydantic-graph integration and basic project structure
  - Define core types (CodeContext, ContentMatch, config models)
  - Create provider system foundation (abstract classes + inference function)
  - Implement basic nodes (AnalyzeIntent, End conditions)
  - Single working graph end-to-end (even if minimal)

  Phase 2: Provider Implementation (1 week)

  - Port existing VoyageAI and Qdrant implementations to new interfaces
  - Integrate pydantic-ai providers
  - File discovery and chunking (reuse existing ast-grep integration)

  Phase 3: Core Workflows (1-2 weeks)

  - Complete all basic nodes (EmbedQuery, SearchVectors, RerankResults)
  - Full indexing workflow (DiscoverFiles → ChunkFiles → EmbedChunks → StoreVectors)
  - Basic search workflow fully functional
  - FastMCP server integration with single `find_code` tool

  Phase 4: Advanced Features (1 week)

  - Hybrid search with parallel execution (ForkFlow, JoinResults)
  - Enhanced intent analysis logic
  - Error handling and edge cases
  - Basic caching layer

  Phase 5: Polish & Tools (1 week)

  - Complete CLI with cyclopts
  - Configuration system and environment integration
  - Testing, debugging, and performance tuning
  - Documentation and examples


  ## Random Thoughts/Considerations

  - We need some mechanism in the indexing for invalidating stale/changed results
    - something like difftastic maybe -- compare semantic reality with the state and somehow apply it as a filter -- if the current state doesn't reflect the core result within a certain level of precision it's filtered out
    - We need to consider mapping state changes to branches. This repo right now is a good example of the dangers -- I wiped the original codebase from this branch completely so we could build fresh and without polluting agent context, but a vector search would likely turn up the abandoned structure as a relevant result
    - invalidating vectors? 