# Dependency Plan

Note: All mentions of `pydantic` refer to the version 2+ API. (We use the latest stable, `pydantic 2.11.7`)

> IMPORTANT:  A word on MPC
> 
> CodeWeaver is at its core an MCP server and MCP extension platform. While MCP uses a client-server model, don't make the mistake of assuming it's equivalent to *http* server-client models. `FastMCP` does such a good job of abstracting away the underlying protocols that it can be easy to forget what's happening under the hood. But some key points:
>    - modern MCP uses one of two transports: 
>        1. `stdio`, which is a protocol for local communications. In this scenario your server and client are both local, or optionally the local server is proxying a remote server. Our main server will default to `stdio`, as this is the most common use case, but easily modified to alternatives.
>        2. `streamable http`, the protocol for remote/web-based communication. 
>
>    - A third transport, SSE, is now deprecated in favor of streamable HTTP
>    - FastMCP exposes additional transports beyond the two MCP protocols
>      - You can expose custom http endpoints. You could use this for providing an admin dashboard, for example. (https://gofastmcp.com/deployment/running-server#custom-routes) It is a starlette powered web server *and* and an MCP server.
>      - There's a `FastMCP` transport that's mostly for debugging/dev -- creating a direct link between the client and server. They also call this the `in-memory` transport. 
>
>    - While many communications **are *over* http**, the actual data exchange uses `JSON-RPC`. Like I said, `FastMCP` handles all this transparently *if you use its features*. If you venture out of that then you invite those complications on yourself (read: don't). FastMCP provides everything you could want in this regard -- ability to push requests (`elicitation` and `sampling`) and messages (`messages`) to clients, log exchanges, data exchanges, and more. It has a robust dependency injection-driven middleware ability that provides just about anything -- multiple auth, logging, error handling, progress reporting... you name it. If it involves possible communicating with a client, it's best to use FastMCP to do it.
>    - **The relationships aren't clean-cut in MCP**. Unlike in most traditional http client-server deployments, the client-server relationships can be much more dynamic:
>       - A server may routinely act as a client for various tasks, such as to proxy other MCP services either for its own use or to pass on to the client. 
>       - An LLM 'user' may at times act as both a server asset, using the `sampling` protocol, and as a client, and as its own server.
>       - clients also routinely act as servers, such as to proxy remote servers. 
>       - One deployment model `FastMCP` recommends is essentially dividing your app into multiple MCP servers with a central proxy or decentralized bilateral client-server relationships between each other (see https://gofastmcp.com/servers/composition)
>         - This has the advantage of keeping each component isolated and easier for larger teams to maintain
>
> **final note**: Always remember that the direct 'user' of an MCP tool is an AI agent (LLM). The human "user" is the benefactor.
>
>  For clarity and to keep these relationships clear we distinguish these stakeholders as: `The Agent` -- the LLM user, `Developer` or `end user` to describe the *human* making the decision to download and enable our tools and whose needs we are all ultimately trying to satisfy, and we... we're `we` -- part of a human-agent team to make agent-developer teams far more effective.

## CLI

`cyclopts` for cli. Has a similar API to `typer` but a bit more modern, and only uses a single `Parameter` object instead of distinguishing between arguments and parameters. You use the python `*` and `/` markers to denote keyword and positional arguments like:
        
    ```python
    from cyclopts import App, Parameter
    from typing import Annotated

    app = App()

    @app.command  # <-- no `()` required, but you can provide args. 
                  # `Command's api surface is identical to App`
    def add(
           # required positional argument
        x: Annotated[int, Parameter(name=["--first-value", "-f"], alias="first")],

           # this will be an optional positional argument because of the default value
        y: Annotated[int, Parameter(name=["--second-value", "-s"], alias="second")] = 10,

        *, # <-- anything after this is a keyword argument, requiring a flag to be set

        z: Annotated[int, Parameter(name=["--third-value", "-t"], alias="third")] = 20
    ) -> int:
        return x + y + z

    if __name__ == "__main__":
        app()  # also supports async, you'd just use asyncio.run here
    ```

## App-Wide Config System

`pydantic-settings` powers all configuration
  - pydantic `BaseSettings` models provide full validation, toml reading, nested env vars with minor config 
  - Provides a clean pipeline to the rest of the app that is entirely built from `BaseModel`
     - App also uses `FastMCP 2.10+`, `pydantic-ai 0.62`, `pydantic-graph 0.62` -- all built on pydantic v2+ (the latter two along with pydantic settings are created/maintained by the pydantic team)


## Server

- `FastMCP` provides a `FastAPI`-like extensible server platform, built on `starlette` and `httpx` (see notes above). Capabilities important to CodeWeaver include:
  - the core server (and client, which we expose as a convenience through the CLI for debugging, quick use, etc). 
  - Middleware -- error_handling, logging, timing, rate-limiting
  - auth -- most deployments don't need auth but it needs to be there, and it's easy to expose (permit.io, AuthKit, and Eunomia, it also exposes an oauth model and jwt verification)
     - While easy to expose we should look to register scopes for our internal tools so the user has granular control over CodeWeaver's behavior and permissions.
  - cli
     - Our CLI largely wraps FastMCPs for debugging, MCP setup and similar functionality
  - Sampling, progress reporting, possibly elicitation -- sampling in particular should form a baseline foundation to power our intent layer system.
  - Structured input and output

- Nearly all of FastMCP's types are pydantic models. 
  - See above regarding FastMCP server composition -- that's one possible approach for architecture -- modeling the app internally as a collection of servers providing servers, with fastmcp doing the heavy lifting. Notably this capability has complete process isolation, so each server is isolated and thread safe.

- FastMCP's `Tool.from_tool` and generaling proxying abilities are another way to add internal capabilities and services from remote and packaged sources (e.g. use an MCP tool internally as part of our pipeline)

- Context. When passing a request, `FastMCP` provides a very robust context object that can heavily inform our intent resolution/context selection.

## Other General Services

- `PostHog` -- post hog for app-wide opt-out telemetry (ask at first use of CLI and make it super clear in the readme and docs). Our implementation should be privacy minded -- the goal here is understanding how people and agents use the tools and what makes them successful or unsuccessful

- No specific dependencies in mind, but we need a strong statistics, status, health, and token use backend service both to support telemetry and to support marketing on the commercial side (i.e. users of codeweaver use 40% less tokens), and a/b testing.

## Functional Services

- `watchlist` (also from the creators of pydantic...) is a rust-backed python library for fast/efficient filesystem monitoring. It's basically a fast watchdog. 


- `tiktoken` -- estimating token use, costs for telemetry, user reporting, and sticking to quotas (user config or agent selected caps)

- `tomli-w` -- for writing generated configs (pydantic settings reads and serializes)

- `rignore` respecting .gitignore and similar ignore files, quickly walking directories while respecting boundaries

- Note: Probably not an initial-release feature, but we'd like to plan for abstracted file systems -- our file filter and watching services should be able to be remote and hook/http driven just as easily as locally deployed. 

- Together the above form the foundational services for efficient automated/background indexing and of course assisting developers with configs

### Search/discovery

- `ast-grep-py` -- leverage ast-grep's core semantic search capabilities for intelligent indexing (find by kind and index/generate embeddings)

- `qdrant-client` - The first search provider should be qdrant, allowing for local databasing or using qdrant cloud (most repos will stay in the free tier). Qdrant has top notch hybrid search capabilities these days with multi-embedding and sparse indexing and resolution capabilities. We want to make it very easy to extend CodeWeaver to other providers, but our first implementation is qdrant

- `voyage-ai` - Voyage's embeddings and reranking models are best-in-class and outperform much heavier weight models (it significantly outperforms OpenAI's text-embedding-large with a fourth of the vectors). Most repos will stay comfortably in its generous free tier.

- We will flexibly ingrain pydantic-ai for all other providers, and in fact our implementation for voyage-ai should probably use pydantic-ai's protocols/abc/models.
  - pydantic ai allows for flexible installation, so we can expose feature flags for different providers with the qdrant/voyage as default. 
  - more below on pydantic-ai

## App core

Fundamentally we have several design goals for the core functionality of CodeWeaver:
  
1. Fully developer extensible, and ability to define resolution pipelines and strategies programmatically or through configuration
2. Simple for developers; simple for agents. Those two things are very different. 
   - Pydantic-like abilities for developers allows powerful extension of all parts without adding complexity.
   - For agents, simplicity means "delivering the exact context they need to accomplish a task" -- aggressively minimizing excess and context pollution
     - Agent interfaces should be simple and intuitive, allowing them to focus solely on the task at hand
     - As such, **we expose the user's agent to a single tool** -- the `code_context` tool
        - using the `sampling` capability, we can expose a different instance of the same agent to more internal tooling and use it to sift through data, but we keep that context isolated, exposing CodeWeaver's full capabilities for an agent to curate for the receiving agent.
        - Importantly we don't want to rely entirely on this -- we need very capable fallback options for sandboxed environments, privacy-focused folks, or just when the internet connection is splotchy.

3. Abstracted and modular, but not over-engineered. Pydantic and FastMCP (and others... FastAPI, SQLModel, et all) provide great models for architectures that are powerful, flexible, but remain simple and intuitive. 

### Pydantic-AI

- Pydantic-AI gives us the instant ability to leverage ~10 AI providers for internal use in our strategy/intent resolution and for offering embedding alternatives to end users. Currently available providers (each optionally installed individually by feature flags):

   ```plaintext

   Pydantic AI is model-agnostic and has built-in support for multiple model providers:

    OpenAI
    Anthropic
    Google Gemini (via two different APIs: Generative Language API and VertexAI API)
    Groq
    Mistral
    Cohere
    Bedrock (AWS)
    Hugging Face
    OpenAI-compatible Providers


    In addition, many providers are compatible with the OpenAI API, and can be used with OpenAIModel in Pydantic AI:

    DeepSeek
    Grok (xAI)
    Ollama
    OpenRouter
    Vercel AI Gateway
    Perplexity
    Fireworks AI
    Together AI
    Azure AI Foundry
    Heroku
    GitHub Models
    ```

This gives us out-of-the-box local, privacy-friendly, and commercial embedding and reranking models, and also BYOM-type capabilities, such as for enterprises (with bedrock, openai compatible APIs, azure AI and github models). 

- Besides embedding generation and possibly augmenting client sampling, we can also use Pydantic AI for *precontext generation*. One of our major launch features is essentially exposing the agent interface to the human developer for context generation, (or general question/answer and providing context for human needs). We can't use sampling for that because sampling requires an active mcp session. But we can expose our own client over the CLI to the end user, with a provided API key/chosen provider, or local models, (or just our fallback capabilities) to generate precontext -- the goal is to launch the conversation with an agent who immediately understands the exact information required to assist the user. Pydantic AI and FastMCP make that trivial from a supporting services standpoint.
- Pydantic AI exposes useful tools for agentic programming that we can leverage in all parts of the app, like its `Agent` model.
- Finally, pydantic ai exposes some common tools that can support information discovery and context resolution:
   - tavily search's api (webscraper, site maps, crawler, fast search resolution, etc)
   - duckduckgo search for people who don't want to pay for tavily
   - It also exposes powerful builtin tools for python, node execution in a secure environment
        - This is one way we could use another data source I'd like to pull in the (node) `context7` mcp tool, which is basically a searchable vector database of library docs, which we could use both directly and with an agent to sift through it. 
   - longer term, we can use pydantic-ai's compatibility with the AG-UI protocol to provide a web dashboard for tool monitoring and interaction with the app, which could be combined with fastmcp's custom routes for a variety of uses (ag-ui is a standardized protocol for agent interactions and communications with front end UIs)

- Agent `evals` (pydantic evals is its own dependency) provides a powerful framework for automatically and systematically evaluating model and pipeline performance and results and includes open telemetry support


### Pydantic-Graph

- `pydantic-graph` is the pipelining foundation that pydantic-ai uses, so it's already an indirect dependency. Its `Graph` and `Node` models provide really simple and powerful ways to define and orchestrate modular pipelines. We could use this as the foundation for our extensible/definable strategy/intent system (just as the `Agent` model in Pydantic AI does). 
  - As an added bonus, it can *generate mermaid charts of itself*, which is not only useful for providing clear and intuitive developer experience and ui, but also we could potentially use it as part of our context generation intelligence.
  - We actually don't strictly need pydantic-graph as a dependency because pydantic-ai rexports it: https://ai.pydantic.dev/graph/#graph (unless there are small parts we need that aren't exported)
    - I think our use case makes sense, and that using pydantic-graph is well-aligned with out architecture, but we should probably carefully consider the warning at the top of that page...

### Other data sources

- Not necessarily for launch, but we need to be able to integrate diverse data sources. Two tools that I particularly want to bring into the context resolution are


### The fallback

- We need intelligence context resolution outside of relying on agents, and on of our goals is cutting context bloat overall, so sifting it to different agents doesn't solve the problem.
  - I think we could use something like `gensim` and our own searching capabilities to cull down the initial tranche of data for the agent to evaluate
- We would need something more robust for resolution without agents .... like a SpaCy, which is a pretty heavy dependency so we should give options. 