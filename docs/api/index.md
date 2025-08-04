# API Reference

Complete API documentation for CodeWeaver.

## Backends

- [backends › base](backends/base.md)
- [backends › base_config](backends/base_config.md)
- [backends › config](backends/config.md)
- [backends › factory](backends/factory.md)
- [backends › providers › docarray › adapter](backends/providers/docarray/adapter.md)
- [backends › providers › docarray › config](backends/providers/docarray/config.md)
- [backends › providers › docarray › factory](backends/providers/docarray/factory.md)
- [backends › providers › docarray › qdrant](backends/providers/docarray/qdrant.md)
- [backends › providers › docarray › schema](backends/providers/docarray/schema.md)
- [backends › providers › qdrant](backends/providers/qdrant.md)

## Cli

- [cli › app](cli/app.md)
- [cli › commands › client_commands](cli/commands/client_commands.md)
- [cli › commands › config_commands](cli/commands/config_commands.md)
- [cli › commands › index_commands](cli/commands/index_commands.md)
- [cli › commands › services_commands](cli/commands/services_commands.md)
- [cli › commands › stats_commands](cli/commands/stats_commands.md)
- [cli › types](cli/types.md)
- [cli › utils › client_logger](cli/utils/client_logger.md)
- [cli › utils › client_manager](cli/utils/client_manager.md)
- [cli › utils › config_helper](cli/utils/config_helper.md)
- [cli › utils › helpers](cli/utils/helpers.md)
- [cli › utils › server_manager](cli/utils/server_manager.md)

## Commands

- [commands › client](commands/client.md)
- [commands › index](commands/index.md)
- [commands › init_codeweaver](commands/init_codeweaver.md)
- [commands › insert](commands/insert.md)
- [commands › services](commands/services.md)
- [commands › stats](commands/stats.md)

## Config

- [config](config.md)

## Cw_Types

- [cw_types › backends › base](cw_types/backends/base.md)
- [cw_types › backends › capabilities](cw_types/backends/capabilities.md)
- [cw_types › backends › enums](cw_types/backends/enums.md)
- [cw_types › backends › providers](cw_types/backends/providers.md)
- [cw_types › base_enum](cw_types/base_enum.md)
- [cw_types › config](cw_types/config.md)
- [cw_types › content](cw_types/content.md)
- [cw_types › enums](cw_types/enums.md)
- [cw_types › exceptions](cw_types/exceptions.md)
- [cw_types › factories › core](cw_types/factories/core.md)
- [cw_types › factories › data_structures](cw_types/factories/data_structures.md)
- [cw_types › intent › base](cw_types/intent/base.md)
- [cw_types › intent › data](cw_types/intent/data.md)
- [cw_types › intent › enums](cw_types/intent/enums.md)
- [cw_types › intent › exceptions](cw_types/intent/exceptions.md)
- [cw_types › intent › learning](cw_types/intent/learning.md)
- [cw_types › language](cw_types/language.md)
- [cw_types › providers › capabilities](cw_types/providers/capabilities.md)
- [cw_types › providers › enums](cw_types/providers/enums.md)
- [cw_types › providers › registry](cw_types/providers/registry.md)
- [cw_types › services › config](cw_types/services/config.md)
- [cw_types › services › data](cw_types/services/data.md)
- [cw_types › services › enums](cw_types/services/enums.md)
- [cw_types › services › exceptions](cw_types/services/exceptions.md)
- [cw_types › services › services](cw_types/services/services.md)
- [cw_types › sources › capabilities](cw_types/sources/capabilities.md)
- [cw_types › sources › enums](cw_types/sources/enums.md)
- [cw_types › sources › providers](cw_types/sources/providers.md)

## Factories

- [factories › backend_registry](factories/backend_registry.md)
- [factories › base](factories/base.md)
- [factories › codeweaver_factory](factories/codeweaver_factory.md)
- [factories › error_handling](factories/error_handling.md)
- [factories › extensibility_manager](factories/extensibility_manager.md)
- [factories › factory](factories/factory.md)
- [factories › initialization](factories/initialization.md)
- [factories › plugin_protocols](factories/plugin_protocols.md)
- [factories › registry](factories/registry.md)
- [factories › service_registry](factories/service_registry.md)
- [factories › source_registry](factories/source_registry.md)

## Intent

- [intent › caching › intent_cache](intent/caching/intent_cache.md)
- [intent › middleware › intent_bridge](intent/middleware/intent_bridge.md)
- [intent › parsing › confidence_scorer](intent/parsing/confidence_scorer.md)
- [intent › parsing › factory](intent/parsing/factory.md)
- [intent › parsing › pattern_matcher](intent/parsing/pattern_matcher.md)
- [intent › recovery › fallback_handler](intent/recovery/fallback_handler.md)
- [intent › strategies › adaptive](intent/strategies/adaptive.md)
- [intent › strategies › analysis_workflow](intent/strategies/analysis_workflow.md)
- [intent › strategies › registry](intent/strategies/registry.md)
- [intent › strategies › simple_search](intent/strategies/simple_search.md)
- [intent › workflows › orchestrator](intent/workflows/orchestrator.md)

## Language_Constants

- [language_constants › general](language_constants/general.md)
- [language_constants › javascript_family](language_constants/javascript_family.md)
- [language_constants › python](language_constants/python.md)

## Main

- [main](main.md)

## Middleware

- [middleware › chunking](middleware/chunking.md)
- [middleware › filtering](middleware/filtering.md)
- [middleware › telemetry](middleware/telemetry.md)

## Providers

- [providers › base](providers/base.md)
- [providers › config](providers/config.md)
- [providers › custom](providers/custom.md)
- [providers › factory](providers/factory.md)
- [providers › nlp › spacy](providers/nlp/spacy.md)
- [providers › providers › cohere](providers/providers/cohere.md)
- [providers › providers › huggingface](providers/providers/huggingface.md)
- [providers › providers › openai](providers/providers/openai.md)
- [providers › providers › sentence_transformers](providers/providers/sentence_transformers.md)
- [providers › providers › voyageai](providers/providers/voyageai.md)

## Server

- [server](server.md)

## Services

- [services › manager](services/manager.md)
- [services › middleware_bridge](services/middleware_bridge.md)
- [services › providers › auto_indexing](services/providers/auto_indexing.md)
- [services › providers › base_provider](services/providers/base_provider.md)
- [services › providers › caching](services/providers/caching.md)
- [services › providers › chunking](services/providers/chunking.md)
- [services › providers › context_intelligence](services/providers/context_intelligence.md)
- [services › providers › file_filtering](services/providers/file_filtering.md)
- [services › providers › implicit_learning](services/providers/implicit_learning.md)
- [services › providers › intent_orchestrator](services/providers/intent_orchestrator.md)
- [services › providers › middleware](services/providers/middleware.md)
- [services › providers › rate_limiting](services/providers/rate_limiting.md)
- [services › providers › telemetry](services/providers/telemetry.md)
- [services › providers › zero_shot_optimization](services/providers/zero_shot_optimization.md)

## Sources

- [sources › api](sources/api.md)
- [sources › base](sources/base.md)
- [sources › config](sources/config.md)
- [sources › factory](sources/factory.md)
- [sources › integration](sources/integration.md)
- [sources › providers › database](sources/providers/database.md)
- [sources › providers › filesystem](sources/providers/filesystem.md)
- [sources › providers › git](sources/providers/git.md)
- [sources › providers › web](sources/providers/web.md)

## Testing

- [testing › benchmarks](testing/benchmarks.md)
- [testing › factory_validation](testing/factory_validation.md)
- [testing › integration](testing/integration.md)
- [testing › mocks](testing/mocks.md)
- [testing › protocol_compliance](testing/protocol_compliance.md)

## Utils

- [utils › decorators](utils/decorators.md)
- [utils › helpers](utils/helpers.md)

