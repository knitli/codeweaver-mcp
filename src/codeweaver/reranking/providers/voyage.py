from collections.abc import Sequence
from typing import cast

from codeweaver.reranking.providers.base import ScoredCodeChunk


try:
    from voyageai.object.reranking import RerankingObject, RerankingResult
    

def voyage_reranking_output_transformer(
    returned_result: RerankingObject,
) -> Sequence[ScoredCodeChunk]:
    """Transform the output of the Voyage AI reranking model."""
    results, token_count = returned_result.results, returned_result.total_tokens
    
    # we need to return the results in the same order as the input
    sorted_results: list[RerankingResult] = sorted(results, key=lambda x: cast(int, x[0]))
    return [cast(float, result[2]) for result in sorted_results]
