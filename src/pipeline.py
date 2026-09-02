"""Orchestrates the routing + extraction pipeline over a batch of prompts."""

from typing import Any

from .decoder import CallEngine
from .schemas import FunctionSpec, PromptRequest


def run_batch(
    engine: CallEngine,
    functions: list[FunctionSpec],
    prompts: list[PromptRequest],
) -> list[dict[str, Any]]:
    """Resolve every prompt in *prompts* into a structured function call.

    Args:
        engine: Constrained-decoding engine bound to a loaded model.
        functions: The catalog a prompt's function may be chosen from.
        prompts: The batch of natural-language requests to resolve.

    Returns:
        One JSON-serializable object per prompt, each holding exactly
        the ``prompt``, ``name``, and ``parameters`` keys.
    """
    results = []
    for request in prompts:
        function = engine.choose_function(request.prompt, functions)
        parameters = engine.fill_parameters(request.prompt, function)
        results.append({
            "prompt": request.prompt,
            "name": function.name,
            "parameters": parameters,
        })
    return results
