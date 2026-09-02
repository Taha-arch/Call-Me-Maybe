"""Text templates that give the LLM context before constrained decoding."""

from .schemas import FunctionSpec


def routing_prompt(user_prompt: str, functions: list[FunctionSpec]) -> str:
    """Build the context used to pick which function answers *user_prompt*.

    The model only ever fills in the function name; the surrounding JSON
    punctuation is injected by the decoder, so this text only needs to
    give the model enough signal to make a good choice.
    """
    catalog = "\n".join(
        f"- {fn.name}({', '.join(fn.parameters)}): {fn.description}"
        for fn in functions
    )
    return (
        "You are a function routing engine. Read the request and pick "
        "the one function from the catalog that satisfies it.\n\n"
        f"Catalog:\n{catalog}\n\n"
        f'Request: "{user_prompt}"\n'
        'Chosen function name: "'
    )


def extraction_prompt(user_prompt: str, function: FunctionSpec) -> str:
    """Build the context used to fill in *function*'s argument values.

    Ends right after the opening ``{"parameters": {`` so the decoder can
    keep appending ``"key": value`` pairs onto the same running context.
    """
    fields = ", ".join(
        f"{name} ({spec.type})" for name, spec in function.parameters.items()
    )
    return (
        "You extract argument values for a function call from a "
        "request. Use only information present in the request.\n"
        f"Function: {function.name} - {function.description}\n"
        f"Arguments to fill: {fields}\n"
        f'Request: "{user_prompt}"\n\n'
        '{"parameters": {'
    )
