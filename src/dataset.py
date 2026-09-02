"""Loading and validating the two JSON input files for this project."""

import json
from typing import Any

from .schemas import FunctionSpec, PromptRequest


def read_json_array(path: str) -> list[Any]:
    """Read *path* as JSON and require it to hold a non-empty array.

    Args:
        path: Filesystem path to a JSON file.

    Raises:
        ValueError: If the file is missing, unreadable, not valid JSON,
            or does not contain a non-empty array.

    Returns:
        The parsed JSON array.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise ValueError(f"input file not found: '{path}'")
    except PermissionError:
        raise ValueError(f"permission denied reading '{path}'")
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in '{path}': {error}")

    if not isinstance(data, list) or not data:
        raise ValueError(f"'{path}' must contain a non-empty JSON array")
    return data


def load_dataset(
    functions_path: str, prompts_path: str
) -> tuple[list[FunctionSpec], list[PromptRequest]]:
    """Load and validate the function catalog and the prompt batch.

    Args:
        functions_path: Path to ``functions_definition.json``.
        prompts_path: Path to ``function_calling_tests.json``.

    Returns:
        A tuple of ``(functions, prompts)``, both validated by pydantic.
    """
    functions = [
        FunctionSpec(**item) for item in read_json_array(functions_path)
    ]
    prompts = [
        PromptRequest(**item) for item in read_json_array(prompts_path)
    ]
    return functions, prompts
