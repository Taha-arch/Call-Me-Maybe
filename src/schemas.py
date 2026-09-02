"""Pydantic models describing the two input files this project consumes."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ParameterType = Literal["number", "string", "integer", "boolean"]


class PromptRequest(BaseModel):
    """A single natural-language request from ``function_calling_tests.json``.

    Attributes:
        prompt: The raw request text as typed by a user.
    """

    prompt: str = Field(min_length=1)

    @model_validator(mode="after")
    def strip_and_check(self) -> "PromptRequest":
        """Trim whitespace and reject prompts that are blank once trimmed."""
        self.prompt = self.prompt.strip()
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        return self


class ParameterSpec(BaseModel):
    """The type of a single function argument or return value."""

    type: ParameterType


class FunctionSpec(BaseModel):
    """A callable function as declared in ``functions_definition.json``.

    Attributes:
        name: Unique identifier the model can select.
        description: Human-readable summary used to help the LLM route
            a request to this function.
        parameters: Mapping of argument name to its expected type.
        returns: The type of value this function produces.
    """

    name: str
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterSpec]
    returns: ParameterSpec

    @model_validator(mode="after")
    def strip_and_check(self) -> "FunctionSpec":
        """Trim whitespace and reject blank identifiers or parameter keys."""
        self.name = self.name.strip()
        self.description = self.description.strip()

        if not self.name:
            raise ValueError("function name must not be empty")
        if not self.description:
            raise ValueError("function description must not be empty")
        for key in self.parameters:
            if not key.strip():
                raise ValueError("parameter names must not be empty")
        return self
