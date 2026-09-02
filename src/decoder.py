"""Constrained decoding: turn model logits into schema-valid JSON leaves.

The output object's braces, keys, and separators are always written by
this module, never sampled from the model — that is what guarantees
valid JSON regardless of what the model does. The model is only ever
asked to produce a *leaf* value (a function name, a boolean, a number,
or a string), and each leaf is restricted, token by token, to the small
set of continuations that are still valid for that value's type. Any
token that would break the type is simply never offered to the model.
"""

import re
from collections.abc import Callable
from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model

from .prompting import extraction_prompt, routing_prompt
from .schemas import FunctionSpec

# How many of the model's top-logit tokens are inspected before falling
# back to a deterministic choice. This stands in for masking the full
# vocabulary to -inf: rather than building a validity mask over every
# token id, we walk the best candidates in order and take the first one
# that keeps generation on a valid path.
TOKEN_SEARCH_WIDTH = 12
MAX_ENUM_STEPS = 20
MAX_VALUE_TOKENS = 24
STRING_STOP_CHARACTERS = '"'
NUMBER_STOP_CHARACTERS = ',} \n"'


class CallEngine:
    """Drives a `Small_LLM_Model` through constrained JSON generation."""

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model

    def choose_function(
        self, user_prompt: str, functions: list[FunctionSpec]
    ) -> FunctionSpec:
        """Pick which function in *functions* answers *user_prompt*.

        The function name is decoded as an enum: only tokens that stay a
        prefix of at least one real function name are ever accepted, so
        the model cannot hallucinate a name that was not offered to it.
        """
        prompt = routing_prompt(user_prompt, functions)
        input_ids = list(self.model.encode(prompt)[0])
        names = [fn.name for fn in functions]
        chosen = self._decode_enum(input_ids, names)
        return next(fn for fn in functions if fn.name == chosen)

    def fill_parameters(
        self, user_prompt: str, function: FunctionSpec
    ) -> dict[str, Any]:
        """Decode a typed value for every parameter of *function*.

        The `"key": ` separators are injected directly into the model's
        context rather than generated, so the model only ever has to
        produce the value that follows each key.
        """
        prompt = extraction_prompt(user_prompt, function)
        input_ids = list(self.model.encode(prompt)[0])
        values: dict[str, Any] = {}
        for index, (key, spec) in enumerate(function.parameters.items()):
            opening = f'"{key}": ' if index == 0 else f', "{key}": '
            input_ids.extend(list(self.model.encode(opening)[0]))
            values[key] = self._decode_value(input_ids, spec.type)
        return values

    # -- one decoder per JSON leaf type --------------------------------

    def _decode_value(self, input_ids: list[int], value_type: str) -> Any:
        if value_type == "string":
            input_ids.extend(list(self.model.encode('"')[0]))
            return self._decode_string(input_ids)
        if value_type == "boolean":
            picked = self._decode_enum(input_ids, ["true", "false"])
            return picked == "true"
        if value_type == "integer":
            return int(self._decode_number(input_ids, allow_decimal=False))
        return self._decode_number(input_ids, allow_decimal=True)

    def _decode_enum(self, input_ids: list[int], choices: list[str]) -> str:
        """Generate text that is guaranteed to end up equal to one of
        *choices*, by only ever accepting tokens that keep the running
        text a prefix of at least one choice."""
        generated = ""
        for _ in range(MAX_ENUM_STEPS):
            remaining = [c for c in choices if c.startswith(generated)]
            if generated in remaining:
                return generated
            if len(remaining) == 1:
                self._append_text(input_ids, remaining[0][len(generated):])
                return remaining[0]

            def keeps_a_choice_open(piece: str) -> bool:
                return any(c.startswith(generated + piece)
                           for c in remaining)

            token_id = self._best_valid_token(input_ids, keeps_a_choice_open)
            if token_id is None:
                # Nothing in the search window keeps every option open;
                # commit deterministically instead of trusting argmax.
                self._append_text(input_ids, remaining[0][len(generated):])
                return remaining[0]
            input_ids.append(token_id)
            generated += self.model.decode([token_id])
        return choices[0]

    def _decode_string(self, input_ids: list[int]) -> str:
        text = ""
        for _ in range(MAX_VALUE_TOKENS):
            if self._peek_stop(input_ids, STRING_STOP_CHARACTERS):
                break
            token_id = self._best_valid_token(
                input_ids,
                lambda piece: '"' not in piece and "\n" not in piece,
            )
            if token_id is None:
                break
            input_ids.append(token_id)
            text += self.model.decode([token_id])
        return text.strip()

    def _decode_number(
        self, input_ids: list[int], allow_decimal: bool
    ) -> float:
        pattern = re.compile(r"-?\d*\.?\d*" if allow_decimal else r"-?\d*")
        text = ""
        for _ in range(MAX_VALUE_TOKENS):
            if text and self._peek_stop(input_ids, NUMBER_STOP_CHARACTERS):
                break
            token_id = self._best_valid_token(
                input_ids,
                lambda piece: bool(pattern.fullmatch(text + piece)),
            )
            if token_id is None:
                break
            input_ids.append(token_id)
            text += self.model.decode([token_id])
        try:
            return float(text)
        except ValueError:
            return 0.0

    # -- low-level token selection --------------------------------------

    def _best_valid_token(
        self, input_ids: list[int], is_valid: Callable[[str], bool]
    ) -> int | None:
        """Return the highest-logit token whose decoded text keeps
        *is_valid* true, searching only the top `TOKEN_SEARCH_WIDTH`
        candidates. Returns `None` if none of them qualify."""
        logits = self.model.get_logits_from_input_ids(input_ids)
        ranked = np.argsort(logits)[::-1][:TOKEN_SEARCH_WIDTH]
        for token_id in ranked:
            piece = self.model.decode([int(token_id)])
            if piece and is_valid(piece):
                return int(token_id)
        return None

    def _peek_stop(self, input_ids: list[int], stop_chars: str) -> bool:
        """Look at the model's single best next token, without
        committing it, to decide whether it is trying to end the
        current value."""
        logits = self.model.get_logits_from_input_ids(input_ids)
        piece = self.model.decode([int(np.argmax(logits))])
        return not piece or piece[0] in stop_chars

    def _append_text(self, input_ids: list[int], text: str) -> None:
        """Encode *text* and append it straight to the running context,
        for the parts of the output this module authors deterministically
        instead of asking the model to generate them."""
        if text:
            input_ids.extend(list(self.model.encode(text)[0]))
