*This project has been created as part of the 42 curriculum by tel-atou.*

# Description

**Call Me Maybe** turns a natural-language request such as *"What is the
sum of 2 and 3?"* into a structured, machine-executable function call:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

It runs a small local language model (`Qwen/Qwen3-0.6B`, 0.6B
parameters) and drives its token generation with **constrained
decoding**: the JSON braces, keys, and separators are written by this
program, never sampled from the model, and every value the model *is*
asked to produce is restricted, token by token, to the small set of
continuations that stay valid for its declared type. The model never
gets a free hand at "please output JSON" and hope for the best — it is
only ever offered choices that keep the output correct.

# Instructions

## Installation

```bash
make install     # uv sync — installs all dependencies from pyproject.toml
```

`llm_sdk/` must sit next to `src/` at the project root (it already does
in this repository — it is the vendor-provided wrapper around the
model and is used as-is, unmodified).

## Usage

```bash
make run
```

which is equivalent to:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All three flags are optional; the paths above are the defaults.

## Other Makefile targets

```bash
make debug        # run the program under pdb
make lint         # flake8 + mypy (mandatory flag set)
make lint-strict   # flake8 + mypy --strict
make clean         # remove __pycache__ / .mypy_cache
```

# Resources

## Classic references

- [How LLMs Actually Generate Text](https://www.youtube.com/watch?v=NKnZYvZA7w4)
- [What are Transformers (Machine Learning Model)?](https://youtu.be/ZXiruGOCn9s)
- [Guiding Text Generation with Constrained Decoding](https://huggingface.co/blog/constrained-beam-search)
- [Structured Generation with LLMs — a survey of approaches](https://arxiv.org/abs/2403.06988)
- [uv documentation](https://docs.astral.sh/uv/)

## AI usage

AI (Claude) was used as a pair-programmer while designing this
project: helping map the subject's requirements to an architecture,
drafting the decoding module from that design, and reviewing two
peers' implementations for inspiration on pitfalls to avoid (notably,
the "extra `error` key on failure" issue described below). All
generated code was read, tested against a scripted fake model (see
*Testing strategy*), and is understood well enough to be defended and
modified during evaluation, per the project's AI-usage guidelines.

# Algorithm Explanation

Generation happens in two passes over the *same* growing token
context, per prompt:

**1. Function routing.** The model is shown every function's name and
description and asked to complete `Chosen function name: "`. This is
decoded as an **enum**: at each step, only tokens whose text keeps the
generated string a *prefix of at least one real function name* are
ever offered to the model (`CallEngine._decode_enum`). If a candidate
step narrows the possibilities to a single function, the rest of its
name is appended directly without any further model calls. The model
therefore cannot hallucinate a function that was never in the catalog
— there is no code path that accepts it.

**2. Parameter extraction.** Once a function is chosen, its parameter
keys and separators (`"key": `) are injected into the context
directly — the model is never asked to reproduce them. For each
parameter's declared type, a matching leaf decoder runs:

- `string` — an opening `"` is injected, then tokens are accepted only
  while they contain no literal `"` or newline; generation stops as
  soon as the model's own top prediction wants to start a `"`.
- `number` / `integer` — tokens are accepted only while the digits
  generated so far still match `-?\d*\.?\d*` (or `-?\d*` for
  `integer`, so a stray decimal point can never appear); generation
  stops once the model's top prediction is a delimiter (`,`, `}`,
  whitespace, or `"`).
- `boolean` — decoded the same way as a function name, as an enum over
  `["true", "false"]`.

Each leaf decoder walks the model's top `TOKEN_SEARCH_WIDTH` (12)
highest-logit tokens and accepts the first one that keeps the value on
a valid path — the closest practical approximation, without a full
vocabulary-to-string table, of masking every invalid token's logit to
`-inf`. If none of the top candidates qualify (rare, since the search
window is wide relative to the tiny valid alphabets involved), the
decoder falls back to a **deterministic** completion instead of
trusting the model's raw argmax, so a bad sampling step can never
produce invalid output.

Finally, every result is assembled as a plain Python `dict` and
serialized with `json.dumps` — the model's text is never spliced
straight into the output file, so JSON validity does not depend on the
model getting escaping right.

# Design Decisions

- **Structure is authored, not generated.** Braces, keys, and
  separators come from this program; only leaf values come from the
  model. This turns "is the JSON valid?" from a probabilistic outcome
  into a structural guarantee.
- **Enum decoding via prefix filtering** unifies function-name and
  boolean selection under one routine (`_decode_enum`) instead of two
  separate ad-hoc string checks.
- **Deterministic fallbacks, not argmax fallbacks.** When the
  constrained search finds nothing valid in its window, the decoder
  commits to a known-valid completion itself rather than trusting an
  unchecked model token — the one place a peer implementation this
  project drew inspiration from could still emit invalid output under
  a fallback, this closes that gap.
- **Final values are re-serialized, not hand-escaped.** Building the
  result dict and calling `json.dumps` at the end sidesteps the
  regex-escaping bugs peer implementations ran into when trying to
  keep the model's raw text valid JSON during generation.
- **Exactly three keys, always.** Output objects never gain an
  `"error"` key on a rough prompt — the subject specifies the output
  schema exactly, so the engine always resolves *some* valid function
  and typed parameters instead of degrading the schema on failure.

# Performance Analysis

- **Structural validity: 100%.** Guaranteed by construction — the
  decoder authors the JSON skeleton itself.
- **Semantic accuracy** depends on how well the routing and extraction
  prompts convey the request; the bundled test set covers arithmetic,
  greetings, string operations, and boolean toggles across all four
  parameter types (`number`, `string`, `integer`, `boolean`).
- **Speed** scales with the number of parameters per function, since
  each leaf value costs at most `MAX_VALUE_TOKENS` (24) model calls;
  simple one-argument functions resolve in a handful of forward
  passes.

# Challenges Faced

- **Telling a hallucinated name from a real one.** Early designs
  checked the *decoded text* for validity after the fact. Switching to
  filtering candidate *tokens* against a live list of still-possible
  names (a prefix trie, in effect) made an invalid function name
  structurally unreachable instead of merely unlikely.
- **Knowing when to stop a value.** A model has no explicit "I'm done"
  signal mid-number. Peeking at its *next* predicted token (without
  committing it) to see whether it wants a delimiter next turned out
  to be a reliable stopping heuristic for both strings and numbers.
- **Avoiding escaping bugs.** Trying to keep the model's raw output
  valid JSON while it generates (handling stray backslashes, embedded
  quotes, etc.) is fragile. Delaying escaping to a single
  `json.dumps` call at the very end removed this class of bug
  entirely.

# Testing Strategy

Since the model itself isn't practical to unit test deterministically,
`CallEngine`'s decoding logic was exercised against a small scripted
fake model (a character-level oracle that always "wants" to produce a
fixed, sometimes deliberately invalid, string) to confirm:

- an out-of-catalog function name is never accepted as a final choice;
- `number` values stop at the right point and never pick up a stray
  decimal point when the target type is `integer`;
- `string` values stop cleanly at the model's own signal to close the
  quote;
- `boolean` values always resolve to exactly `true` or `false`;
- a full `fill_parameters` pass across multiple parameters in one
  context, and a full `run_batch` pass, produce output containing
  exactly the `prompt`, `name`, and `parameters` keys and round-trip
  through `json.dumps` without error.

Beyond that, `data/input/` was run end to end against the real model
and the resulting `data/output/function_calling_results.json` was
checked by hand against `functions_definition.json` for key and type
correctness.

# Example Usage

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Given this input prompt:

```json
{"prompt": "Repeat the word 'ha' 3 times"}
```

the program produces:

```json
{
  "prompt": "Repeat the word 'ha' 3 times",
  "name": "fn_repeat_string",
  "parameters": {"s": "ha", "times": 3}
}
```
