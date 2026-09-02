# Call Me Maybe — Full Project Walkthrough

This document exists for one purpose: so you can sit down, read it top to
bottom, and afterwards explain **any single line** of this project from
memory — which is exactly what the 42 evaluation guidelines expect of you.
`README.md` answers "what is this and how do I run it." This document
answers "why does it work, and what is every line actually doing."

It is organized in four parts:

1. **Theory** — the background knowledge the code assumes you have.
2. **Project Map** — what each file is responsible for, and how they fit
   together.
3. **Execution Trace** — the entire program, followed in the exact order
   it actually runs, line by line, starting at the `if __name__ ==
   "__main__":` guard and ending when the process exits.
4. **A worked example** — one prompt followed all the way through, with
   concrete values at every step, plus a Q&A of the non-obvious design
   choices.

---

## Part 1 — Theory

### 1.1 The problem: LLMs don't naturally speak JSON

A language model like `Qwen/Qwen3-0.6B` is trained to do one thing:
given some text, predict a probability distribution over "what token
comes next." Nothing about that objective guarantees the output is
valid JSON, uses the right function name, or has the right argument
types. Ask a small model to "output JSON" and, left to its own
devices, it might succeed 30% of the time — it drifts, forgets a
closing brace, writes a number as `"42"` instead of `42`, or invents a
function that doesn't exist. Production systems that claim 99%+
reliability with small models are not relying on the model's good
behavior; they are **not letting the model make structural decisions
at all.**

### 1.2 How a model actually generates text

Every call to the model advances the conversation by exactly one
token, through this pipeline:

```
prompt text
   │  tokenizer.encode()
   ▼
input_ids            (a list of integers, one per sub-word token)
   │  model forward pass
   ▼
logits                (one raw score per vocabulary entry, ~150k of them
   │                    for Qwen — NOT yet a probability distribution)
   │  argmax / sampling / (our constrained selection)
   ▼
next token_id
   │  tokenizer.decode()
   ▼
next token's text, appended to the prompt — repeat
```

A **logit** is just a raw, unnormalized score the model assigns to a
candidate token; a higher logit means the model thinks that token is a
more likely continuation. `argmax` means "the index of the largest
value" — i.e. "the single token the model likes best." Normally you'd
either take the argmax every step (greedy decoding) or sample from the
distribution. Both let the model decide the *entire* output, including
its structure.

### 1.3 Constrained decoding, in theory

Constrained decoding intervenes **before** a token is chosen:

1. The model produces logits for every token in its vocabulary.
2. You already know, from the grammar/schema you're enforcing, which
   of those tokens would keep the output valid if chosen.
3. You set the logits of every *invalid* token to `-infinity`.
4. Whatever selection method you use afterwards (argmax or sampling)
   can now only ever land on a valid token, because the invalid ones
   have been mathematically removed from consideration.

That's the textbook version, and it requires knowing, for every one of
Qwen's ~150,000 vocabulary entries, whether appending its text would
still be valid. Doing that exactly requires a full token-id → string
table (built from the tokenizer's vocabulary file) and re-running your
grammar check against all ~150,000 candidates on every single step.

### 1.4 The practical version this project actually uses

Building and maintaining a full-vocabulary validity mask for every
step is expensive and, for a small BPE (byte-pair encoding) vocabulary
with awkward space/punctuation encodings, fiddly to get exactly right.
This project uses a cheaper approximation that achieves the *same
guarantee* for the cases that matter here:

> Instead of checking all ~150,000 tokens, ask the model for its
> **top `TOKEN_SEARCH_WIDTH` (12) highest-logit tokens**, and check
> only those against the validity rule. Accept the first one that
> passes. If literally none of the top 12 pass — which is rare, since
> the valid alphabet at any point (digits, or letters of a known
> function name) is small and the model is a language model, so it
> tends to already prefer plausible characters — **fall back to
> completing the value deterministically ourselves**, rather than
> trusting an unchecked token.

The second half of that (the deterministic fallback) is the part that
turns "highly reliable" into "provably always valid": there is no code
path in this project where an invalid token can end up in the output,
because the one case the top-12 search can't handle is handled by code
that doesn't ask the model at all.

The other half of the strategy — and arguably the more important one
— is explained next.

### 1.5 The real trick: don't generate what you can just write

Every JSON object this project produces has the shape:

```json
{"prompt": "...", "name": "...", "parameters": {"key": value, ...}}
```

Almost all of those characters — `{`, `}`, `"prompt":`, `"name":`,
`"parameters":`, the commas, the key names, the colons — are *known in
advance*. There is no reason to make a 0.6B-parameter model spend a
generation step deciding whether the next character is `{` or
whitespace. This project's decoder **writes every structural character
itself** and only ever calls the model to fill in a *leaf value*: a
function name, a boolean, a number, or a string. This is why the
output JSON is always syntactically valid regardless of what the model
does with its leaf values — validity was never in question, because
the model was never in charge of structure.

### 1.6 Glossary (skip if these are already familiar)

| Term | Meaning |
|---|---|
| **Token** | A sub-word unit the tokenizer splits text into (e.g. `"hello"` might be one token, `"unbelievable"` might be three). |
| **Token id** | The integer index of a token in the model's vocabulary. |
| **`input_ids`** | The list of token ids representing everything the model has "seen" so far. |
| **Logit** | The model's raw, unnormalized score for one candidate next token. Higher = more likely. |
| **`argmax`** | "The index of the largest value" — the single best-scoring candidate. |
| **Autoregressive generation** | Producing text one token at a time, each new token conditioned on everything generated before it. |
| **Constrained decoding** | Restricting which tokens can be selected at each generation step so the output is guaranteed to satisfy some structure. |
| **Enum decoding** | A special case of constrained decoding where the valid continuations are a small, fixed set of complete strings (here: function names, or `"true"`/`"false"`). |
| **Prefix filtering / trie** | Only accepting a candidate if it keeps the text-so-far a *prefix* of at least one still-possible final answer. |
| **Pydantic model** | A Python class that validates and parses data against declared types automatically. |

---

## Part 2 — Project Map

### 2.1 Directory layout

```
Call-Me-Maybe/
├── llm_sdk/__init__.py       vendor-provided model wrapper (unmodified)
├── data/input/                sample functions + prompts
├── src/
│   ├── __main__.py            entry point / CLI / orchestration
│   ├── schemas.py              pydantic input models
│   ├── dataset.py               file loading + validation
│   ├── prompting.py            text templates fed to the model
│   ├── decoder.py               the constrained-decoding engine
│   └── pipeline.py              loops the engine over every prompt
├── pyproject.toml, Makefile, README.md, .gitignore
```

### 2.2 Who is responsible for what

| Module | Responsibility |
|---|---|
| `llm_sdk` | Loads Qwen3-0.6B and exposes `encode`, `decode`, and `get_logits_from_input_ids`. Provided; not our code. |
| `schemas.py` | Defines what a valid prompt / function definition looks like, and rejects anything that doesn't fit. |
| `dataset.py` | Reads the two JSON files off disk and turns them into validated `schemas.py` objects. |
| `prompting.py` | Builds the two pieces of natural-language context the model sees (one for choosing a function, one for filling its arguments). |
| `decoder.py` | The actual constrained decoding: turns model logits into a guaranteed-valid function name and argument values. |
| `pipeline.py` | Glue: run the decoder over every prompt, and shape the results into the required 3-key objects. |
| `__main__.py` | CLI parsing, wiring everything together, error handling, writing the output file. |

### 2.3 Data flow for one prompt

```
"What is the sum of 2 and 3?"
        │
        ▼
routing_prompt() ──▶ CallEngine.choose_function() ──▶ FunctionSpec("fn_add_numbers")
        │
        ▼
extraction_prompt() ──▶ CallEngine.fill_parameters() ──▶ {"a": 2.0, "b": 3.0}
        │
        ▼
{"prompt": "...", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

---

## Part 3 — Execution Trace, line by line

This section follows the program in the order it **actually executes**,
not file order. Whenever the call stack jumps into another file, this
document jumps with it.

### 3.0 — Where everything begins

Running `uv run python -m src` makes Python execute `src/__main__.py`
as the `__main__` module. The very last lines of that file are where
control first lands:

```python
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted, exiting.")
    except Exception as error:  # noqa: BLE001 - top-level safety net
        print(f"[!] Unexpected error: {error}")
```

- `if __name__ == "__main__":` — true when this file is run directly
  (as opposed to imported), which is exactly what `python -m src`
  does for a package's `__main__.py`.
- The whole run is wrapped in `try/except` as a **safety net**: per the
  subject's rule that the program "must never crash unexpectedly," if
  `Ctrl+C` is pressed, we print a clean message instead of a
  traceback; if literally anything else goes wrong that wasn't already
  caught closer to its source, we print it and exit cleanly instead of
  dumping a stack trace on the user.
- This is the *outermost* frame. Everything else in this document
  happens inside the `main()` call on the second line.

### 3.1 — `main()` begins: parsing arguments

```python
def main() -> None:
    """Run the full pipeline: load, decode, and persist the results."""
    args = parse_args()
```

`parse_args()` is defined just above `main()` in the same file:

```python
def parse_args() -> argparse.Namespace:
    """Define and parse the program's command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into structured "
                    "function calls using a constrained local LLM.",
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function catalog JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the batch of natural-language prompts.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path the resolved function calls are written to.",
    )
    return parser.parse_args()
```

- `argparse.ArgumentParser(...)` creates a parser that will also
  auto-generate `--help` text from the `description` and each
  argument's `help=`.
- Each `add_argument` call declares one optional CLI flag. None of
  them are marked `required=True`, and each carries a `default=`, so
  the program runs with zero flags exactly as the subject specifies:
  reading from `data/input/` and writing to `data/output/` unless the
  user overrides a path.
- `parser.parse_args()` reads `sys.argv`, matches it against the
  declared flags, and returns an `argparse.Namespace` — an object
  where `args.functions_definition`, `args.input`, and `args.output`
  are attributes holding either the flag's value or its default.

Back in `main()`, `args` now holds all three resolved paths.

### 3.2 — Loading and validating the input files

```python
    try:
        functions, prompts = load_dataset(
            args.functions_definition, args.input
        )
    except ValidationError as error:
        print(f"[!] Invalid input data: {error.errors()[0]['msg']}")
        return
    except ValueError as error:
        print(f"[!] {error}")
        return
```

This calls into `src/dataset.py`. Two different exception types are
caught deliberately: `pydantic.ValidationError` (a field had the wrong
shape — e.g. a parameter's `type` wasn't one of the four allowed
values) prints pydantic's own message; `ValueError` (a file is
missing, unreadable, or not valid JSON — raised by our own code)
prints that message instead. Either way, `return` exits `main()`
*before* ever touching the model, so a bad input file fails fast with
a clear message rather than crashing deep inside model code.

#### 3.2.1 — `load_dataset()` (`src/dataset.py`)

```python
def load_dataset(
    functions_path: str, prompts_path: str
) -> tuple[list[FunctionSpec], list[PromptRequest]]:
    """Load and validate the function catalog and the prompt batch."""
    functions = [
        FunctionSpec(**item) for item in read_json_array(functions_path)
    ]
    prompts = [
        PromptRequest(**item) for item in read_json_array(prompts_path)
    ]
    return functions, prompts
```

- `read_json_array(functions_path)` (walked through next) returns a
  plain Python `list` of `dict`s straight from JSON — at this point,
  totally unvalidated.
- `[FunctionSpec(**item) for item in ...]` is a list comprehension
  that, for every raw `dict`, calls `FunctionSpec(**item)` —
  unpacking the dict's keys as keyword arguments into the pydantic
  model's constructor (e.g. `FunctionSpec(name=..., description=...,
  parameters=..., returns=...)`). This is the moment validation
  actually happens: pydantic checks every field's type and runs the
  model's custom validators (see §3.2.3). If anything is wrong, this
  line is what raises `ValidationError`, caught back in `main()`.
- The same pattern validates every prompt into a `PromptRequest`.
- Both lists are returned as a tuple, unpacked in `main()` as
  `functions, prompts`.

#### 3.2.2 — `read_json_array()` (`src/dataset.py`)

```python
def read_json_array(path: str) -> list[Any]:
    """Read *path* as JSON and require it to hold a non-empty array."""
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
```

- `with open(path, "r", encoding="utf-8") as handle:` opens the file
  as a **context manager** — the file is guaranteed to be closed when
  the block exits, even if an exception is raised inside it (this is
  the "prefer context managers for resources like files" rule from
  the subject, satisfied for free by `with`).
- `json.load(handle)` parses the file's contents into Python objects
  (`list`, `dict`, `str`, `float`, `bool`, `None`, following JSON's
  own type rules).
- The three `except` clauses each translate a specific low-level
  Python exception into one clear, user-facing `ValueError` — this is
  the "handle exceptions gracefully" / "clear error messages" rule
  from the subject: a missing file, a permissions problem, and
  malformed JSON are three genuinely different problems, so they get
  three different messages instead of one generic "something broke."
- After the `try/except`, `data` holds whatever `json.load` produced
  — but the subject requires the top level of both input files to be
  a JSON **array**. `isinstance(data, list)` checks that Python-side;
  `or not data` additionally rejects an empty array (an empty
  functions catalog or an empty prompt batch can't do anything
  useful).
- On success, the raw (still unvalidated per-item) list is returned.

#### 3.2.3 — The pydantic models (`src/schemas.py`)

```python
ParameterType = Literal["number", "string", "integer", "boolean"]
```

`Literal[...]` is a type that only accepts these four exact string
values — this alone is what enforces "unsupported parameter type"
rejection; pydantic checks it automatically, no custom code needed.

```python
class PromptRequest(BaseModel):
    """A single natural-language request from `function_calling_tests.json`."""

    prompt: str = Field(min_length=1)

    @model_validator(mode="after")
    def strip_and_check(self) -> "PromptRequest":
        """Trim whitespace and reject prompts that are blank once trimmed."""
        self.prompt = self.prompt.strip()
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        return self
```

- `class PromptRequest(BaseModel):` — inheriting from pydantic's
  `BaseModel` is what turns a plain class into a self-validating data
  contract.
- `prompt: str = Field(min_length=1)` declares one field, `prompt`,
  which must be a string and, at the raw-input stage, at least 1
  character. (This catches `""` but not `"   "` — that's what the
  validator below is for.)
- `@model_validator(mode="after")` marks the method below to run
  *after* pydantic has already checked the field types — so inside
  it, `self.prompt` is guaranteed to already be a `str`.
- `self.prompt = self.prompt.strip()` removes leading/trailing
  whitespace.
- `if not self.prompt:` — an empty string is falsy in Python, so this
  catches a prompt that was *only* whitespace (which passed the
  `min_length=1` check before stripping, but shouldn't count as a
  real prompt).
- `raise ValueError(...)` — pydantic catches `ValueError` raised
  inside a validator and wraps it into its own `ValidationError`,
  which is what `main()` catches in §3.2.
- `return self` — `mode="after"` validators must return the
  (possibly modified) model instance.

```python
class ParameterSpec(BaseModel):
    """The type of a single function argument or return value."""

    type: ParameterType
```

A one-field model: `{"type": "number"}` becomes a `ParameterSpec`
whose `.type` is guaranteed to be one of the four allowed strings.
Used both for each entry in a function's `parameters` dict and for
its `returns`.

```python
class FunctionSpec(BaseModel):
    """A callable function as declared in `functions_definition.json`."""

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
```

- `parameters: dict[str, ParameterSpec]` is the interesting field: it
  tells pydantic that every *value* in the `parameters` dict from the
  JSON file (e.g. `{"a": {"type": "number"}}`) must itself parse into
  a `ParameterSpec` — so nested validation happens automatically, one
  level deep, with no manual loop required.
- `returns: ParameterSpec` — same idea for the single `returns`
  field.
- The validator mirrors `PromptRequest`'s: strip `name` and
  `description`, reject either if now blank, and reject any parameter
  whose key is blank (or only whitespace) once stripped.

At the end of §3.2, `functions` is a `list[FunctionSpec]` and
`prompts` is a `list[PromptRequest]` — both fully validated, both back
in `main()`.

### 3.3 — Loading the model

```python
    print(f"[*] Loaded {len(functions)} function(s) and "
          f"{len(prompts)} prompt(s).")
    print("[*] Loading model, this may take a moment...")

    model = Small_LLM_Model()
    engine = CallEngine(model)
```

- Two progress prints — useful because model loading (next line) can
  take a while the first time (weight download) or a few seconds
  every time after (loading weights into memory).
- `Small_LLM_Model()` is the vendor SDK class, imported at the top of
  this file via `from llm_sdk import Small_LLM_Model`. Walking into
  it now, since this is the moment it actually runs:

#### 3.3.1 — `llm_sdk/__init__.py` (provided; not modified)

```python
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, \
        PreTrainedTokenizer, PreTrainedModel, logging
    from huggingface_hub import hf_hub_download
    from typing import Any
except ImportError as e:
    print(f"Import Error: {e}")
    exit()
```

If `torch`/`transformers`/`huggingface_hub` aren't installed (i.e.
`uv sync` wasn't run), this prints a message and calls `exit()`
immediately at *import* time — before any of our own code runs. This
is why `make install` has to happen before `make run`.

```python
logging.set_verbosity_error()  # keep the console clean
```

Suppresses `transformers`' usually very chatty informational logging,
so only errors show up.

```python
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name
```

`Small_LLM_Model()` with no arguments defaults `model_name` to
`"Qwen/Qwen3-0.6B"`, exactly as the subject mandates. The `*` makes
every parameter after it keyword-only (you can't accidentally pass
`device` positionally).

```python
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device
```

Auto-picks the best available compute device: Apple Silicon GPU
(`mps`), then NVIDIA GPU (`cuda`), then falls back to `cpu` — so the
same code runs anywhere without configuration.

```python
        if dtype is None:
            dtype = torch.float16 if self._device in ["cuda", "mps"] else\
                  torch.float32
        self._dtype = dtype
```

On a GPU, use 16-bit floats to halve memory use; on CPU, use 32-bit
floats for correctness (many CPU kernels don't support fp16 well).

```python
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
```

Downloads (or loads from cache) Qwen's tokenizer. If it has no
padding token defined, it reuses the end-of-sequence token as one —
this project never batches multiple sequences together, so padding is
never actually exercised, but the guard avoids a crash if any
internal `transformers` code path expects one to exist.

```python
        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        for p in self._model.parameters():
            p.requires_grad = False
```

Downloads/loads the actual model weights, moves the model onto the
chosen device, and switches it to **inference mode** twice over:
`.eval()` disables training-only behavior like dropout, and setting
`requires_grad = False` on every parameter tells PyTorch not to track
gradients — this project only ever reads logits, never trains, so
gradient tracking would be pure wasted memory and compute.

Three methods on this class are what the rest of the project actually
calls, at every single generation step:

```python
    def encode(self, text: str) -> torch.Tensor:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)
```

Turns a string into token ids (the model's own tokenizer decides how
it splits text — `"hello"`, `"fn_add_numbers"`, `" the"`, etc. may
each be a different number of tokens; you never need to know this in
advance). `add_special_tokens=False` means no automatic
beginning/end-of-sequence markers are injected — this project builds
its own prompt text explicitly and doesn't want the tokenizer adding
anything extra. The result is wrapped as a batch of size 1
(`[ids]`), matching what the model's forward pass expects.

```python
    def decode(self, ids: torch.Tensor | list[int]) -> Any:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)
```

The inverse: token ids back to text. Our code always passes plain
`list[int]`, so the `isinstance` branch is mostly there for
flexibility; `skip_special_tokens=True` strips anything like an
end-of-sequence marker out of the returned text.

```python
    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        input_tensor = torch.tensor(
            [input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]
```

This is the single most important method in the whole project: given
*every* token id generated so far (the entire running context), run
one forward pass through the model and return the score the model
assigns to each possible **next** token. `torch.no_grad()` disables
gradient tracking for this computation (inference only, no training).
`out.logits` has shape `(batch, sequence_length, vocab_size)`;
`[0, -1]` selects batch item 0 and the *last* position in the
sequence — i.e., "what comes after everything we've fed in so far."
`.tolist()` converts the tensor to a plain Python list so the rest of
the project never has to import or touch `torch` directly (recall:
the subject forbids importing `torch`/`transformers` in `src/`
directly — only through this SDK).

The three `get_path_to_*_file()` methods (vocab/merges/tokenizer file
download helpers) exist on the SDK but are **not used** by this
project: they're the tools you'd reach for to build the exact
full-vocabulary token-id → string table described in §1.3. This
project's top-`TOKEN_SEARCH_WIDTH` search (§1.4) sidesteps needing
that table by asking the model itself (via `decode`) what each
candidate token's text is, one candidate at a time, instead of
pre-building a table for all ~150,000 of them.

#### 3.3.2 — `CallEngine(model)` construction (`src/decoder.py`)

```python
class CallEngine:
    """Drives a `Small_LLM_Model` through constrained JSON generation."""

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
```

Nothing clever here — the engine just holds onto the model instance
it was given, so every other method can call `self.model.encode(...)`
etc. without needing it passed in again.

### 3.4 — Running the batch

Back in `main()`:

```python
    started = time.monotonic()
    results = run_batch(engine, functions, prompts)
    elapsed = time.monotonic() - started
    print(f"[*] Resolved {len(results)} prompt(s) in {elapsed:.1f}s.")
```

`time.monotonic()` is used instead of `time.time()` because it can
never jump backwards (e.g. due to a system clock adjustment) — it's
the correct tool for measuring elapsed duration. The real work happens
inside `run_batch`.

#### 3.4.1 — `run_batch()` (`src/pipeline.py`)

```python
def run_batch(
    engine: CallEngine,
    functions: list[FunctionSpec],
    prompts: list[PromptRequest],
) -> list[dict[str, Any]]:
    """Resolve every prompt in *prompts* into a structured function call."""
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
```

For every validated prompt, in order:

1. `engine.choose_function(...)` picks which `FunctionSpec` answers
   this request (§3.5).
2. `engine.fill_parameters(...)` decodes a typed value for every one
   of that function's declared parameters (§3.6).
3. A plain `dict` with exactly the three required keys — `prompt`,
   `name`, `parameters` — is appended to `results`. Note this dict is
   built here in Python, from Python values (a `str` and another
   `dict`), **not** by string-splicing the model's own text — that
   detail is what guarantees `json.dumps` (used later, in §3.7) can
   never fail or produce malformed output.

This function is called once from `main()` and returns the full list
of results after every prompt has been processed.

### 3.5 — Choosing a function: `choose_function()` (`src/decoder.py`)

```python
    def choose_function(
        self, user_prompt: str, functions: list[FunctionSpec]
    ) -> FunctionSpec:
        """Pick which function in *functions* answers *user_prompt*."""
        prompt = routing_prompt(user_prompt, functions)
        input_ids = list(self.model.encode(prompt)[0])
        names = [fn.name for fn in functions]
        chosen = self._decode_enum(input_ids, names)
        return next(fn for fn in functions if fn.name == chosen)
```

- `routing_prompt(user_prompt, functions)` builds the natural-language
  context (walked through in §3.5.1).
- `self.model.encode(prompt)[0]` — `encode` returns a 2-D tensor of
  shape `(1, sequence_length)` (batch size 1); `[0]` selects that one
  sequence, giving a 1-D tensor of token ids; `list(...)` converts it
  to a plain Python `list[int]` — the mutable, growable representation
  used for the rest of decoding.
- `names = [fn.name for fn in functions]` — the closed set of valid
  answers the enum decoder is allowed to converge on.
- `self._decode_enum(input_ids, names)` does the actual constrained
  generation (§3.5.2) and returns one of `names`, guaranteed.
- `next(fn for fn in functions if fn.name == chosen)` — a generator
  expression combined with `next()`: walks `functions` until it finds
  the one whose `.name` matches, and returns it immediately (short-
  circuiting, so it doesn't scan the rest of the list once found).
  This converts the plain string `chosen` back into the full
  `FunctionSpec` object (with its parameter types), which is what the
  rest of the pipeline needs next.

#### 3.5.1 — `routing_prompt()` (`src/prompting.py`)

```python
def routing_prompt(user_prompt: str, functions: list[FunctionSpec]) -> str:
    """Build the context used to pick which function answers *user_prompt*."""
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
```

- The generator expression builds one line per function, e.g.
  `- fn_add_numbers(a, b): Add two numbers together and return their
  sum.` — `', '.join(fn.parameters)` joins a `dict`'s keys (iterating
  a dict yields its keys) with `", "`.
  `"\n".join(...)` stacks all those lines into one multi-line string.
- The returned string ends with `'Chosen function name: "'` —
  deliberately left **open** with an unclosed quote. This is a
  standard prompting technique: by ending the prompt exactly where we
  want the model to continue, the model's very next tokens are forced
  to be a function name, without needing to "ask" for one and hope it
  complies.

#### 3.5.2 — `_decode_enum()` — the enum/prefix-trie decoder

```python
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
                self._append_text(input_ids, remaining[0][len(generated):])
                return remaining[0]
            input_ids.append(token_id)
            generated += self.model.decode([token_id])
        return choices[0]
```

This is used for both function-name selection (`choices` = every
function name) and boolean values (`choices = ["true", "false"]`).
Walking through it step by step:

- `generated = ""` — the text produced so far for this value; starts
  empty.
- `for _ in range(MAX_ENUM_STEPS):` — a hard cap (20 iterations) so a
  pathological case can never loop forever; the loop is expected to
  `return` from inside long before this cap is reached.
- `remaining = [c for c in choices if c.startswith(generated)]` — of
  all the possible final answers, which ones still *start with* what
  we've generated so far? Early on (`generated == ""`), every choice
  starts with `""`, so `remaining` is everything. As characters get
  appended, wrong choices drop out of `remaining` automatically.
- `if generated in remaining: return generated` — if what we've
  generated so far is *itself* one of the valid choices (not just a
  prefix of one — an exact match), we're done; return it immediately.
  This is what lets `"true"` finish right after its 4th character
  without waiting to see if the model tries to keep typing.
- `if len(remaining) == 1:` — if exactly one candidate is still
  possible, there's no ambiguity left to resolve — appending its
  remaining characters is a foregone conclusion. Rather than
  continuing to spend model calls confirming what we already know:
  - `self._append_text(input_ids, remaining[0][len(generated):])`
    — `remaining[0][len(generated):]` is a slice: the *tail* of the
    one remaining choice, starting right after what's already been
    generated (e.g. if `generated == "fn_gr"` and the only remaining
    choice is `"fn_greet"`, this slice is `"eet"`). `_append_text`
    (§3.5.4) encodes that tail and appends its token ids straight
    into `input_ids`, with no model call at all.
  - `return remaining[0]` — the finished choice.
- If we reach here, more than one choice is still possible and none
  of them exactly match yet — a real decision has to be made:
  - `def keeps_a_choice_open(piece: str) -> bool:` — a small nested
    function (a **closure**, meaning it can see `generated` and
    `remaining` from its enclosing scope) that answers: "if we
    appended this candidate token's text, would the result still be
    a valid prefix of at least one remaining choice?" `piece` is a
    candidate token's decoded text (potentially several characters,
    since tokens aren't always single characters).
  - `token_id = self._best_valid_token(input_ids, keeps_a_choice_open)`
    — asks the model for its best token *that satisfies
    `keeps_a_choice_open`* (full mechanism in §3.5.3). Returns `None`
    if nothing in the search window qualifies.
  - `if token_id is None:` — the deterministic-fallback case from
    §1.4: rather than trust an unchecked token, commit to the first
    remaining choice's tail exactly as in the `len(remaining) == 1`
    branch above, guaranteeing forward progress and a valid result.
  - Otherwise, `input_ids.append(token_id)` commits the chosen token
    to the running context — every future call to
    `get_logits_from_input_ids` will now see it — and
    `generated += self.model.decode([token_id])` updates our local
    tracking string to match.
- `return choices[0]` — only reachable if `MAX_ENUM_STEPS` iterations
  passed without ever converging (practically never, given how fast
  `remaining` narrows), as one last safety net.

#### 3.5.3 — `_best_valid_token()` — the core token filter

```python
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
```

This one function is shared by every decoding routine in the file —
it's the practical stand-in for "mask the whole vocabulary to -inf"
described in §1.4.

- `logits = self.model.get_logits_from_input_ids(input_ids)` — one
  forward pass through the model (§3.3.1), returning one score per
  vocabulary entry for "what comes next."
- `np.argsort(logits)` — NumPy's `argsort` returns the *indices* that
  would sort the array in **ascending** order (lowest logit first).
- `[::-1]` — reverses that, so it's now indices in **descending**
  order (highest logit first) — i.e., the model's ranked preferences,
  best first.
- `[:TOKEN_SEARCH_WIDTH]` — keeps only the top 12. This is the actual
  "search window" from §1.4: we never look past the model's 12 best
  guesses.
- `for token_id in ranked:` — walk them in order, best first.
- `piece = self.model.decode([int(token_id)])` — find out what text
  this specific candidate token actually represents (`int(...)`
  converts NumPy's integer type to a plain Python `int`, which
  `decode` — and `input_ids.append` later — expect).
- `if piece and is_valid(piece):` — `piece` must be non-empty (guards
  against a token decoding to `""`, which would trivially pass most
  validity checks without adding anything) *and* pass the
  caller-supplied validity rule.
- `return int(token_id)` — the first (i.e. highest-logit) candidate
  that qualifies wins immediately; ties in logit score never occur in
  practice, and even if they did, `argsort` breaks them consistently.
- `return None` — none of the top 12 satisfied `is_valid`; the caller
  is responsible for what happens next (every caller in this file
  falls back to a deterministic choice, per §1.4).

`is_valid` is a `Callable[[str], bool]` — a function that takes a
`str` and returns a `bool`. Every call site in this file passes a
*different* one in (`keeps_a_choice_open` above; charset checks for
numbers and strings, coming up in §3.6.4–3.6.5) — this is what lets
one shared token-selection routine serve every leaf type without
duplicating the "ask the model, rank, filter" logic four times.

#### 3.5.4 — `_append_text()`

```python
    def _append_text(self, input_ids: list[int], text: str) -> None:
        """Encode *text* and append it straight to the running context,
        for the parts of the output this module authors deterministically
        instead of asking the model to generate them."""
        if text:
            input_ids.extend(list(self.model.encode(text)[0]))
```

A small helper: turn a known string into token ids and extend
`input_ids` with them, in place. `if text:` guards against calling
`encode("")`, which would just be wasted work. This is the same
"inject known text as tokens" pattern used for JSON punctuation in
`fill_parameters` (next), reused here for the tail of an already-
determined enum choice.

At the end of §3.5, `choose_function` returns a `FunctionSpec` — back
in `run_batch` (§3.4.1), this becomes the `function` variable, about
to be handed to `fill_parameters`.

### 3.6 — Filling in the arguments: `fill_parameters()` (`src/decoder.py`)

```python
    def fill_parameters(
        self, user_prompt: str, function: FunctionSpec
    ) -> dict[str, Any]:
        """Decode a typed value for every parameter of *function*."""
        prompt = extraction_prompt(user_prompt, function)
        input_ids = list(self.model.encode(prompt)[0])
        values: dict[str, Any] = {}
        for index, (key, spec) in enumerate(function.parameters.items()):
            opening = f'"{key}": ' if index == 0 else f', "{key}": '
            input_ids.extend(list(self.model.encode(opening)[0]))
            values[key] = self._decode_value(input_ids, spec.type)
        return values
```

- `extraction_prompt(user_prompt, function)` builds a *fresh* prompt
  string (§3.6.1) — note this starts a brand-new `input_ids` context,
  separate from the one built in `choose_function`; the two stages
  don't share token history.
- `values: dict[str, Any] = {}` — will accumulate `{key: decoded
  value}` pairs, in the same order the function declares them.
- `for index, (key, spec) in enumerate(function.parameters.items()):`
  — `function.parameters` is a `dict[str, ParameterSpec]`; `.items()`
  yields `(key, spec)` pairs; `enumerate(...)` additionally numbers
  them from 0, so the loop knows whether it's on the *first*
  parameter or a later one.
- `opening = f'"{key}": ' if index == 0 else f', "{key}": '` — this is
  where the JSON structure is authored by hand: the first parameter
  gets `"key": ` (no leading comma, since `extraction_prompt` already
  left the context ending in `{"parameters": {`); every subsequent
  parameter gets `, "key": ` (with a leading comma to separate it from
  the previous one). The model never decides where a comma goes.
- `input_ids.extend(list(self.model.encode(opening)[0]))` — encodes
  that known text and appends it to the running context, exactly like
  `_append_text` does (this one is inlined rather than routed through
  the helper simply because it's a one-off, not reused elsewhere in
  this method).
- `values[key] = self._decode_value(input_ids, spec.type)` — decodes
  one typed leaf value (next) and stores it under its key.
- After the loop, `values` — e.g. `{"a": 2.0, "b": 3.0}` — is
  returned to `run_batch`.

#### 3.6.1 — `extraction_prompt()` (`src/prompting.py`)

```python
def extraction_prompt(user_prompt: str, function: FunctionSpec) -> str:
    """Build the context used to fill in *function*'s argument values."""
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
```

Very similar in spirit to `routing_prompt`: `fields` lists each
parameter's name and expected type (e.g. `"a (number), b (number)"`)
so the model has a hint about what kind of value belongs where, and
the returned string ends mid-JSON, at `'{"parameters": {'`, exactly
where `fill_parameters`'s loop is about to keep appending `"key":
value` pairs.

#### 3.6.2 — `_decode_value()` — dispatch by type

```python
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
```

A simple dispatcher, one branch per allowed `ParameterType`:

- **`"string"`** — first, an opening `"` is injected directly (again,
  structure written by us, not generated), *then* `_decode_string`
  (§3.6.4) fills in the characters between the quotes.
- **`"boolean"`** — reuses the exact same `_decode_enum` machinery
  from §3.5.2, just with `["true", "false"]` as the closed set of
  choices instead of function names, and converts the resulting
  string to a real Python `bool` with `picked == "true"`.
- **`"integer"`** — calls the shared number decoder (§3.6.5) with
  `allow_decimal=False` (so a stray `.` can never even be considered
  valid mid-generation), then wraps the result in `int(...)` for good
  measure.
- **`"number"`** (the default / fallthrough) — same decoder, with
  `allow_decimal=True`, returned as a `float` directly.

#### 3.6.3 — `MAX_VALUE_TOKENS` and the stop-character constants

Just below the imports, three module-level constants configure the
string/number decoders:

```python
MAX_VALUE_TOKENS = 24
STRING_STOP_CHARACTERS = '"'
NUMBER_STOP_CHARACTERS = ',} \n"'
```

`MAX_VALUE_TOKENS` bounds how many model calls a single leaf value can
cost, as a safety net against runaway generation. The two
`STOP_CHARACTERS` strings are used as membership tests (`char in
stop_chars`) by `_peek_stop` (§3.6.6) to recognize "the model is
signalling it wants to end this value here."

#### 3.6.4 — `_decode_string()`

```python
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
```

- `text = ""` — the raw (unescaped, unquoted) content of the string
  value being built.
- Each iteration first **peeks** (§3.6.6): "if I let the model choose
  freely right now, would it pick something that looks like the end
  of this string (a `"`)?" If yes, `break` immediately — the string is
  done, and critically, the closing-quote token itself is *never*
  appended to `text` (we only asked what the model *would* pick; we
  didn't commit it) — the final `"` of the JSON string is written
  later, implicitly, by `json.dumps` (§3.7) wrapping `text` back into
  a proper JSON string.
- If not stopping yet, `_best_valid_token` is asked for the model's
  best token whose decoded text contains **no literal `"` and no
  newline** — those are the two characters that would otherwise break
  the surrounding JSON string if they slipped through raw. (Any other
  character — including a lone backslash — is allowed through raw;
  proper JSON escaping of the *final* string is entirely delegated to
  `json.dumps` later, which is why this project never has to hand-
  implement escape-sequence handling the way the two peer
  implementations this project drew inspiration from did.)
- If nothing qualifies, `break` (safety net, rarely hit — free-form
  text has a huge valid alphabet).
- Otherwise commit the token to `input_ids` and append its text to
  `text`.
- `return text.strip()` — trims any incidental leading/trailing
  whitespace the model produced (e.g. a leading space token right
  after the opening quote) before handing the clean value back.

#### 3.6.5 — `_decode_number()`

```python
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
```

- `pattern = re.compile(...)` — a compiled regular expression
  defining the entire valid alphabet for this value, chosen once up
  front based on `allow_decimal`:
  - `r"-?\d*\.?\d*"` (numbers): an optional leading `-`, then any
    digits, then an optional single `.`, then any more digits — this
    shape can never contain two decimal points or a decimal point
    before a sign, because `pattern.fullmatch` is checked against the
    *entire* accumulated text every time (not just the newest piece),
    so any candidate that would create a second `.` simply fails to
    match and is rejected.
  - `r"-?\d*"` (integers): the same, minus the decimal-point clause
    entirely — a `.` character can *never* pass validation for an
    `integer`-typed parameter, at the token level, not as a post-hoc
    cleanup.
- `if text and self._peek_stop(...)`: — `text and ...` means the
  stop-peek is only performed once at least one digit has actually
  been generated (an empty number isn't something we want to stop
  before starting on).
- The loop body otherwise mirrors `_decode_string`: filter candidates
  by `pattern.fullmatch(text + piece)` (does appending this token's
  text keep the whole value matching the numeric pattern?), commit the
  first one that passes, or fall back to `break` if none do.
- `return float(text)` — parse the accumulated digit string into an
  actual Python `float`. `_decode_value` (§3.6.2) additionally wraps
  this in `int(...)` for `"integer"`-typed parameters.
- `except ValueError: return 0.0` — a defensive fallback for the
  edge case where `text` never accumulated anything meaningful (e.g.
  `""` or just `"-"`, neither of which `float()` accepts) — this
  guarantees `_decode_number` always returns a valid Python number,
  never raises, keeping the promise that the final output object's
  `parameters` values always match their declared JSON types.

#### 3.6.6 — `_peek_stop()`

```python
    def _peek_stop(self, input_ids: list[int], stop_chars: str) -> bool:
        """Look at the model's single best next token, without
        committing it, to decide whether it is trying to end the
        current value."""
        logits = self.model.get_logits_from_input_ids(input_ids)
        piece = self.model.decode([int(np.argmax(logits))])
        return not piece or piece[0] in stop_chars
```

- `logits = self.model.get_logits_from_input_ids(input_ids)` — same
  forward pass as always, but note: **`input_ids` is not modified
  here.** This is a read-only look-ahead — "if a token had to be
  chosen right now, what would the single best one be?" — without
  actually choosing it.
- `np.argmax(logits)` — the index of the single highest logit (no
  ranking of several candidates needed here, unlike
  `_best_valid_token`; we only care about the model's *top* pick).
- `piece = self.model.decode([...])` — that token's text.
- `return not piece or piece[0] in stop_chars` — true if the token
  decoded to nothing at all, *or* if its first character is one of
  the caller's stop characters (a `"` for strings; `,`, `}`,
  whitespace, or `"` for numbers). This is the heuristic that answers
  "does the model think this value is finished?" — a real signal from
  the model's own preferences, just inspected before being acted on.

At the end of §3.6, `fill_parameters` returns to `run_batch`, which
(§3.4.1) assembles the final `{"prompt": ..., "name": ...,
"parameters": ...}` dict and appends it to `results`. The `for
request in prompts:` loop then repeats §3.5–3.6 for every remaining
prompt.

### 3.7 — Back in `main()`: writing the output file

Once every prompt in the batch has been resolved and `run_batch`
returns, control is back in `main()` (§3.4):

```python
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[*] Results written to '{args.output}'.")
```

- `os.path.dirname(args.output)` — the directory portion of the
  output path (e.g. `"data/output"` from
  `"data/output/function_calling_results.json"`); this is `""` if the
  user passed a bare filename with no directory component.
- `if output_dir: os.makedirs(output_dir, exist_ok=True)` — creates
  the output directory (and any missing parent directories) if it
  doesn't already exist; `exist_ok=True` means this doesn't raise if
  the directory is already there. The `if output_dir:` guard avoids
  calling `os.makedirs("")`, which would error.
- `with open(args.output, "w", ...) as handle:` — again a context
  manager, guaranteeing the file is closed (and its contents flushed
  to disk) even if something goes wrong mid-write.
- `json.dump(results, handle, indent=2)` — this is the line that
  actually turns the list of Python `dict`s built in §3.4.1 into the
  file's JSON text, with 2-space indentation for readability. Because
  `results` is a plain nested structure of `dict`/`list`/`str`/
  `float`/`int`/`bool` (never raw model text spliced in), this call
  cannot fail on escaping or structure — every guarantee this project
  makes about "100% valid JSON" cashes out at this exact line.
- A final progress message confirms where the file was written.

`main()` then returns (implicitly, falling off the end of the
function), back to the `if __name__ == "__main__":` block from §3.0,
which then also falls off the end — the program exits normally, code
0.

### 3.8 — What if something goes wrong mid-run?

Two things can still interrupt a run after the model has loaded and
generation is underway: a `KeyboardInterrupt` (the user pressing
`Ctrl+C`) or literally any other unexpected exception. Both are caught
by the `try/except` wrapped around `main()` in §3.0, and both result
in a clean one-line message instead of a raw Python traceback —
satisfying the subject's "must never crash unexpectedly" rule for the
entire run, not just the file-loading stage covered in §3.2.

---

## Part 4 — A worked example, start to finish

Trace the prompt `"Repeat the word 'ha' 3 times"` against the sample
`fn_repeat_string(s: string, times: integer)` function from
`data/input/functions_definition.json`:

1. **Routing.** `routing_prompt` lists every function; the prompt ends
   at `Chosen function name: "`. `_decode_enum` starts with
   `generated = ""` and `remaining` = all six sample function names.
   As the model's top tokens are checked against
   `keeps_a_choice_open`, `remaining` narrows — `"fn_r"` still matches
   both `fn_reverse_string` and `fn_repeat_string`; the moment a token
   makes it `"fn_re"` vs `"fn_rep"`, only one candidate survives, and
   `_decode_enum` finishes the rest of `"fn_repeat_string"` itself,
   with zero further model calls.
2. **Extraction setup.** `extraction_prompt` builds a fresh context:
   `Arguments to fill: s (string), times (integer)`, ending at
   `{"parameters": {`.
3. **First parameter, `s` (string, index 0).** `fill_parameters`
   injects `"s": `, then `_decode_value` injects the opening `"` and
   calls `_decode_string`. Token by token the model produces `h`,
   `a`; before a third token, `_peek_stop` notices the model's top
   choice now looks like a closing `"`, so generation stops. `text =
   "ha"`.
4. **Second parameter, `times` (integer, index 1).**
   `fill_parameters` injects `, "times": `. `_decode_value` calls
   `_decode_number(allow_decimal=False)`. The model produces `3`;
   `_peek_stop` sees its top choice next is `}` (a `NUMBER_STOP_
   CHARACTERS` member), so generation stops with `text = "3"`.
   `int(float("3"))` → `3`.
5. **Assembly.** `run_batch` builds
   `{"prompt": "Repeat the word 'ha' 3 times", "name":
   "fn_repeat_string", "parameters": {"s": "ha", "times": 3}}` as a
   plain dict.
6. **Serialization.** `json.dump` writes it out — no escaping
   decisions were ever left to chance, because nothing about steps
   3–5 depended on the model producing valid JSON syntax; it only ever
   had to produce the *characters between* the quotes.

---

## Part 5 — Design Q&A

**Q: Why only check the top 12 tokens instead of the full vocabulary?**
Building an exact token-id → string table for Qwen's ~150,000-entry
BPE vocabulary (via `get_path_to_vocab_file`) and re-validating all of
them every step is the textbook-correct approach, but is expensive and
easy to get subtly wrong around BPE's space/punctuation encoding.
Checking the model's own top 12 preferences achieves the same
practical reliability for the small, simple valid-alphabets involved
here (digits, known function-name characters, `true`/`false`), because
a competent language model already ranks plausible characters highly
— we rarely need to look past its top few guesses to find one that
also happens to be structurally valid.

**Q: What actually guarantees the JSON is valid, then, if only the top
12 tokens are checked?** Two independent things, not one: (1) every
brace, key, colon, and comma is written by this code, never sampled —
so *structure* was never at risk regardless of search width; and (2)
every leaf decoder has a **deterministic fallback** for the rare case
where none of the top 12 candidates qualify — it never falls through
to trusting an unchecked token. Between the two, there is no code path
that can emit a syntactically invalid character.

**Q: Why decode two separate stages (`choose_function` then
`fill_parameters`) instead of one combined generation pass?**
Separating "which function" from "what are its arguments" means the
second stage's prompt can be built with full knowledge of exactly one
function's parameter names and types — a more focused context than
trying to describe an entire catalog and its arguments all at once.
It also means a completely fresh token context can start once a
function is known, rather than carrying the (irrelevant, once decided)
full function catalog forward into argument extraction.

**Q: Why build the final `dict` in Python and call `json.dumps` at the
end, instead of asking the model to produce properly escaped JSON
text directly?** Because escaping is a solved, mechanical problem that
Python's own `json` module already does correctly and does not need a
language model's help with. Trying to keep a model's raw token stream
valid JSON while it's still generating (handling stray backslashes,
embedded quotes, etc.) is exactly the class of bug this project's
README calls out in the two peer implementations that inspired it.

**Q: What happens if a prompt is genuinely ambiguous or doesn't match
any function well?** `choose_function` still always returns a real
`FunctionSpec` — the enum decoder is defined over the closed set of
real function names, so there is no way for it to output anything
else. The subject's output-schema requirement (exactly `prompt`,
`name`, `parameters`, on every object) is met unconditionally; whether
the *chosen* function is the semantically best one for an ambiguous
prompt is a separate, accuracy question rather than a structural one
(see the "Performance Analysis" section of `README.md`).
