"""CLI entry point: read prompts, resolve function calls, write results."""

import argparse
import json
import os
import time

from pydantic import ValidationError

from llm_sdk import Small_LLM_Model

from .dataset import load_dataset
from .decoder import CallEngine
from .pipeline import run_batch


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


def main() -> None:
    """Run the full pipeline: load, decode, and persist the results."""
    args = parse_args()

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

    print(f"[*] Loaded {len(functions)} function(s) and "
          f"{len(prompts)} prompt(s).")
    print("[*] Loading model, this may take a moment...")

    model = Small_LLM_Model()
    engine = CallEngine(model)

    started = time.monotonic()
    results = run_batch(engine, functions, prompts)
    elapsed = time.monotonic() - started
    print(f"[*] Resolved {len(results)} prompt(s) in {elapsed:.1f}s.")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"[*] Results written to '{args.output}'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted, exiting.")
    except Exception as error:  # noqa: BLE001 - top-level safety net
        print(f"[!] Unexpected error: {error}")
