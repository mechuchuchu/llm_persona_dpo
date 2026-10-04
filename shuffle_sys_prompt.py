#!/usr/bin/env python3
"""Balance system prompts over generated pairs and write preference parquet."""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DEFAULT_PAIRS = ROOT / "data" / "user_answer_pairs.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "shuffle_sys_prompt.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                        help="Parquet or JSONL with user, chosen, and rejected fields")
    parser.add_argument("--system-prompts-dir", type=Path, required=True,
                        help="Directory containing system prompt .txt files (searched recursively)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_system_prompts(directory: Path) -> list[tuple[str, str]]:
    files = sorted(path for path in directory.rglob("*.txt") if path.is_file())
    prompts = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            prompts.append((path.relative_to(directory).as_posix(), text))
    if not prompts:
        raise ValueError(f"No non-empty .txt system prompts found under {directory}")
    return prompts


def read_pairs(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        pairs = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}: {error}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Line {line_number} of {path} must be a JSON object.")
                pairs.append(row)
        return pairs

    if path.suffix.lower() == ".parquet":
        return pq.read_table(path, columns=["user", "chosen", "rejected"]).to_pylist()

    raise ValueError(f"Pairs input must be a .parquet or .jsonl file: {path}")


def main() -> None:
    args = parse_args()
    pairs = read_pairs(args.pairs)
    if not pairs:
        raise ValueError(f"No preference pairs found in {args.pairs}")
    required_fields = ("user", "chosen", "rejected")
    for row_number, row in enumerate(pairs, start=1):
        if any(not isinstance(row.get(field), str) for field in required_fields):
            raise ValueError(
                f"Preference pair {row_number} must contain string fields: "
                "'user', 'chosen', and 'rejected'."
            )
        if not row["user"].strip():
            raise ValueError(f"Preference pair {row_number} has an empty 'user' prompt.")

    prompts = read_system_prompts(args.system_prompts_dir)
    rows = []
    # Replicating every pair once per prompt gives each prompt exactly one row
    # per source pair, even when the source row count is not divisible by the
    # number of system prompts. Each row has only one shared system/user prefix.
    for pair in pairs:
        for prompt_file, system_prompt in prompts:
            rows.append({
                "system": system_prompt,
                "user": pair["user"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
                "system_prompt_file": prompt_file,
            })

    random.Random(args.seed).shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = pa.Table.from_pylist(rows, schema=pa.schema([
        ("system", pa.string()),
        ("user", pa.string()),
        ("chosen", pa.string()),
        ("rejected", pa.string()),
        ("system_prompt_file", pa.string()),
    ]))
    pq.write_table(output, args.output, compression="zstd")
    print(
        f"Saved {len(rows)} rows ({len(pairs)} source pairs × {len(prompts)} system prompts) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
