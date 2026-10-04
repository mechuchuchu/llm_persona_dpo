#!/usr/bin/env python3
"""Balance system prompts over generated pairs and write preference parquet."""

import argparse
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
                        help="Parquet with user, chosen, and rejected columns")
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


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.pairs, columns=["user", "chosen", "rejected"])
    pairs = table.to_pylist()
    if not pairs:
        raise ValueError(f"No preference pairs found in {args.pairs}")
    if any(not row["user"] or row["chosen"] is None or row["rejected"] is None
           for row in pairs):
        raise ValueError("Input pairs contain an empty user prompt or a missing response.")

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
