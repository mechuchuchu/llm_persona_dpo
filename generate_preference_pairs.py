#!/usr/bin/env python3
"""Generate user/chosen/rejected rows using separate prompts for both models."""

import argparse
import gc
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DEFAULT_USERS = ROOT / "data" / "first_prompt.txt"
DEFAULT_CHOSEN_MODEL = ROOT / "models" / "Qwen3.5-2B"
DEFAULT_REJECTED_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_CHOSEN_SYSTEM = ROOT / "prompts" / "qwen3_5_2b_system.txt"
DEFAULT_REJECTED_SYSTEM = ROOT / "prompts" / "qwen3_0_6b_system.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=Path, default=DEFAULT_USERS,
                        help=".txt prompt, .jsonl with user/prompt fields, or parquet with a user/prompt column")
    parser.add_argument("--chosen-system-prompt", type=Path,
                        default=DEFAULT_CHOSEN_SYSTEM,
                        help="System prompt used only to generate the chosen model's response")
    parser.add_argument("--rejected-system-prompt", type=Path,
                        default=DEFAULT_REJECTED_SYSTEM,
                        help="System prompt used only to generate the rejected model's response")
    parser.add_argument("--chosen-model", default=str(DEFAULT_CHOSEN_MODEL),
                        help="Chosen model path or Hugging Face model ID (default: local Qwen3.5-2B)")
    parser.add_argument("--rejected-model", default=DEFAULT_REJECTED_MODEL,
                        help="Rejected model path or Hugging Face model ID (default: Qwen/Qwen3-0.6B)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data" / "user_answer_pairs.parquet")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def _get_user_value(record: dict, line_number: int) -> str:
    value = record.get("user", record.get("prompt"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prompt row {line_number} has no non-empty 'user' or 'prompt' string.")
    return value.strip()


def read_user_prompts(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Prompt file is empty: {path}")
        return [content]

    if suffix == ".jsonl":
        prompts = []
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if line.strip():
                    prompts.append(_get_user_value(json.loads(line), line_number))
        if not prompts:
            raise ValueError(f"No prompts found in {path}")
        return prompts

    if suffix == ".parquet":
        table = pq.read_table(path)
        column_name = "user" if "user" in table.column_names else "prompt"
        if column_name not in table.column_names:
            raise ValueError(f"{path} must contain a 'user' or 'prompt' column.")
        prompts = [_get_user_value({column_name: value}, i + 1)
                   for i, value in enumerate(table[column_name].to_pylist())]
        if not prompts:
            raise ValueError(f"No prompts found in {path}")
        return prompts

    raise ValueError("--users must be a .txt, .jsonl, or .parquet file.")


def generate_for_model(model: str, users: list[str], system_prompt: str,
                       args: argparse.Namespace) -> list[str]:
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=model,
        dtype="auto",
        trust_remote_code=False,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.batch_size,
    )
    responses: list[str] = []
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
    )
    try:
        for start in range(0, len(users), args.batch_size):
            batch = users[start:start + args.batch_size]
            conversations = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                for user_prompt in batch
            ]
            outputs = engine.chat(
                conversations,
                sampling,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"{model} returned {len(outputs)} outputs for a batch of {len(batch)} prompts."
                )
            responses.extend(output.outputs[0].text for output in outputs)
    finally:
        del engine
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    return responses


def main() -> None:
    args = parse_args()
    users = read_user_prompts(args.users)
    chosen_system = args.chosen_system_prompt.read_text(encoding="utf-8").strip()
    rejected_system = args.rejected_system_prompt.read_text(encoding="utf-8").strip()
    if not chosen_system:
        raise ValueError(f"Chosen system prompt is empty: {args.chosen_system_prompt}")
    if not rejected_system:
        raise ValueError(f"Rejected system prompt is empty: {args.rejected_system_prompt}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    # Each model gets its own system prompt; both receive the same ordered users.
    print(f"Generating chosen responses with {args.chosen_model} ...", flush=True)
    chosen = generate_for_model(args.chosen_model, users, chosen_system, args)
    print(f"Generating rejected responses with {args.rejected_model} ...", flush=True)
    rejected = generate_for_model(args.rejected_model, users, rejected_system, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "user": pa.array(users, type=pa.string()),
        "chosen": pa.array(chosen, type=pa.string()),
        "rejected": pa.array(rejected, type=pa.string()),
    })
    pq.write_table(table, args.output, compression="zstd")
    print(f"Saved {len(users)} user/chosen/rejected rows to {args.output}")


if __name__ == "__main__":
    main()
