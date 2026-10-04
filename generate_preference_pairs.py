#!/usr/bin/env python3
"""Generate user/chosen/rejected rows using separate prompts for both models."""

import argparse
import gc
import json
from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DEFAULT_USERS = ROOT / "data" / "first_prompt.txt"
DEFAULT_DATASET = "allenai/Dolci-Instruct-SFT-No-Tools"
DEFAULT_CHOSEN_MODEL = "Qwen/Qwen3.5-2B"    #ROOT / "models" / "Qwen3.5-2B"
DEFAULT_REJECTED_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_CHOSEN_SYSTEM = ROOT / "generation_prompts" / "chosen_system.txt"
DEFAULT_REJECTED_SYSTEM = ROOT / "generation_prompts" / "rejected_system.txt"
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
SYSTEM_TRUNCATION_MARKER = "[Middle of the system prompt omitted to fit the model context.]"
USER_TRUNCATION_MARKER = "[Earlier part of the user prompt omitted to fit the model context.]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--users", type=Path,
                        help=".txt prompt, .jsonl with user/prompt fields, or parquet with a user/prompt column")
    source.add_argument("--dataset", default=None,
                        help=f"HF dataset with a messages field (example: {DEFAULT_DATASET})")
    parser.add_argument("--split", default="train", help="Dataset split used with --dataset")
    parser.add_argument("--num-prompts", type=int, default=None,
                        help="Number of user prompts to process; required with --dataset")
    parser.add_argument("--dataset-shuffle-buffer", type=int, default=10000,
                        help="Streaming shuffle buffer size when reading --dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for streaming dataset shuffle")
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
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file; defaults to data/user_answer_pairs with the selected format suffix")
    parser.add_argument("--output-format", choices=("parquet", "jsonl"), default=None,
                        help="Output format (default: infer from .jsonl suffix, otherwise parquet)")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def resolve_output(args: argparse.Namespace) -> tuple[Path, str]:
    output_format = args.output_format
    if output_format is None:
        output_format = (
            "jsonl"
            if args.output and args.output.suffix.lower() == ".jsonl"
            else "parquet"
        )

    output = args.output
    if output is None:
        suffix = ".jsonl" if output_format == "jsonl" else ".parquet"
        output = ROOT / "data" / f"user_answer_pairs{suffix}"
    else:
        suffix_formats = {".jsonl": "jsonl", ".parquet": "parquet"}
        suffix_format = suffix_formats.get(output.suffix.lower())
        if suffix_format is not None and suffix_format != output_format:
            raise ValueError(
                f"Output path {output} has a {suffix_format} suffix, but "
                f"--output-format is {output_format}."
            )

    return output, output_format


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


def read_dataset_prompts(dataset_id: str, split: str, count: int,
                         shuffle_buffer: int, seed: int) -> list[str]:
    from datasets import load_dataset

    stream = load_dataset(dataset_id, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=shuffle_buffer)
    prompts = []
    for row in stream:
        first_user = next(
            (
                message.get("content", "")
                for message in row.get("messages", [])
                if message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ),
            None,
        )
        if first_user is not None:
            prompts.append(first_user.strip())
        if len(prompts) == count:
            break
    if len(prompts) != count:
        raise ValueError(
            f"Requested {count} prompts from {dataset_id}/{split}, but found only {len(prompts)}."
        )
    return prompts


def _truncate_tokens(tokenizer, token_ids: list[int], keep: int, original: str,
                     marker: str, keep_ends: bool) -> str:
    if keep >= len(token_ids):
        return original
    if keep <= 0:
        return marker
    if keep_ends:
        prefix_count = (keep + 1) // 2
        suffix_count = keep // 2
        prefix = tokenizer.decode(token_ids[:prefix_count], skip_special_tokens=False)
        suffix = (
            tokenizer.decode(token_ids[-suffix_count:], skip_special_tokens=False)
            if suffix_count else ""
        )
        return f"{prefix}\n{marker}\n{suffix}"
    suffix = tokenizer.decode(token_ids[-keep:], skip_special_tokens=False)
    return f"{marker}\n{suffix}"


def _fit_chat_context(tokenizer, system_prompt: str, user_prompt: str,
                      prompt_token_budget: int) -> tuple[list[dict[str, str]], bool, bool]:
    system_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    user_tokens = tokenizer.encode(user_prompt, add_special_tokens=False)

    def make_messages(system_keep: int, user_keep: int) -> list[dict[str, str]]:
        fitted_system = _truncate_tokens(
            tokenizer, system_tokens, system_keep, system_prompt,
            SYSTEM_TRUNCATION_MARKER, keep_ends=True
        )
        fitted_user = _truncate_tokens(
            tokenizer, user_tokens, user_keep, user_prompt,
            USER_TRUNCATION_MARKER, keep_ends=False
        )
        return [
            {"role": "system", "content": fitted_system},
            {"role": "user", "content": fitted_user},
        ]

    def token_count(messages: list[dict[str, str]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        token_ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
        return len(token_ids)

    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if token_count(full_messages) <= prompt_token_budget:
        return full_messages, False, False

    # Keep a useful tail of the user prompt even when the system prompt itself
    # needs shortening. For shorter prompts, keep the complete user message.
    minimum_user_tokens = min(128, len(user_tokens))
    if token_count(make_messages(0, minimum_user_tokens)) > prompt_token_budget:
        raise ValueError(
            "The model context is too small for the chat template and the minimum "
            "system/user prompt markers. Increase --max-model-len or reduce "
            "--max-new-tokens."
        )

    # First preserve the full system prompt and trim the user prompt. If even a
    # minimal user message does not fit, shorten the system prompt from its
    # middle while retaining its beginning and ending instructions.
    if token_count(make_messages(len(system_tokens), minimum_user_tokens)) <= prompt_token_budget:
        system_keep = len(system_tokens)
    else:
        low, high = 0, len(system_tokens)
        system_keep = 0
        while low <= high:
            candidate = (low + high) // 2
            if token_count(make_messages(candidate, minimum_user_tokens)) <= prompt_token_budget:
                system_keep = candidate
                low = candidate + 1
            else:
                high = candidate - 1
        while token_count(make_messages(system_keep, minimum_user_tokens)) > prompt_token_budget:
            system_keep -= 1

    low, high = minimum_user_tokens, len(user_tokens)
    user_keep = minimum_user_tokens
    while low <= high:
        candidate = (low + high) // 2
        if token_count(make_messages(system_keep, candidate)) <= prompt_token_budget:
            user_keep = candidate
            low = candidate + 1
        else:
            high = candidate - 1
    messages = make_messages(system_keep, user_keep)
    while token_count(messages) > prompt_token_budget and user_keep > minimum_user_tokens:
        user_keep -= 1
        messages = make_messages(system_keep, user_keep)
    if token_count(messages) > prompt_token_budget:
        raise RuntimeError("Could not fit a prompt within the configured model context length.")

    return messages, user_keep < len(user_tokens), system_keep < len(system_tokens)


def generate_for_model(model: str, users: list[str], system_prompt: str,
                       args: argparse.Namespace) -> list[str]:
    from vllm import LLM, SamplingParams
    from tqdm.auto import tqdm

    prompt_token_budget = args.max_model_len - args.max_new_tokens
    if prompt_token_budget < 1:
        raise ValueError(
            "--max-model-len must be greater than --max-new-tokens so the prompt "
            "and generated response can fit in the model context."
        )

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
    progress = None
    truncated_user_count = 0
    truncated_system_count = 0
    try:
        tokenizer = engine.get_tokenizer()
        progress = tqdm(
            total=len(users),
            desc=f"Generating {model.split('/')[-1]}",
            unit="prompt",
        )
        for start in range(0, len(users), args.batch_size):
            batch = users[start:start + args.batch_size]
            conversations = []
            for user_prompt in batch:
                messages, user_truncated, system_truncated = _fit_chat_context(
                    tokenizer, system_prompt, user_prompt, prompt_token_budget
                )
                conversations.append(messages)
                truncated_user_count += user_truncated
                truncated_system_count += system_truncated
            outputs = engine.chat(
                conversations,
                sampling,
                use_tqdm=False,
                chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
            )
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"{model} returned {len(outputs)} outputs for a batch of {len(batch)} prompts."
                )
            responses.extend(output.outputs[0].text for output in outputs)
            progress.update(len(batch))
    finally:
        if progress is not None:
            progress.close()
        del engine
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    if truncated_user_count or truncated_system_count:
        print(
            f"{model}: shortened prompts for context length in "
            f"{truncated_user_count} user row(s) and "
            f"{truncated_system_count} system prompt row(s).",
            flush=True,
        )
    return responses


def main() -> None:
    args = parse_args()
    if args.num_prompts is not None and args.num_prompts < 1:
        raise ValueError("--num-prompts must be at least 1.")
    if args.dataset:
        if args.num_prompts is None:
            raise ValueError("--num-prompts is required when using --dataset.")
        if args.dataset_shuffle_buffer < 1:
            raise ValueError("--dataset-shuffle-buffer must be at least 1.")
        users = read_dataset_prompts(
            args.dataset,
            args.split,
            args.num_prompts,
            args.dataset_shuffle_buffer,
            args.seed,
        )
    else:
        users = read_user_prompts(args.users or DEFAULT_USERS)
        if args.num_prompts is not None:
            if args.num_prompts > len(users):
                raise ValueError(
                    f"Requested {args.num_prompts} prompts, but {args.users or DEFAULT_USERS} "
                    f"contains only {len(users)}."
                )
            users = users[:args.num_prompts]
    chosen_system = args.chosen_system_prompt.read_text(encoding="utf-8").strip()
    rejected_system = args.rejected_system_prompt.read_text(encoding="utf-8").strip()
    if not chosen_system:
        raise ValueError(f"Chosen system prompt is empty: {args.chosen_system_prompt}")
    if not rejected_system:
        raise ValueError(f"Rejected system prompt is empty: {args.rejected_system_prompt}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    output, output_format = resolve_output(args)

    print(f"Generating {len(users)} response pairs with batch size {args.batch_size}.", flush=True)
    # Each model gets its own system prompt; both receive the same ordered users.
    print(f"Generating chosen responses with {args.chosen_model} ...", flush=True)
    chosen = generate_for_model(args.chosen_model, users, chosen_system, args)
    print(f"Generating rejected responses with {args.rejected_model} ...", flush=True)
    rejected = generate_for_model(args.rejected_model, users, rejected_system, args)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "jsonl":
        with output.open("w", encoding="utf-8") as destination:
            for user, chosen_response, rejected_response in zip(users, chosen, rejected):
                json.dump(
                    {
                        "user": user,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                    },
                    destination,
                    ensure_ascii=False,
                )
                destination.write("\n")
    else:
        table = pa.table({
            "user": pa.array(users, type=pa.string()),
            "chosen": pa.array(chosen, type=pa.string()),
            "rejected": pa.array(rejected, type=pa.string()),
        })
        pq.write_table(table, output, compression="zstd")
    print(f"Saved {len(users)} user/chosen/rejected rows to {output}")


if __name__ == "__main__":
    main()
