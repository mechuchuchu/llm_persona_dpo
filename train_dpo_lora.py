#!/usr/bin/env python3
"""DPO fine-tuning with a LoRA adapter from user/chosen/rejected JSONL pairs."""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3.5-4B")
    parser.add_argument("--data", default="data/custom_answer_pairs_9b.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen3.5-4b-dpo-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_pairs(path: Path, tokenizer):
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            for field in ("user", "chosen", "rejected"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"Line {line_number} needs a non-empty string '{field}'")
            # Qwen3.5's default template puts an unfinished <think> block in a
            # generation prompt, but closes that block before a supplied answer.
            # Render an empty assistant turn, remove its end marker, and keep the
            # preference completions as strings so both tokenize from one exact
            # prefix without TRL re-templating the conversations independently.
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": row["user"]},
                    {"role": "assistant", "content": ""},
                ],
                tokenize=False,
            )
            end_marker = f"{tokenizer.eos_token}\n"
            if not prompt.endswith(end_marker):
                raise ValueError("Unexpected Qwen chat template ending; cannot form an exact DPO prefix")
            prompt = prompt[: -len(end_marker)]
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": row["chosen"] + end_marker,
                    "rejected": row["rejected"] + end_marker,
                }
            )
    if not rows:
        raise ValueError(f"No preference pairs found in {path}")
    return rows


def main():
    args = parse_args()
    set_seed(args.seed)

    model_path = Path(args.model).resolve()
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Preference data does not exist: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=0.1,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="no",
        report_to="none",
        optim="adamw_torch",
        warmup_steps=3,
        lr_scheduler_type="cosine",
        seed=args.seed,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(load_pairs(data_path, tokenizer)),
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(output_dir / "final")


if __name__ == "__main__":
    main()
