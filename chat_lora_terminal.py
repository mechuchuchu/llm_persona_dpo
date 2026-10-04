#!/usr/bin/env python3
"""Chat with a 4-bit bitsandbytes model and an optional PEFT LoRA adapter."""

import argparse

from lora_runtime import ROOT, generate_reply, load_model, load_system_prompt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-4B"))
    parser.add_argument(
        "--adapter",
        default=str(ROOT / "outputs" / "qwen3.5-4b-dpo-lora" / "final"),
        help="PEFT LoRA adapter directory; pass an empty string to run the base model only.",
    )
    parser.add_argument(
        "--system-prompt",
        default=str(ROOT / "generation_prompts" / "chosen_system.txt"),
        help="Text file to use as the initial system prompt; pass an empty string to disable it.",
    )
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model, args.adapter)
    messages = []
    system_prompt = load_system_prompt(args.system_prompt)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print(f"모델: {args.model}")
    print(f"LoRA: {args.adapter or '사용 안 함'} (4-bit bitsandbytes)")
    print("대화 시작. /clear 는 대화 초기화, /exit 또는 /quit 은 종료입니다.")

    while True:
        try:
            message = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not message:
            continue
        if message in {"/exit", "/quit"}:
            print("종료합니다.")
            break
        if message == "/clear":
            messages = messages[:1] if messages and messages[0]["role"] == "system" else []
            print("대화를 초기화했습니다.")
            continue

        messages.append({"role": "user", "content": message})
        print("응답 생성 중...", flush=True)
        answer = generate_reply(
            model,
            tokenizer,
            messages,
            args.max_input_length,
            args.temperature,
            args.max_new_tokens,
        )
        messages.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}")


if __name__ == "__main__":
    main()
