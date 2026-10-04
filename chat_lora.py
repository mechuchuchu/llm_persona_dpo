#!/usr/bin/env python3
"""Run a small Gradio chat UI with a 4-bit base model and a PEFT LoRA adapter."""

import argparse

import gradio as gr
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
        help="Text file to use as the initial system prompt.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--max-input-length", type=int, default=4096)
    return parser.parse_args()


def make_chat_fn(model, tokenizer, max_input_length: int):
    def respond(message, history, system_prompt, temperature, max_new_tokens):
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.extend(history or [])
        messages.append({"role": "user", "content": message})
        return generate_reply(
            model,
            tokenizer,
            messages,
            max_input_length,
            temperature,
            max_new_tokens,
        )

    return respond


def main():
    args = parse_args()
    model, tokenizer = load_model(args.model, args.adapter)
    system_prompt = load_system_prompt(args.system_prompt)

    chat = gr.ChatInterface(
        fn=make_chat_fn(model, tokenizer, args.max_input_length),
        title="LoRA 채팅 테스트",
        description=(
            f"베이스: `{args.model}`  ·  "
            f"LoRA: `{args.adapter or '사용 안 함'}`  ·  4-bit bitsandbytes"
        ),
        chatbot=gr.Chatbot(type="messages", height=560),
        textbox=gr.Textbox(placeholder="메시지를 입력하세요", container=False),
        additional_inputs=[
            gr.Textbox(
                label="시스템 프롬프트",
                value=system_prompt,
                lines=6,
            ),
            gr.Slider(
                label="Temperature",
                minimum=0,
                maximum=1.5,
                value=0.7,
                step=0.1,
            ),
            gr.Slider(
                label="최대 생성 토큰",
                minimum=32,
                maximum=2048,
                value=512,
                step=32,
            ),
        ],
        examples=[
            "요즘 내가 중요하게 생각하는 가치가 뭐였지?",
            "피부가 건조한데 아침 스킨케어 순서를 추천해줘.",
        ],
    )
    chat.queue().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
