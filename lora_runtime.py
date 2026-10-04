"""Shared bitsandbytes and PEFT helpers for the LoRA chat interfaces."""

from pathlib import Path
from threading import Thread

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


ROOT = Path(__file__).resolve().parent


def load_model(model_name_or_path: str, adapter_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for 4-bit bitsandbytes loading.")

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization,
        device_map="auto",
        dtype=compute_dtype,
    )

    if adapter_path:
        adapter_dir = Path(adapter_path).expanduser()
        if not adapter_dir.is_dir():
            raise FileNotFoundError(
                f"LoRA adapter directory not found: {adapter_dir}\n"
                "Train it first with train_dpo_lora.py, or pass --adapter '' to test the base model."
            )
        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    return model, tokenizer


def load_system_prompt(path: str) -> str:
    if not path:
        return ""
    prompt_path = Path(path).expanduser()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _prepare_generation(
    model,
    tokenizer,
    messages,
    max_input_length: int,
    temperature: float,
    max_new_tokens: int,
) -> dict:
    original_messages = messages
    messages = list(messages)
    while True:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        system_count = int(bool(messages and messages[0].get("role") == "system"))
        if len(prompt_tokens) <= max_input_length or len(messages) <= system_count + 1:
            break
        # Drop the oldest user/assistant turn while keeping the system prompt and latest input.
        del messages[system_count : system_count + 2]

    if isinstance(original_messages, list):
        original_messages[:] = messages

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    generation = {
        **encoded,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": float(temperature) > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if float(temperature) > 0:
        generation["temperature"] = float(temperature)
        generation["top_p"] = 0.9

    return generation


def generate_reply(
    model,
    tokenizer,
    messages,
    max_input_length: int,
    temperature: float,
    max_new_tokens: int,
) -> str:
    generation = _prepare_generation(
        model, tokenizer, messages, max_input_length, temperature, max_new_tokens
    )
    with torch.inference_mode():
        output = model.generate(**generation)
    prompt_length = generation["input_ids"].shape[-1]
    answer = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True).strip()
    return answer or "(빈 응답)"


def stream_reply(
    model,
    tokenizer,
    messages,
    max_input_length: int,
    temperature: float,
    max_new_tokens: int,
):
    generation = _prepare_generation(
        model, tokenizer, messages, max_input_length, temperature, max_new_tokens
    )
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation["streamer"] = streamer
    errors = []

    def run_generation():
        try:
            with torch.inference_mode():
                model.generate(**generation)
        except Exception as exc:
            errors.append(exc)
            streamer.end()

    worker = Thread(target=run_generation, daemon=True)
    worker.start()
    try:
        for fragment in streamer:
            if fragment:
                yield fragment
    finally:
        worker.join()

    if errors:
        raise errors[0]
