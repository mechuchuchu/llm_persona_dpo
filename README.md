# Qwen3.5 prompt comparison

`extract_first_prompt.py` fetches only row 0 of `allenai/Dolci-Instruct-SFT-No-Tools`
through the Hugging Face dataset viewer API and saves its first non-empty user
message to `data/first_prompt.txt`. It records the dataset row ID and source in
`data/first_prompt.metadata.json`.

The model-specific system prompts are in `prompts/`. After both model folders are
present under `models/`, run:

```bash
python extract_first_prompt.py
python generate_responses.py
```

The script disables Qwen's thinking mode and runs the models sequentially to fit
the available GPU memory. It saves the prompt, generation settings, and both
outputs to `data/responses.json`.

For paired preference generation with separate system prompts for each model, followed by
balanced system-prompt shuffling into Parquet, see
[`README_preference_data.md`](README_preference_data.md). Those scripts are
prepared separately and have not been run.

## DPO fine-tuning with Qwen3.5-4B and LoRA

Download the base model and train on `data/custom_answer_pairs_9b.jsonl`:

```bash
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
python train_dpo_lora.py
```

The trainer reads JSONL rows with non-empty `user`, `chosen`, and `rejected`
strings. Defaults are 3 epochs, LoRA rank 16, bf16, and a maximum sequence
length of 1024 tokens. Override them with `--help` options, such as
`--epochs`, `--learning-rate`, or `--max-length`.

The final adapter is saved to `outputs/qwen3.5-4b-dpo-lora/final`; the base
model remains in `models/Qwen3.5-4B`. Intermediate epoch checkpoints are saved
in the same output directory.

## Test a LoRA adapter in a chat UI

Install the chat dependencies into the PyTorch environment:

```bash
source /venv/main/bin/activate
uv pip install -r requirements-chat.txt
```

After training, start the Gradio chat page:

```bash
python chat_lora.py
```

It loads the local Qwen3.5-4B base model in 4-bit NF4 with bitsandbytes, then
attaches `outputs/qwen3.5-4b-dpo-lora/final`. The system prompt defaults to
`generation_prompts/chosen_system.txt` and can be edited in the UI. You can
override the model, adapter, prompt file, or local bind address at startup:

```bash
python chat_lora.py --model models/Qwen3.5-4B --adapter outputs/qwen3.5-4b-dpo-lora/final
```

To compare against the base model without an adapter, pass `--adapter ''`.
The default server binds to `127.0.0.1:7860`; from your own computer, reach it
privately with SSH local forwarding (`-L 7860:127.0.0.1:7860`) and open
`http://localhost:7860`.

For a terminal-only chat, use the same model, adapter, and system-prompt defaults:

```bash
uv pip install -r requirements-terminal.txt
python chat_lora_terminal.py
```

Enter `/clear` to reset the conversation, or `/exit` to quit. The terminal
version uses the shared model loader and does not require Gradio.
