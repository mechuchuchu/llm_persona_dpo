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
