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
