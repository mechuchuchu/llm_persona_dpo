# Preference data preparation

The preference-pair generator accepts text, JSONL, parquet, or streamed dataset
prompts and can write generated pairs as Parquet or JSONL.

`generate_preference_pairs.py` sends the same user prompts to both models in
non-thinking mode, with a separate generation system prompt for each model. It
frees the GPU between models and writes only `user`, `chosen`, and `rejected`.
The default output is `data/user_answer_pairs.parquet`. The chosen model
defaults to the downloaded Qwen3.5-2B; the rejected model defaults to
`Qwen/Qwen3-0.6B`. Both model and system prompt paths can be overridden.
`--batch-size` controls how many user
prompts are sent to vLLM per call; `--num-prompts` controls how many pairs are
generated. Each input user prompt produces one chosen/rejected pair.

For custom JSONL prompts, put one JSON object on each line (do not wrap the
records in a JSON array). Each object needs a non-empty string in either
`user` or `prompt`:

```jsonl
{"user":"Sort these words: pear, apple"}
{"prompt":"Explain why the sky looks blue."}
```

Generated JSONL has one object per prompt, with the prompt and both responses:

```jsonl
{"user":"Sort these words: pear, apple","chosen":"apple, pear","rejected":"pear, apple"}
```

`--output-format jsonl` selects that output format; it is also inferred when
`--output` ends in `.jsonl`. Without an explicit output path, JSONL output goes
to `data/user_answer_pairs.jsonl`.

```bash
python generate_preference_pairs.py --users data/custom_prompts.jsonl --output-format jsonl --output data/custom_answer_pairs.jsonl
```

During generation, a progress bar shows completed prompts for each model. The
script counts tokens after applying the model's chat template and reserves
`--max-new-tokens` within `--max-model-len`. If a prompt is too long, it trims
the middle of the system prompt while keeping its beginning and end, then trims
the beginning of the user prompt if needed. The script reports how many rows
were shortened.

```bash
python generate_preference_pairs.py --users data/first_prompt.txt --num-prompts 1 --batch-size 1 --chosen-system-prompt generation_prompts/chosen_system.txt --rejected-system-prompt generation_prompts/rejected_system.txt --chosen-model models/Qwen3.5-2B --rejected-model Qwen/Qwen3-0.6B
```

The two generation system prompts are independent, while both model runs receive
the same ordered user prompts. Editable example prompts are in
`generation_prompts/chosen_system.txt` and
`generation_prompts/rejected_system.txt`; they currently contain the same plain,
general-purpose instruction. Edit them independently or override either path.

To stream a requested number of prompts from the original dataset, use:

```bash
python generate_preference_pairs.py --dataset allenai/Dolci-Instruct-SFT-No-Tools --split train --num-prompts 1000 --batch-size 4
```

Streaming input is shuffled with a 10,000-row buffer by default; change that
with `--dataset-shuffle-buffer`. With file input, `--num-prompts` selects the
first N prompts, and it cannot exceed the number available in that file.

If vLLM exits with `Cannot re-initialize CUDA in forked subprocess`, start it
with the `spawn` multiprocessing method. Set `VLLM_WORKER_MULTIPROC_METHOD`
before launching the script:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn uv run generate_preference_pairs.py --dataset allenai/Dolci-Instruct-SFT-No-Tools --split train --num-prompts 1000 --batch-size 4
```

For a regular Python environment, replace `uv run` with `python`.

`shuffle_sys_prompt.py` reads every non-empty `.txt` file recursively from the
specified directory. Its `--pairs` input can be either Parquet or JSONL. JSONL
must contain one object per line with string fields `user`, `chosen`, and
`rejected`, for example:

```jsonl
{"user":"Sort these words: pear, apple","chosen":"apple, pear","rejected":"pear, apple"}
```

The script duplicates each source pair once per system prompt, shuffles the
rows with a reproducible seed, and writes
`data/shuffle_sys_prompt.parquet`. The output columns are `system`, `user`,
`chosen`, `rejected`, and `system_prompt_file`. The final system and user prompt
occur once per row and are shared by the chosen and rejected responses. These
shuffled prompts are added after response generation.

```bash
python shuffle_sys_prompt.py --pairs data/user_answer_pairs.parquet --system-prompts-dir /path/to/system_prompts --output data/shuffle_sys_prompt.parquet --seed 42
```

To use JSONL preference pairs as input and still produce the final Parquet:

```bash
python shuffle_sys_prompt.py --pairs data/user_answer_pairs.jsonl --system-prompts-dir /path/to/system_prompts --output data/shuffle_sys_prompt.parquet --seed 42
```

Each system prompt appears exactly once per source pair, so prompt proportions
are equal. The row order is shuffled after balancing.
