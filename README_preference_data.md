# Preference data preparation

These scripts are prepared but have not been run.

`generate_preference_pairs.py` sends the same user prompts to both models in
non-thinking mode, with a separate generation system prompt for each model. It
frees the GPU between models and writes only `user`, `chosen`, and `rejected` to
`data/user_answer_pairs.parquet`. The chosen model defaults to the downloaded
Qwen3.5-2B; the rejected model defaults to `Qwen/Qwen3-0.6B`. Both model and
system prompt paths can be overridden. `--batch-size` controls how many user
prompts are sent to vLLM per call; `--num-prompts` controls how many pairs are
generated. Each input user prompt produces one chosen/rejected pair.

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
specified directory. It duplicates each source pair once per system prompt,
shuffles the rows with a reproducible seed, and writes
`data/shuffle_sys_prompt.parquet`. The output columns are `system`, `user`,
`chosen`, `rejected`, and `system_prompt_file`. The final system and user prompt
occur once per row and are shared by the chosen and rejected responses. These
shuffled prompts are added after response generation.

```bash
python shuffle_sys_prompt.py --pairs data/user_answer_pairs.parquet --system-prompts-dir /path/to/system_prompts --output data/shuffle_sys_prompt.parquet --seed 42
```

Each system prompt appears exactly once per source pair, so prompt proportions
are equal. The row order is shuffled after balancing.
