# Preference data preparation

These scripts are prepared but have not been run.

`generate_preference_pairs.py` sends the same user prompts to both models in
non-thinking mode, with a separate generation system prompt for each model. It
frees the GPU between models and writes only `user`, `chosen`, and `rejected` to
`data/user_answer_pairs.parquet`. The chosen model defaults to the downloaded
Qwen3.5-2B; the rejected model defaults to `Qwen/Qwen3-0.6B`. Both model and
system prompt paths can be overridden.

```bash
python generate_preference_pairs.py --users data/first_prompt.txt --chosen-system-prompt prompts/qwen3_5_2b_system.txt --rejected-system-prompt prompts/qwen3_0_6b_system.txt --chosen-model models/Qwen3.5-2B --rejected-model Qwen/Qwen3-0.6B
```

The two generation system prompts are independent, while both model runs receive
the same ordered user prompts. Override either system prompt or the user prompt
input path as needed.

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
