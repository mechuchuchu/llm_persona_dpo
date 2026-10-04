#!/usr/bin/env python3
"""Generate one response from each local Qwen3.5 model with vLLM."""

import gc
import json
from pathlib import Path

from vllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "data" / "first_prompt.txt"
METADATA_PATH = ROOT / "data" / "first_prompt.metadata.json"
OUTPUT_PATH = ROOT / "data" / "responses.json"

MODELS = (
    (
        "Qwen/Qwen3.5-2B",
        ROOT / "models" / "Qwen3.5-2B",
        ROOT / "prompts" / "qwen3_5_2b_system.txt",
    ),
    (
        "Qwen/Qwen3.5-0.8B",
        ROOT / "models" / "Qwen3.5-0.8B",
        ROOT / "prompts" / "qwen3_5_0_8b_system.txt",
    ),
)


def main() -> None:
    user_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=1536,
    )
    results = {
        "dataset_record": metadata,
        "user_prompt": user_prompt,
        "generation": {
            "enable_thinking": False,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 1536,
            "max_model_len": 4096,
        },
        "models": [],
    }

    for model_id, model_path, system_path in MODELS:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory is missing: {model_path}")
        system_prompt = system_path.read_text(encoding="utf-8").strip()
        print(f"Loading {model_id} from {model_path} ...", flush=True)
        engine = LLM(
            model=str(model_path),
            dtype="auto",
            trust_remote_code=False,
            max_model_len=4096,
            gpu_memory_utilization=0.88,
            max_num_seqs=1,
        )
        try:
            outputs = engine.chat(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}],
                sampling,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            result = {
                "model_id": model_id,
                "model_path": str(model_path),
                "system_prompt_file": str(system_path.relative_to(ROOT)),
                "system_prompt": system_prompt,
                "response": outputs[0].outputs[0].text,
            }
            results["models"].append(result)
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"Generated response for {model_id}", flush=True)
        finally:
            del engine
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"Saved both responses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
