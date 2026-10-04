#!/usr/bin/env python3
"""Extract the first user message from the first Dolci-Instruct-SFT-No-Tools row."""

import json
from pathlib import Path
from urllib.request import urlopen


DATASET = "allenai/Dolci-Instruct-SFT-No-Tools"
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=allenai%2FDolci-Instruct-SFT-No-Tools"
    "&config=default&split=train&offset=0&length=1"
)
ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "data" / "first_prompt.txt"
METADATA_PATH = ROOT / "data" / "first_prompt.metadata.json"


def main() -> None:
    with urlopen(ROWS_URL, timeout=60) as response:
        payload = json.load(response)

    row = payload["rows"][0]["row"]
    first_user_message = next(
        (
            message.get("content", "")
            for message in row.get("messages", [])
            if message.get("role") == "user" and message.get("content")
        ),
        None,
    )
    if first_user_message is None:
        raise RuntimeError("The first dataset row has no non-empty user message.")

    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_PATH.write_text(first_user_message.rstrip() + "\n", encoding="utf-8")
    METADATA_PATH.write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "split": "train",
                "row_offset": 0,
                "id": row.get("id"),
                "source": row.get("source"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved the first user message to {PROMPT_PATH}")
    print(f"Record ID: {row.get('id')}")


if __name__ == "__main__":
    main()
