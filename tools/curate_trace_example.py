"""Extract one compact subject trajectory from an experiment JSONL trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source trace JSONL file")
    parser.add_argument("output", type=Path, help="Destination example JSONL file")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--start-ts", required=True, help="Inclusive ISO timestamp")
    parser.add_argument("--end-ts", required=True, help="Inclusive ISO timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected: list[dict] = []

    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            timestamp = event.get("ts", "")
            if (
                event.get("_event") == "turn"
                and event.get("subject") == args.subject
                and args.start_ts <= timestamp <= args.end_ts
            ):
                selected.append(event)

    if not selected:
        raise ValueError("No matching turn events found")
    if selected[0].get("ts") != args.start_ts or selected[-1].get("ts") != args.end_ts:
        raise ValueError("Timestamp bounds did not match the selected trajectory exactly")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for event in selected:
            destination.write(json.dumps(event, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} turns to {args.output}")


if __name__ == "__main__":
    main()
