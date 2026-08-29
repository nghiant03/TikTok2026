from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def run_training(output_dir: Path, seed: int, fidelity: str) -> Path:
    if fidelity not in {"smoke", "proxy", "full"}:
        raise ValueError("unknown fidelity")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    predictions = output_dir / "predictions.json"
    predictions.write_text(
        json.dumps({"schema_version": "1", "scores": [random.random() for _ in range(8)]}),
        encoding="utf-8",
    )
    bundle = output_dir / "checkpoint_bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "seed": seed,
                "fidelity": fidelity,
                "predictions": predictions.name,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fidelity", choices=("smoke", "proxy", "full"), required=True)
    arguments = parser.parse_args()
    run_training(arguments.output_dir, arguments.seed, arguments.fidelity)


if __name__ == "__main__":
    main()
