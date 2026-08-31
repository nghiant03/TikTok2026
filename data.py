#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
from pathlib import Path


SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test":  (20220429, 20220508),
}

INPUT_LOGS = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity_line(row_id: str, user_id: str, item_id: str) -> bytes:
    # Must match TikTok2026 encode_row_identity():
    # json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    payload = json.dumps(
        [row_id, user_id, item_id],
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return (payload + "\n").encode("utf-8")


def split_for_date(date: int) -> str | None:
    for name, (lo, hi) in SPLITS.items():
        if lo <= date <= hi:
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        help="Path to extracted KuaiRand-Pure directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output TikTok2026 dataset root",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()

    # Accept either:
    #   KuaiRand-Pure/
    # or
    #   KuaiRand-Pure/data/
    data = source / "data" if (source / "data").is_dir() else source

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load the Starter Kit's one required side feature:
    # video_id -> author_id
    # --------------------------------------------------------

    video_features = data / "video_features_basic_pure.csv"

    if not video_features.is_file():
        raise FileNotFoundError(video_features)

    video_to_author: dict[str, str] = {}

    with video_features.open(
        newline="",
        encoding="utf-8",
    ) as f:
        for row in csv.DictReader(f):
            video_to_author[row["video_id"]] = row["author_id"]

    print(f"Loaded {len(video_to_author):,} video->author mappings")

    # --------------------------------------------------------
    # Inspect/validate source schemas
    # --------------------------------------------------------

    source_columns = None

    for filename in INPUT_LOGS:
        path = data / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = tuple(reader.fieldnames or ())

        required = {"user_id", "video_id", "date", "long_view"}

        if not required <= set(columns):
            raise RuntimeError(
                f"{filename} missing required columns: "
                f"{sorted(required - set(columns))}"
            )

        if source_columns is None:
            source_columns = columns
        elif columns != source_columns:
            raise RuntimeError(
                "standard interaction files have different schemas"
            )

    assert source_columns is not None

    # Canonical columns required by TikTok2026 first.
    generated = ("row_id", "item_id", "label", "author_id")

    # Don't duplicate if a future dataset version already has one.
    output_columns = generated + tuple(
        c for c in source_columns if c not in generated
    )

    # --------------------------------------------------------
    # Open output split CSVs
    # --------------------------------------------------------

    handles = {}
    writers = {}

    counters = {
        "train": 0,
        "valid": 0,
        "test": 0,
    }

    identity_hashes = {
        name: hashlib.sha256()
        for name in SPLITS
    }

    try:
        for split in SPLITS:
            path = output / f"{split}.csv"

            handle = path.open(
                "w",
                newline="",
                encoding="utf-8",
            )

            handles[split] = handle

            writer = csv.DictWriter(
                handle,
                fieldnames=output_columns,
                extrasaction="ignore",
            )

            writer.writeheader()
            writers[split] = writer

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Process the files in the same order as baseline/data.py.
        # Rows remain in original source order.
        # ----------------------------------------------------

        ignored = 0

        for filename in INPUT_LOGS:
            path = data / filename

            print(f"Reading {path}")

            with path.open(
                newline="",
                encoding="utf-8",
            ) as f:

                reader = csv.DictReader(f)

                for raw in reader:
                    date = int(raw["date"])
                    split = split_for_date(date)

                    if split is None:
                        ignored += 1
                        continue

                    row_id = str(counters[split])
                    user_id = raw["user_id"]
                    item_id = raw["video_id"]

                    long_view = raw["long_view"]

                    if long_view not in {"0", "1"}:
                        raise ValueError(
                            f"unexpected long_view={long_view!r}"
                        )

                    prepared = dict(raw)

                    prepared["row_id"] = row_id
                    prepared["item_id"] = item_id
                    prepared["label"] = long_view
                    prepared["author_id"] = video_to_author.get(
                        item_id,
                        "UNK",
                    )

                    writers[split].writerow(prepared)

                    identity_hashes[split].update(
                        identity_line(
                            row_id,
                            user_id,
                            item_id,
                        )
                    )

                    counters[split] += 1

    finally:
        for handle in handles.values():
            handle.close()

    # --------------------------------------------------------
    # Build manifest
    # --------------------------------------------------------

    files = []

    for split in ("train", "valid", "test"):
        filename = f"{split}.csv"
        path = output / filename

        files.append(
            {
                "path": filename,
                "sha256": sha256_file(path),
                "schema": list(output_columns),
                "split": split,
            }
        )

    manifest = {
        "schema_version": "1",
        "manifest_id": "kuairand-pure-techjam2026-official-split-v1",
        "data_root_env": "TIKTOK2026_KUAIRAND_PURE_DATA",

        # Explicitly use the names expected by Docker staging.
        "row_identity_encoding": "json-array-v1",
        "row_identity_columns": [
            "row_id",
            "user_id",
            "item_id",
        ],

        "user_id_column": "user_id",
        "item_id_column": "item_id",
        "label_column": "label",

        "non_label_feature_columns": [],

        "files": files,

        "splits": {
            split: {
                "files": [f"{split}.csv"],
                "identity_sha256":
                    identity_hashes[split].hexdigest(),
            }
            for split in ("train", "valid", "test")
        },
    }

    manifest_path = output / "manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Prepared dataset:")
    print(output)

    for split in ("train", "valid", "test"):
        print(
            f"{split:5s}: "
            f"{counters[split]:,} rows"
        )

    print(f"ignored: {ignored:,} rows")
    print()
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
