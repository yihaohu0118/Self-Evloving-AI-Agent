#!/usr/bin/env python3
"""
Generate a mixed multi-turn dataset for SEAL training.

Reads from multi_turn_processed.jsonl (already preprocessed with initial_config
and involved_classes) and splits by category prefix.

Categories: base, miss_func, miss_param, long_context
Each: 100 train + 100 val (total 400 train, 400 val)

Outputs:
  multi_turn_mixed_processed.jsonl   — all 800 entries
  multi_turn_mixed_split_ids.json    — {"train": [...], "val": [...]}

Usage (on VM):
  cd env_service/environments/bfcl
  python bfcl_data/gen_mixed_split.py
"""
import json
import random
from pathlib import Path

SEED = 42
TRAIN_PER_CAT = 100

CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

DATA_DIR = Path(__file__).parent
SOURCE_FILE = DATA_DIR / "multi_turn_processed.jsonl"


def load_jsonl(path: Path):
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    random.seed(SEED)

    if not SOURCE_FILE.exists():
        print(
            f"ERROR: {SOURCE_FILE} not found.\n"
            f"Run bfcl_dataprocess.py with test_categories=['multi_turn'] first."
        )
        return

    all_entries = load_jsonl(SOURCE_FILE)
    print(f"Loaded {len(all_entries)} entries from {SOURCE_FILE.name}")

    # Group by category prefix
    by_cat = {cat: [] for cat in CATEGORIES}
    for entry in all_entries:
        tid = entry.get("id", "")
        cat = tid.rsplit("_", 1)[0] if "_" in tid else ""
        if cat in by_cat:
            by_cat[cat].append(entry)

    all_train_ids = []
    all_val_ids = []
    out_entries = []

    for cat in CATEGORIES:
        entries = by_cat[cat]
        if not entries:
            print(f"WARNING: no entries for {cat}")
            continue

        random.shuffle(entries)

        train_part = entries[:TRAIN_PER_CAT]
        val_part = entries[TRAIN_PER_CAT : TRAIN_PER_CAT * 2]

        # Tag data_source for per-category val metrics
        for e in train_part + val_part:
            e["data_source"] = cat

        all_train_ids.extend([e["id"] for e in train_part])
        all_val_ids.extend([e["id"] for e in val_part])
        out_entries.extend(train_part)
        out_entries.extend(val_part)

        print(f"  {cat}: {len(train_part)} train, {len(val_part)} val")

    # Write combined JSONL
    out_jsonl = DATA_DIR / "multi_turn_mixed_processed.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for entry in out_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out_entries)} entries → {out_jsonl.name}")

    # Write split IDs
    split = {"train": all_train_ids, "val": all_val_ids}
    out_split = DATA_DIR / "multi_turn_mixed_split_ids.json"
    with open(out_split, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"Wrote split IDs ({len(all_train_ids)} train, {len(all_val_ids)} val) → {out_split.name}")


if __name__ == "__main__":
    main()
