#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(__file__).resolve().parent

SOURCE_JSONL = DATA_DIR / "multi_turn_processed.jsonl"
TRAIN200_SPLIT_JSON = DATA_DIR / "multi_turn_envtuning_train200_split_ids.json"
OUT_SPLIT_JSON = DATA_DIR / "multi_turn_envtuning_train200_test600_split_ids.json"

TRAIN_PARQUET = ROOT / "data" / "bfcl_train.parquet"
TEST_PARQUET = ROOT / "data" / "bfcl_test.parquet"
OUT_TRAIN_PARQUET = ROOT / "data" / "bfcl_train_200.parquet"
OUT_TEST_PARQUET = ROOT / "data" / "bfcl_test_600.parquet"


def category_of(task_id: str) -> str:
    return task_id.rsplit("_", 1)[0]


def task_id_from_extra_info(extra_info: dict) -> str:
    return str(extra_info.get("original_id") or extra_info.get("index") or "")


def synthesize_row(task_id: str) -> dict:
    category = category_of(task_id)
    return {
        "data_source": category,
        "prompt": [{"role": "user", "content": task_id}],
        "ability": "multi_turn_function_calling",
        "reward_model": {"ground_truth": [], "style": "interaction"},
        "extra_info": {
            "dataset_type": category,
            "data_source": category,
            "index": task_id,
            "original_id": task_id,
            "split": "test",
            "interaction_kwargs": {
                "id": task_id,
                "split": "test",
            },
        },
    }


def main() -> None:
    with open(TRAIN200_SPLIT_JSON, "r", encoding="utf-8") as f:
        train200_split = json.load(f)

    train_ids: list[str] = train200_split["train"]

    all_ids: list[str] = []
    with open(SOURCE_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            all_ids.append(entry["id"])

    train_id_set = set(train_ids)
    test_ids = [task_id for task_id in all_ids if task_id not in train_id_set]

    split = {"train": train_ids, "val": test_ids}
    with open(OUT_SPLIT_JSON, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    existing = {}
    for parquet_path in (TRAIN_PARQUET, TEST_PARQUET):
        df = pd.read_parquet(parquet_path)
        for record in df.to_dict(orient="records"):
            task_id = task_id_from_extra_info(record.get("extra_info", {}) or {})
            if task_id:
                existing[task_id] = record

    train_rows = []
    for task_id in train_ids:
        row = existing.get(task_id)
        if row is None:
            raise KeyError(f"Missing train task in source parquet union: {task_id}")
        row = dict(row)
        extra_info = dict(row.get("extra_info", {}) or {})
        extra_info["split"] = "train"
        extra_info.setdefault("dataset_type", category_of(task_id))
        extra_info.setdefault("data_source", category_of(task_id))
        extra_info.setdefault("index", task_id)
        extra_info.setdefault("original_id", task_id)
        row["extra_info"] = extra_info
        row["data_source"] = category_of(task_id)
        train_rows.append(row)

    test_rows = []
    missing = []
    for task_id in test_ids:
        row = existing.get(task_id)
        if row is None:
            missing.append(task_id)
            row = synthesize_row(task_id)
        else:
            row = dict(row)
            extra_info = dict(row.get("extra_info", {}) or {})
            extra_info["split"] = "test"
            extra_info.setdefault("dataset_type", category_of(task_id))
            extra_info.setdefault("data_source", category_of(task_id))
            extra_info.setdefault("index", task_id)
            extra_info.setdefault("original_id", task_id)
            row["extra_info"] = extra_info
            row["data_source"] = category_of(task_id)
        test_rows.append(row)

    train_df = pd.DataFrame(train_rows)
    out_df = pd.DataFrame(test_rows)
    OUT_TRAIN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(OUT_TRAIN_PARQUET, index=False)
    OUT_TEST_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUT_TEST_PARQUET, index=False)

    print(f"Wrote split json -> {OUT_SPLIT_JSON}")
    print(f"train={len(train_ids)} val={len(test_ids)}")
    print(f"Wrote parquet -> {OUT_TRAIN_PARQUET}")
    print(train_df['data_source'].value_counts().to_dict())
    print(f"Wrote parquet -> {OUT_TEST_PARQUET}")
    print(out_df['data_source'].value_counts().to_dict())
    if missing:
        print(f"Synthesized {len(missing)} missing rows")
        print(missing[:20])


if __name__ == "__main__":
    main()
