"""
load_ember_train_v2.py

Memory-safe version. Instead of loading all 6 training shards into one
giant list before building a DataFrame (which can use several GB of RAM
and crash), this version processes ONE shard at a time:

  1. Load shard N
  2. Drop unlabeled rows (label == -1)
  3. Save that shard's cleaned data to its own file on disk
  4. Free the memory
  5. Move to shard N+1

At the end, it combines the smaller saved files together, which uses
far less peak memory than doing everything in one giant step.

Run with:
    python load_ember_train_v2.py
"""

import json
import pandas as pd
import glob
import os
import gc

BASE_DIR = "../data/ember_2018/ember_dataset_2018_2/ember2018"
OUTPUT_DIR = "../data/ember_2018/processed_shards"


def process_one_shard(filepath, out_dir):
    shard_name = os.path.basename(filepath).replace(".jsonl", "")
    out_path = os.path.join(out_dir, f"{shard_name}.pkl")

    # Skip re-processing if we already did this shard before
    if os.path.exists(out_path):
        print(f"  {shard_name}: already processed, skipping.")
        return out_path

    rows = []
    rows_in_shard = 0

    with open(filepath, "r") as f:
        for line in f:
            row = json.loads(line)
            rows_in_shard += 1
            if row["label"] != -1:
                rows.append(row)

    df_shard = pd.DataFrame(rows)
    df_shard.to_pickle(out_path)

    print(f"  {shard_name}: {rows_in_shard} rows read, "
          f"{len(rows)} labeled rows kept, saved to {out_path}")

    # Explicitly free memory before moving to the next shard
    del rows, df_shard
    gc.collect()

    return out_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_files = sorted(glob.glob(os.path.join(BASE_DIR, "train_features_*.jsonl")))

    if not train_files:
        print(f"No train_features_*.jsonl files found in: {BASE_DIR}")
        return

    print(f"Found {len(train_files)} shard files. Processing one at a time...\n")

    shard_pkl_paths = []
    for filepath in train_files:
        path = process_one_shard(filepath, OUTPUT_DIR)
        shard_pkl_paths.append(path)

    print("\nAll shards processed individually.")
    print("Now combining the smaller files into one final dataset...")

    combined = pd.concat(
        [pd.read_pickle(p) for p in shard_pkl_paths],
        ignore_index=True
    )

    final_path = os.path.join(BASE_DIR, "..", "combined_train.pkl")
    combined.to_pickle(final_path)

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Combined training set shape: {combined.shape}")
    print("\nLabel distribution:")
    print(combined["label"].value_counts())
    print(f"\nSaved final combined file to: {final_path}")


if __name__ == "__main__":
    main()
