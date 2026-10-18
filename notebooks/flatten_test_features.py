"""
flatten_test_features.py

Same idea as flatten_features.py, but for the official EMBER test set
(test_features.jsonl) instead of the training shards. This gives you a
proper held-out evaluation set that matches how EMBER is used in the
published literature -- not just a random slice of your training data.

Run with:
    python flatten_test_features.py
"""

import numpy as np
import json
import os
import gc

# ---------------------------------------------------------------------
# COMPATIBILITY PATCH — same fix as flatten_features.py, must run
# BEFORE ember is used.
# ---------------------------------------------------------------------
from sklearn.feature_extraction import FeatureHasher

_original_transform = FeatureHasher.transform

def _patched_transform(self, raw_X):
    if getattr(self, "input_type", None) == "string":
        raw_X = [[x] if isinstance(x, str) else x for x in raw_X]
    return _original_transform(self, raw_X)

FeatureHasher.transform = _patched_transform
# ---------------------------------------------------------------------

from ember import PEFeatureExtractor

TEST_JSONL_PATH = "../data/ember_2018/ember_dataset_2018_2/ember2018/test_features.jsonl"
FINAL_DIR = "../data/ember_2018"
FEATURE_VERSION = 2


def main():
    print("Setting up feature extractor...")
    extractor = PEFeatureExtractor(feature_version=FEATURE_VERSION)
    dim = extractor.dim
    print(f"Each row will become a vector of {dim} numbers.\n")

    # First pass: count rows so we can pre-allocate the array on disk
    print("Counting rows in test_features.jsonl...")
    n_rows = 0
    with open(TEST_JSONL_PATH, "r") as f:
        for _ in f:
            n_rows += 1
    print(f"Found {n_rows} rows.\n")

    X_path = os.path.join(FINAL_DIR, "X_test.npy")
    y_path = os.path.join(FINAL_DIR, "y_test.npy")

    X = np.lib.format.open_memmap(
        X_path, mode="w+", dtype=np.float32, shape=(n_rows, dim)
    )
    y = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.float32, shape=(n_rows,)
    )

    print("Converting rows to feature vectors...")
    with open(TEST_JSONL_PATH, "r") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            X[i, :] = extractor.process_raw_features(row)
            y[i] = row["label"]

            if (i + 1) % 20000 == 0:
                print(f"  {i + 1}/{n_rows} rows converted...")

    X.flush()
    y.flush()

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Test features: {X_path}, shape {X.shape}")
    print(f"Test labels:   {y_path}, shape {y.shape}")
    print("\nLabel distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {int(u)}: {c}")


if __name__ == "__main__":
    main()
