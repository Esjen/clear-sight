"""
flatten_features.py

Converts the raw EMBER JSON rows (nested dicts with histogram, imports,
section info, etc.) into fixed-length numeric feature vectors that a
machine learning model can actually be trained on.

COMPATIBILITY NOTE:
--------------------
The `ember` package was written for an older version of scikit-learn.
Modern scikit-learn added a stricter check inside FeatureHasher that
breaks EMBER's internal code with this error:

    ValueError: Samples can not be a single string. The input must be
    an iterable over iterables of strings.

This is a known issue, not a bug in your setup. The patch below fixes
it by wrapping single strings into one-element lists before they reach
FeatureHasher, which is the documented community workaround. This is
worth a one-line mention in your Chapter 3/4 methodology if you like:
you adapted the EMBER feature extractor for compatibility with a newer
scikit-learn version.

What the rest of the script does:
----------------------------------
1. Loads each processed shard (.pkl) one at a time
2. Converts every row's raw features into a 2381-length numeric vector
3. Saves each shard's numeric features + labels as .npy files
4. Combines all shards into one final X_train.npy / y_train.npy pair,
   written directly to disk so we never hold the full combined dataset
   in RAM all at once.

Run with:
    python flatten_features.py
"""

import numpy as np
import pandas as pd
import glob
import os
import gc

# ---------------------------------------------------------------------
# COMPATIBILITY PATCH — must run BEFORE ember is used.
# Fixes FeatureHasher rejecting bare strings in newer scikit-learn.
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

SHARD_DIR = "../data/ember_2018/processed_shards"
FLAT_DIR = "../data/ember_2018/flattened_shards"
FINAL_DIR = "../data/ember_2018"

FEATURE_VERSION = 2  # EMBER 2018 dataset uses feature version 2


def flatten_one_shard(pkl_path, extractor, out_dir):
    shard_name = os.path.basename(pkl_path).replace(".pkl", "")
    X_path = os.path.join(out_dir, f"X_{shard_name}.npy")
    y_path = os.path.join(out_dir, f"y_{shard_name}.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"  {shard_name}: already flattened, skipping.")
        return X_path, y_path

    df = pd.read_pickle(pkl_path)
    n_rows = len(df)
    dim = extractor.dim  # length of each feature vector (2381)

    X = np.zeros((n_rows, dim), dtype=np.float32)
    y = np.zeros(n_rows, dtype=np.float32)

    for i, row in enumerate(df.to_dict("records")):
        X[i, :] = extractor.process_raw_features(row)
        y[i] = row["label"]

        if (i + 1) % 20000 == 0:
            print(f"    {shard_name}: {i + 1}/{n_rows} rows converted...")

    np.save(X_path, X)
    np.save(y_path, y)
    print(f"  {shard_name}: {n_rows} rows flattened -> shape {X.shape}, saved.")

    del df, X, y
    gc.collect()

    return X_path, y_path


def main():
    os.makedirs(FLAT_DIR, exist_ok=True)

    shard_files = sorted(glob.glob(os.path.join(SHARD_DIR, "*.pkl")))
    if not shard_files:
        print(f"No shard .pkl files found in {SHARD_DIR}")
        print("Run load_ember_train_v2.py first to generate these.")
        return

    print(f"Found {len(shard_files)} shard files. Setting up feature extractor...")
    extractor = PEFeatureExtractor(feature_version=FEATURE_VERSION)
    print(f"Each row will become a vector of {extractor.dim} numbers.\n")

    flat_paths = []
    for pkl_path in shard_files:
        paths = flatten_one_shard(pkl_path, extractor, FLAT_DIR)
        flat_paths.append(paths)

    print("\nAll shards flattened individually.")
    print("Now combining into one final dataset (written directly to disk)...\n")

    shapes = []
    for X_path, _ in flat_paths:
        arr = np.load(X_path, mmap_mode="r")
        shapes.append(arr.shape[0])
    total_rows = sum(shapes)
    dim = extractor.dim

    X_final_path = os.path.join(FINAL_DIR, "X_train.npy")
    y_final_path = os.path.join(FINAL_DIR, "y_train.npy")

    X_final = np.lib.format.open_memmap(
        X_final_path, mode="w+", dtype=np.float32, shape=(total_rows, dim)
    )
    y_final = np.lib.format.open_memmap(
        y_final_path, mode="w+", dtype=np.float32, shape=(total_rows,)
    )

    offset = 0
    for (X_path, y_path), n in zip(flat_paths, shapes):
        X_shard = np.load(X_path, mmap_mode="r")
        y_shard = np.load(y_path, mmap_mode="r")

        X_final[offset:offset + n] = X_shard[:]
        y_final[offset:offset + n] = y_shard[:]

        offset += n
        print(f"  Wrote rows {offset - n} to {offset} into final dataset.")

    X_final.flush()
    y_final.flush()

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Final combined features: {X_final_path}, shape {X_final.shape}")
    print(f"Final combined labels:   {y_final_path}, shape {y_final.shape}")
    print("\nLabel distribution:")
    unique, counts = np.unique(y_final, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {int(u)}: {c}")


if __name__ == "__main__":
    main()
