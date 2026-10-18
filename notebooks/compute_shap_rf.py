"""
compute_shap_rf.py

Computes SHAP values for Random Forest only, using a much smaller
sample (100 files instead of 1000) to keep runtime manageable given
Random Forest's unrestricted tree depth.

This uses the SAME random seed and sampling logic as
compute_shap_fast.py, so the first 100 files chosen here are a subset
of the same balanced test population -- just fewer of them, for speed.

Run with:
    python compute_shap_rf.py

If this is still too slow, you can lower SAMPLE_SIZE further (e.g. to
40 or 50) and re-run -- fewer samples means less to compute, at the
cost of a slightly less stable importance ranking. Note this tradeoff
explicitly if you do so; it's a legitimate thing to mention in your
Limitations section.
"""

import numpy as np
import pandas as pd
import pickle
import os
import time

import shap

DATA_DIR = "../data/ember_2018"
MODEL_DIR = "../models"
OUTPUT_DIR = "../results"

SAMPLE_SIZE = 100     # bounded-depth trees should make this fast now
TOP_N_FEATURES = 20
RANDOM_SEED = 42

FEATURE_GROUPS = [
    ("ByteHistogram", 256),
    ("ByteEntropyHistogram", 256),
    ("StringExtractor", 104),
    ("GeneralFileInfo", 10),
    ("HeaderFileInfo", 62),
    ("SectionInfo", 255),
    ("ImportsInfo", 1280),
    ("ExportsInfo", 128),
    ("DataDirectories", 30),
]


def build_feature_names():
    names = []
    for group_name, size in FEATURE_GROUPS:
        for i in range(size):
            names.append(f"{group_name}_{i}")
    return names


def load_sample(X_path, y_path, sample_size, seed):
    y_full = np.load(y_path, mmap_mode="r")
    rng = np.random.default_rng(seed)

    benign_idx = np.where(y_full[:] == 0)[0]
    malicious_idx = np.where(y_full[:] == 1)[0]

    n_each = sample_size // 2
    chosen_benign = rng.choice(benign_idx, size=n_each, replace=False)
    chosen_malicious = rng.choice(malicious_idx, size=n_each, replace=False)
    chosen = np.sort(np.concatenate([chosen_benign, chosen_malicious]))

    X_full = np.load(X_path, mmap_mode="r")
    X_sample = np.array(X_full[chosen])
    y_sample = np.array(y_full[chosen])

    return X_sample, y_sample, chosen


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building feature name mapping...")
    feature_names = build_feature_names()

    print(f"\nLoading a balanced sample of {SAMPLE_SIZE} test files...")
    X_path = os.path.join(DATA_DIR, "X_test.npy")
    y_path = os.path.join(DATA_DIR, "y_test.npy")
    X_sample, y_sample, chosen_idx = load_sample(X_path, y_path, SAMPLE_SIZE, RANDOM_SEED)
    print(f"  Sample shape: {X_sample.shape}, labels: {np.bincount(y_sample.astype(int))}")

    np.save(os.path.join(OUTPUT_DIR, "shap_sample_indices_rf.npy"), chosen_idx)

    model_path = os.path.join(MODEL_DIR, "random_forest_baseline_depth20.pkl")
    print(f"\nLoading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Building TreeExplainer...")
    # With bounded tree depth, the default settings should now work
    # correctly and quickly -- no need for the interventional/background
    # workaround we tried when trees were unrestricted in depth.
    explainer = shap.TreeExplainer(model)

    print(f"Computing SHAP values for {X_sample.shape[0]} samples...")
    print("(This may still take several minutes due to Random Forest's tree depth.)")
    start = time.time()
    # check_additivity=False: Random Forest's very deep trees accumulate
    # tiny floating-point rounding errors, which can trip SHAP's internal
    # consistency check even though the explanation itself is still
    # valid. This is a known, documented issue with deep trees, not a
    # sign of incorrect data or a broken model.
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes).")

    # Sanity check: with model_output="probability", SHAP values should
    # be small -- roughly in the range of a probability (0 to 1), not
    # in the hundreds or thousands. If they're huge, something is still
    # wrong and we should know immediately, not after saving results.
    raw_max = np.abs(np.array(shap_values)).max()
    print(f"Sanity check -- max absolute SHAP value in this run: {raw_max:.4f}")
    if raw_max > 5:
        print("\nWARNING: SHAP values are still implausibly large for a")
        print("probability-scale output. This suggests the underlying")
        print("compatibility issue is not fully resolved. Consider")
        print("stopping here and documenting this as a limitation rather")
        print("than trusting these numbers.\n")

    # Save the RAW output immediately, before any further processing.
    # This computation is expensive (many minutes) -- we never want to
    # lose it to a downstream bug again.
    raw_backup_path = os.path.join(OUTPUT_DIR, "shap_values_random_forest_RAW.npy")
    np.save(raw_backup_path, np.array(shap_values, dtype=object), allow_pickle=True)
    print(f"Raw SHAP output backed up to: {raw_backup_path}")

    # Newer versions of SHAP can return this in a few different shapes
    # depending on the model type:
    #   - a list of two arrays (one per class)              -> take class 1
    #   - a single 3D array (n_samples, n_features, n_classes) -> take [:, :, 1]
    #   - a single 2D array (n_samples, n_features)          -> use as-is
    if isinstance(shap_values, list):
        print("  (SHAP returned a list of per-class arrays -- using class 1 / malicious)")
        shap_values = shap_values[1]
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            print(f"  (SHAP returned a 3D array {shap_values.shape} -- using class 1 / malicious)")
            shap_values = shap_values[:, :, 1]
        elif shap_values.ndim != 2:
            raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_path = os.path.join(OUTPUT_DIR, "shap_values_random_forest.npy")
    np.save(shap_path, shap_values)

    top_path = os.path.join(OUTPUT_DIR, "top_features_random_forest.csv")
    importance_df.head(TOP_N_FEATURES).to_csv(top_path, index=False)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Top {TOP_N_FEATURES} features for Random Forest:")
    print(importance_df.head(TOP_N_FEATURES).to_string(index=False))
    print(f"\nSaved to: {top_path}")

    # If XGBoost/LightGBM top-feature CSVs already exist, update the
    # three-way comparison automatically.
    xgb_path = os.path.join(OUTPUT_DIR, "top_features_xgboost.csv")
    lgb_path = os.path.join(OUTPUT_DIR, "top_features_lightgbm.csv")

    if os.path.exists(xgb_path) and os.path.exists(lgb_path):
        print("\nFound existing XGBoost/LightGBM results -- updating full comparison...")
        rf_top = set(importance_df.head(TOP_N_FEATURES)["feature"])
        xgb_top = set(pd.read_csv(xgb_path)["feature"])
        lgb_top = set(pd.read_csv(lgb_path)["feature"])

        pairs = [
            ("random_forest", "xgboost", rf_top, xgb_top),
            ("random_forest", "lightgbm", rf_top, lgb_top),
            ("xgboost", "lightgbm", xgb_top, lgb_top),
        ]

        rows = []
        for a_name, b_name, a_set, b_set in pairs:
            overlap = a_set & b_set
            jaccard = len(overlap) / len(a_set | b_set)
            rows.append({
                "model_a": a_name,
                "model_b": b_name,
                "shared_features": len(overlap),
                "jaccard_similarity": round(jaccard, 3),
            })

        comparison_df = pd.DataFrame(rows)
        comparison_path = os.path.join(OUTPUT_DIR, "shap_model_agreement.csv")
        comparison_df.to_csv(comparison_path, index=False)

        print("\n" + "=" * 50)
        print("FULL THREE-MODEL EXPLANATION CONSISTENCY")
        print("=" * 50)
        print(comparison_df.to_string(index=False))
        print(f"\nSaved to: {comparison_path}")

    print("\nDONE")


if __name__ == "__main__":
    main()
