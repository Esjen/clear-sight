"""
compute_shap_fast.py

Same as compute_shap.py, but computes SHAP values only for XGBoost and
LightGBM, skipping Random Forest for now.

Why: Random Forest was trained with unrestricted tree depth
(max_depth=None), so its trees can grow very deep with 600,000 training
rows. SHAP's TreeExplainer algorithm slows down sharply as tree depth
increases, so it can take a very long time (or feel "stuck") for this
particular Random Forest model. XGBoost and LightGBM use shallower
trees by default, so SHAP is fast for them.

This script gets you real, usable results now. A separate, smaller-
sample script for Random Forest is provided separately if you still
want it -- otherwise, noting this computational constraint in your
Limitations section is a legitimate, honest thing to say.

Run with:
    python compute_shap_fast.py
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

SAMPLE_SIZE = 1000
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


def compute_shap_for_model(model_name, model_path, X_sample, feature_names, out_dir):
    print(f"\n--- {model_name} ---")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("  Building TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    print(f"  Computing SHAP values for {X_sample.shape[0]} samples...")
    start = time.time()
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f} seconds.")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_path = os.path.join(out_dir, f"shap_values_{model_name}.npy")
    np.save(shap_path, shap_values)

    top_path = os.path.join(out_dir, f"top_features_{model_name}.csv")
    importance_df.head(TOP_N_FEATURES).to_csv(top_path, index=False)

    print(f"  Top 5 features for {model_name}:")
    print(importance_df.head(5).to_string(index=False))

    return importance_df


def compare_top_features(importance_dfs, out_dir):
    top_sets = {
        name: set(df.head(TOP_N_FEATURES)["feature"])
        for name, df in importance_dfs.items()
    }

    names = list(top_sets.keys())
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = top_sets[a] & top_sets[b]
            jaccard = len(overlap) / len(top_sets[a] | top_sets[b])
            rows.append({
                "model_a": a,
                "model_b": b,
                "shared_features": len(overlap),
                "jaccard_similarity": round(jaccard, 3),
            })

    comparison_df = pd.DataFrame(rows)
    comparison_path = os.path.join(out_dir, "shap_model_agreement.csv")
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 50)
    print("EXPLANATION CONSISTENCY ACROSS MODELS")
    print("=" * 50)
    print(f"(Top {TOP_N_FEATURES} features per model, Jaccard similarity)")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved to: {comparison_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building feature name mapping...")
    feature_names = build_feature_names()

    print(f"\nLoading a balanced sample of {SAMPLE_SIZE} test files...")
    X_path = os.path.join(DATA_DIR, "X_test.npy")
    y_path = os.path.join(DATA_DIR, "y_test.npy")
    X_sample, y_sample, chosen_idx = load_sample(X_path, y_path, SAMPLE_SIZE, RANDOM_SEED)
    print(f"  Sample shape: {X_sample.shape}, labels: {np.bincount(y_sample.astype(int))}")

    np.save(os.path.join(OUTPUT_DIR, "shap_sample_indices.npy"), chosen_idx)

    # NOTE: random_forest intentionally excluded here -- see docstring
    models = {
        "xgboost": os.path.join(MODEL_DIR, "xgboost_baseline.pkl"),
        "lightgbm": os.path.join(MODEL_DIR, "lightgbm_baseline.pkl"),
    }

    importance_dfs = {}
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"\nWARNING: {model_path} not found, skipping {model_name}.")
            continue
        importance_dfs[model_name] = compute_shap_for_model(
            model_name, model_path, X_sample, feature_names, OUTPUT_DIR
        )

    if len(importance_dfs) >= 2:
        compare_top_features(importance_dfs, OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("ALL DONE")
    print("=" * 50)
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
