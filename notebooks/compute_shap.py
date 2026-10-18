"""
compute_shap.py

Computes SHAP (SHapley Additive exPlanations) values for all three
trained models -- Random Forest, XGBoost, and LightGBM -- on the same
sample of test files, then compares which features each model relies
on most. This is the core of your project's actual research
contribution: not just "how accurate is each model" but "do these
models explain their decisions the same way, or differently."

Why we use a SAMPLE instead of the full test set:
----------------------------------------------------
Computing SHAP values is much more expensive than a normal prediction.
Doing it for all 200,000 test rows x 2381 features x 3 models would
take a very long time and a lot of memory. Using a representative
sample (default: 1000 files, balanced between malicious and benign) is
standard practice and is exactly what you already stated in your
Chapter 1 Scope section ("SHAP is computed on a sample for
performance").

What this script does:
------------------------
1. Loads your three trained models
2. Loads a balanced random sample from the test set
3. Computes SHAP values for each model using TreeExplainer (the fast,
   exact SHAP method built specifically for tree-based models)
4. Identifies each model's top 20 most important features (by mean
   absolute SHAP value)
5. Compares how much overlap exists between each pair of models' top
   features (this is your "explanation consistency" finding)
6. Saves everything to disk: SHAP values, top-feature tables, and a
   feature-name mapping so results are labeled meaningfully instead of
   just "feature 1583"

Run with:
    python compute_shap.py

Needs: pip install shap
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

SAMPLE_SIZE = 1000       # total files to sample (split evenly by label)
TOP_N_FEATURES = 20      # how many top features to report per model
RANDOM_SEED = 42

# ---------------------------------------------------------------------
# EMBER feature vector layout (feature_version=2). The 2381-length
# vector is a concatenation of these feature groups, in this order.
# This lets us label SHAP results with meaningful group names instead
# of raw indices like "feature 1583".
# ---------------------------------------------------------------------
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
    """Load a balanced random sample from the (large) test set without
    reading the whole thing into memory at once."""
    y_full = np.load(y_path, mmap_mode="r")
    n_total = len(y_full)

    rng = np.random.default_rng(seed)

    benign_idx = np.where(y_full[:] == 0)[0]
    malicious_idx = np.where(y_full[:] == 1)[0]

    n_each = sample_size // 2
    chosen_benign = rng.choice(benign_idx, size=n_each, replace=False)
    chosen_malicious = rng.choice(malicious_idx, size=n_each, replace=False)

    chosen = np.sort(np.concatenate([chosen_benign, chosen_malicious]))

    X_full = np.load(X_path, mmap_mode="r")
    X_sample = np.array(X_full[chosen])  # load only the chosen rows into RAM
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

    # Some models return a list (one array per class); we want the
    # "malicious" class (class 1) explanations.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f} seconds.")

    # Mean absolute SHAP value per feature = overall importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    # Save full SHAP values array for this model
    shap_path = os.path.join(out_dir, f"shap_values_{model_name}.npy")
    np.save(shap_path, shap_values)

    # Save top features table
    top_path = os.path.join(out_dir, f"top_features_{model_name}.csv")
    importance_df.head(TOP_N_FEATURES).to_csv(top_path, index=False)

    print(f"  Top 5 features for {model_name}:")
    print(importance_df.head(5).to_string(index=False))

    return importance_df


def compare_top_features(importance_dfs, out_dir):
    """Compare how much overlap exists between each pair of models'
    top-N most important features -- this is your explanation
    consistency finding."""
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
    print(f"  {len(feature_names)} feature names built.")

    print(f"\nLoading a balanced sample of {SAMPLE_SIZE} test files...")
    X_path = os.path.join(DATA_DIR, "X_test.npy")
    y_path = os.path.join(DATA_DIR, "y_test.npy")
    X_sample, y_sample, chosen_idx = load_sample(X_path, y_path, SAMPLE_SIZE, RANDOM_SEED)
    print(f"  Sample shape: {X_sample.shape}, labels: {np.bincount(y_sample.astype(int))}")

    # Save which rows were sampled, so results are reproducible / referenceable
    np.save(os.path.join(OUTPUT_DIR, "shap_sample_indices.npy"), chosen_idx)

    models = {
        "random_forest": os.path.join(MODEL_DIR, "random_forest_baseline.pkl"),
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
