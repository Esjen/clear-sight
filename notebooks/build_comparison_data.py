"""
build_comparison_data.py

Makes the explanation comparison fair. Until now, XGBoost and LightGBM were
explained on 1,000 test files, but Random Forest was explained on a different,
smaller sample of 100 files. This script fixes that by explaining the SAME
1,000 files for Random Forest too.

What it does
------------
1. Loads the 1,000 sampled file indices that compute_shap_fast.py already used
   (results/shap_sample_indices.npy), so all three models see identical files.
2. Checks that the existing XGBoost and LightGBM SHAP arrays match that sample.
3. Computes Random Forest SHAP for those files in CHUNKS and saves each chunk
   to disk as it finishes. If the run stops (power cut, sleep, crash), run the
   same command again and it continues from the last finished chunk.
4. Runs a local-accuracy check on the first chunk. SHAP values for one file
   should add up to (model output - baseline). Earlier, an unrestricted Random
   Forest produced wildly wrong values, so this check protects the results.
5. Saves each model's predicted probability for the sampled files, so that
   ClearSight can show a per-file comparison without loading three big models.

Run from the notebooks folder, with the venv active:
    python build_comparison_data.py

Expect roughly one hour for 1,000 files (the earlier 100-file run took about
5 minutes). Plug in the laptop and stop it from sleeping. Safe to stop with
Ctrl+C and re-run.
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import shap

MODEL_FILES = {
    "random_forest": "random_forest_baseline_depth20.pkl",
    "xgboost": "xgboost_baseline.pkl",
    "lightgbm": "lightgbm_baseline.pkl",
}


def load_class1(arr):
    """Normalise whatever SHAP returned into one (n_files, n_features) array
    for the 'malicious' class."""
    if isinstance(arr, list):
        arr = arr[1]
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    return arr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data/ember_2018")
    ap.add_argument("--model-dir", default="../models")
    ap.add_argument("--results-dir", default="../results")
    ap.add_argument("--chunk", type=int, default=50, help="files per saved chunk")
    ap.add_argument("--n", type=int, default=1000, help="expected number of sampled files")
    args = ap.parse_args()

    res = args.results_dir
    chunk_dir = os.path.join(res, "rf_shap_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    # 1. The shared sample --------------------------------------------------
    idx_path = os.path.join(res, "shap_sample_indices.npy")
    if not os.path.exists(idx_path):
        raise SystemExit(f"Cannot find {idx_path}. Run compute_shap_fast.py first; it creates the shared sample.")
    indices = np.load(idx_path)
    if len(indices) != args.n:
        raise SystemExit(f"Expected {args.n} sampled files but found {len(indices)} in {idx_path}.")
    if not np.all(indices[:-1] < indices[1:]):
        raise SystemExit("Sample indices are not sorted; the existing SHAP arrays assume sorted order.")

    X_all = np.load(os.path.join(args.data_dir, "X_test.npy"), mmap_mode="r")
    y_all = np.load(os.path.join(args.data_dir, "y_test.npy"), mmap_mode="r")
    X = np.array(X_all[indices], dtype=np.float32)
    y_true = np.array(y_all[indices], dtype=np.float32)
    n_features = X.shape[1]
    print(f"Shared sample: {len(indices)} files ({int(y_true.sum())} malicious, {int((1 - y_true).sum())} benign)")

    # 2. Existing boosting-model SHAP arrays must match -----------------------
    for name in ("xgboost", "lightgbm"):
        p = os.path.join(res, f"shap_values_{name}.npy")
        if not os.path.exists(p):
            raise SystemExit(f"Cannot find {p}. Run compute_shap_fast.py first.")
        shape = load_class1(np.load(p, mmap_mode="r")).shape
        if shape != (args.n, n_features):
            raise SystemExit(f"{p} has shape {shape}, expected {(args.n, n_features)}. It was not computed on the shared sample.")
    print("XGBoost and LightGBM SHAP arrays match the shared sample.")

    # 3. Random Forest SHAP in resumable chunks ------------------------------
    with open(os.path.join(args.model_dir, MODEL_FILES["random_forest"]), "rb") as f:
        rf = pickle.load(f)
    print("Building Random Forest TreeExplainer...")
    explainer = shap.TreeExplainer(rf)

    starts = list(range(0, args.n, args.chunk))
    todo = [s for s in starts if not os.path.exists(os.path.join(chunk_dir, f"chunk_{s:05d}.npy"))]
    print(f"{len(starts) - len(todo)} of {len(starts)} chunks already done; {len(todo)} to go.")
    t_start = time.time()
    for done, s in enumerate(todo, 1):
        t0 = time.time()
        e = min(s + args.chunk, args.n)
        sv = load_class1(explainer.shap_values(X[s:e], check_additivity=False))
        np.save(os.path.join(chunk_dir, f"chunk_{s:05d}.npy"), sv)
        per = (time.time() - t_start) / done
        eta = per * (len(todo) - done)
        print(f"  chunk {s}-{e - 1} done in {time.time() - t0:.0f}s | max |SHAP| {np.abs(sv).max():.4f} | "
              f"ETA {eta / 60:.0f} min", flush=True)

    # 4. Local-accuracy check on the first chunk ------------------------------
    check_path = os.path.join(res, "rf_additivity_check.json")
    if not os.path.exists(check_path):
        first = np.load(os.path.join(chunk_dir, f"chunk_{0:05d}.npy"))
        m = first.shape[0]
        ev = np.atleast_1d(explainer.expected_value)
        base = float(ev[-1])
        recon = first.sum(axis=1) + base
        actual = rf.predict_proba(X[:m])[:, 1]
        diff = float(np.abs(recon - actual).max())
        out = {"files_checked": int(m), "baseline": base, "max_abs_difference": diff,
               "verdict": "OK" if diff < 0.02 else "FAIL: SHAP values do not add up; do not trust the Random Forest results"}
        json.dump(out, open(check_path, "w"), indent=2)
        print(f"Local-accuracy check (first {m} files): max difference {diff:.5f} -> {out['verdict']}")
    else:
        print("Local-accuracy check already recorded:", json.load(open(check_path))["verdict"])

    # 5. Assemble the aligned Random Forest array ------------------------------
    rf_all = np.concatenate([np.load(os.path.join(chunk_dir, f"chunk_{s:05d}.npy")) for s in starts])
    assert rf_all.shape == (args.n, n_features), rf_all.shape
    np.save(os.path.join(res, "shap_values_random_forest_aligned.npy"), rf_all)
    print("Saved shap_values_random_forest_aligned.npy", rf_all.shape)

    # 6. Predictions on the shared sample (used by ClearSight) -----------------
    probs = {}
    for name, fname in MODEL_FILES.items():
        model = rf if name == "random_forest" else pickle.load(open(os.path.join(args.model_dir, fname), "rb"))
        probs[name] = model.predict_proba(X)[:, 1].astype(np.float32)
    np.savez(os.path.join(res, "sample_predictions.npz"), indices=indices, y_true=y_true,
             p_random_forest=probs["random_forest"], p_xgboost=probs["xgboost"], p_lightgbm=probs["lightgbm"])
    print("Saved sample_predictions.npz")
    print("\nDONE. Next: python bootstrap_agreement.py")


if __name__ == "__main__":
    main()
