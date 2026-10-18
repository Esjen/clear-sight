"""
compare.py

Per-file model comparison for ClearSight. For one test file it returns, for
each of the three models: the predicted probability, the model's top features,
and which of those features the OTHER models also rank in their top 20. It also
returns the pairwise Jaccard similarity of the three models' top-20 features
for that single file.

It reads the SHAP arrays that build_comparison_data.py produced, so it does NOT
load any trained model and answers in well under a second. It only works for the
1,000 files in the shared sample.

Usage (the Node backend calls this):
    python compare.py --row 1234
    python compare.py --random
"""

import argparse
import json
import os

import numpy as np

MODELS = ["random_forest", "xgboost", "lightgbm"]
SHAP_FILES = {
    "random_forest": "shap_values_random_forest_aligned.npy",
    "xgboost": "shap_values_xgboost.npy",
    "lightgbm": "shap_values_lightgbm.npy",
}
FEATURE_GROUPS = [("ByteHistogram", 256), ("ByteEntropyHistogram", 256), ("StringExtractor", 104),
                  ("GeneralFileInfo", 10), ("HeaderFileInfo", 62), ("SectionInfo", 255),
                  ("ImportsInfo", 1280), ("ExportsInfo", 128), ("DataDirectories", 30)]
TOP_SHOWN = 10
TOP_JACCARD = 20


def feature_names():
    return [f"{g}_{i}" for g, size in FEATURE_GROUPS for i in range(size)]


def class1(arr):
    if isinstance(arr, list):
        arr = arr[1]
    return arr[:, :, 1] if arr.ndim == 3 else arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--results-dir", default="../results")
    args = ap.parse_args()
    res = args.results_dir

    try:
        preds = np.load(os.path.join(res, "sample_predictions.npz"))
        indices = preds["indices"]
        shap_arrays = {m: class1(np.load(os.path.join(res, SHAP_FILES[m]), mmap_mode="r")) for m in MODELS}
    except FileNotFoundError as e:
        print(json.dumps({"error": "data_missing",
                          "message": f"Comparison data not found ({e.filename}). Run build_comparison_data.py first."}))
        return

    if args.random:
        row = int(np.random.default_rng().choice(indices))
    elif args.row is not None:
        row = args.row
    else:
        print(json.dumps({"error": "bad_request", "message": "Provide --row or --random."}))
        return

    pos = np.where(indices == row)[0]
    if len(pos) == 0:
        print(json.dumps({"error": "not_in_sample",
                          "message": f"Test file {row} is not in the 1,000-file explanation sample. "
                                     "Use the 'Random sampled file' button."}))
        return
    pos = int(pos[0])

    names = feature_names()
    orders, top20 = {}, {}
    for m in MODELS:
        v = np.asarray(shap_arrays[m][pos], dtype=np.float64)
        orders[m] = np.argsort(-np.abs(v), kind="stable")
        top20[m] = set(orders[m][:TOP_JACCARD].tolist())

    out_models = {}
    for m in MODELS:
        v = np.asarray(shap_arrays[m][pos], dtype=np.float64)
        p = float(preds[f"p_{m}"][pos])
        feats = []
        for rank, j in enumerate(orders[m][:TOP_SHOWN].tolist(), 1):
            feats.append({
                "rank": rank,
                "feature": names[j],
                "shap_value": float(v[j]),
                "direction": "pushes toward malicious" if v[j] > 0 else "pushes toward benign",
                "shared_with": [o for o in MODELS if o != m and j in top20[o]],
            })
        out_models[m] = {
            "probability_malicious": round(p, 4),
            "prediction": "malicious" if p >= 0.5 else "benign",
            "top_features": feats,
        }

    def jac(a, b):
        return round(len(top20[a] & top20[b]) / len(top20[a] | top20[b]), 3)

    out = {
        "row": row,
        "true_label": "malicious" if float(preds["y_true"][pos]) == 1 else "benign",
        "top_k_for_jaccard": TOP_JACCARD,
        "models": out_models,
        "pairwise_jaccard": {
            "xgboost_vs_lightgbm": jac("xgboost", "lightgbm"),
            "random_forest_vs_xgboost": jac("random_forest", "xgboost"),
            "random_forest_vs_lightgbm": jac("random_forest", "lightgbm"),
        },
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
