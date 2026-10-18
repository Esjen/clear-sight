"""
bootstrap_agreement.py

Turns the three SHAP arrays (all computed on the SAME sampled files) into
Jaccard agreement values WITH confidence intervals.

How the bootstrap works
-----------------------
Each round, the sampled files are resampled with replacement (same size as the
original sample). The SAME resample is applied to all three models. For each
model the mean absolute SHAP value of every feature is recomputed, the top-k
features are taken, and the Jaccard similarity between each pair of models is
calculated. Repeating this thousands of times shows how much each Jaccard value
moves just because a different set of files was explained.

What it does NOT cover
----------------------
The three trained models are fixed. The bootstrap captures sampling variability
of the explanation rankings, not the variability you would get from retraining
each model with a different random seed. State this when you report the result.

Also reported
-------------
- Sensitivity to k (top 10, 20, 50 features).
- The "H2 check": how often the boosting pair (XGBoost-LightGBM) agrees MORE
  than both Random Forest pairs, and the gap with its confidence interval.

Run from the notebooks folder:
    python bootstrap_agreement.py
Takes a few seconds. Writes into ../results:
    agreement_summary.json  (read by ClearSight)
    jaccard_bootstrap_summary.csv
    top_features_<model>_aligned.csv
    fig_jaccard_bootstrap.png (only if matplotlib is installed)
"""

import argparse
import csv
import json
import os

import numpy as np

MODELS = ["random_forest", "xgboost", "lightgbm"]
LABEL = {"random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
SHAP_FILES = {
    "random_forest": "shap_values_random_forest_aligned.npy",
    "xgboost": "shap_values_xgboost.npy",
    "lightgbm": "shap_values_lightgbm.npy",
}
PAIRS = [("xgboost", "lightgbm"), ("random_forest", "xgboost"), ("random_forest", "lightgbm")]

FEATURE_GROUPS = [("ByteHistogram", 256), ("ByteEntropyHistogram", 256), ("StringExtractor", 104),
                  ("GeneralFileInfo", 10), ("HeaderFileInfo", 62), ("SectionInfo", 255),
                  ("ImportsInfo", 1280), ("ExportsInfo", 128), ("DataDirectories", 30)]


def feature_names():
    return [f"{g}_{i}" for g, size in FEATURE_GROUPS for i in range(size)]


def load_class1(path):
    arr = np.load(path)
    if isinstance(arr, list):
        arr = arr[1]
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    return arr


def top_k(mean_abs, k):
    return set(np.argsort(-mean_abs, kind="stable")[:k].tolist())


def jaccard(a, b):
    return len(a & b) / len(a | b)


def pair_name(a, b):
    return f"{LABEL[a]} vs {LABEL[b]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--primary-k", type=int, default=20)
    args = ap.parse_args()
    res = args.results_dir

    A = {}
    for m in MODELS:
        p = os.path.join(res, SHAP_FILES[m])
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p}. Run build_comparison_data.py first.")
        A[m] = np.abs(load_class1(p)).astype(np.float64)
    n, n_feat = A[MODELS[0]].shape
    for m in MODELS:
        if A[m].shape != (n, n_feat):
            raise SystemExit(f"{m} array has shape {A[m].shape}, expected {(n, n_feat)}. Arrays must cover the same files.")
    print(f"Loaded {n} files x {n_feat} features for each of the 3 models.")

    names = feature_names()
    # Top features table (full sample), useful for the report
    for m in MODELS:
        mean_abs = A[m].mean(axis=0)
        order = np.argsort(-mean_abs, kind="stable")[:20]
        with open(os.path.join(res, f"top_features_{m}_aligned.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["rank", "feature", "mean_abs_shap"])
            for r, j in enumerate(order, 1):
                w.writerow([r, names[j], f"{mean_abs[j]:.6f}"])

    # Point estimates on the full sample
    point_mean = {m: A[m].mean(axis=0) for m in MODELS}
    point = {}
    for k in args.ks:
        sets = {m: top_k(point_mean[m], k) for m in MODELS}
        for a, b in PAIRS:
            point[(k, a, b)] = (jaccard(sets[a], sets[b]), len(sets[a] & sets[b]))

    # Bootstrap
    rng = np.random.default_rng(args.seed)
    reps = {(k, a, b): np.empty(args.n_boot) for k in args.ks for a, b in PAIRS}
    for bi in range(args.n_boot):
        counts = rng.multinomial(n, np.full(n, 1.0 / n))
        w = counts / n
        means = {m: w @ A[m] for m in MODELS}
        for k in args.ks:
            sets = {m: top_k(means[m], k) for m in MODELS}
            for a, b in PAIRS:
                reps[(k, a, b)][bi] = jaccard(sets[a], sets[b])
    print(f"Bootstrap finished: {args.n_boot} resamples, seed {args.seed}.")

    def ci(x):
        return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

    rows = []
    for k in args.ks:
        for a, b in PAIRS:
            lo, hi = ci(reps[(k, a, b)])
            pt, shared = point[(k, a, b)]
            rows.append({"k": k, "pair": pair_name(a, b), "shared": shared, "jaccard": round(pt, 3),
                         "boot_mean": round(float(reps[(k, a, b)].mean()), 3),
                         "ci_low": round(lo, 3), "ci_high": round(hi, 3)})

    # H2 check at the primary k
    pk = args.primary_k
    jbb = reps[(pk, "xgboost", "lightgbm")]
    jrx = reps[(pk, "random_forest", "xgboost")]
    jrl = reps[(pk, "random_forest", "lightgbm")]
    gap = jbb - (jrx + jrl) / 2
    glo, ghi = ci(gap)
    h2 = {"k": pk,
          "fraction_boosting_pair_higher_than_both_rf_pairs": round(float(np.mean((jbb > jrx) & (jbb > jrl))), 4),
          "gap_boosting_minus_mean_rf_pairs": round(float(gap.mean()), 3),
          "gap_ci_low": round(glo, 3), "gap_ci_high": round(ghi, 3)}

    summary = {
        "n_files": int(n), "n_bootstrap": args.n_boot, "seed": args.seed, "primary_k": pk,
        "scope": ("Bootstrap over the sampled files only. The three trained models are fixed, so this captures "
                  "sampling variability of the explanation rankings, not variability from retraining the models."),
        "pairs": [r for r in rows if r["k"] == pk],
        "sensitivity": rows,
        "h2": h2,
    }
    json.dump(summary, open(os.path.join(res, "agreement_summary.json"), "w"), indent=2)
    with open(os.path.join(res, "jaccard_bootstrap_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\nJaccard similarity of top-{pk} SHAP features (95% bootstrap CI)")
    for r in summary["pairs"]:
        print(f"  {r['pair']:<28} {r['jaccard']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]   shared {r['shared']}/{pk}")
    print(f"\nH2 check: boosting pair higher than BOTH Random Forest pairs in "
          f"{h2['fraction_boosting_pair_higher_than_both_rf_pairs'] * 100:.1f}% of resamples; "
          f"gap {h2['gap_boosting_minus_mean_rf_pairs']:.3f} [{h2['gap_ci_low']:.3f}, {h2['gap_ci_high']:.3f}]")
    print("\nSensitivity to k:")
    for r in rows:
        print(f"  k={r['k']:<3} {r['pair']:<28} {r['jaccard']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pr = summary["pairs"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = np.arange(len(pr)); vals = [r["jaccard"] for r in pr]
        err = [[r["jaccard"] - r["ci_low"] for r in pr], [r["ci_high"] - r["jaccard"] for r in pr]]
        ax.bar(xs, vals, yerr=err, capsize=6, color=["#55A868", "#C44E52", "#C44E52"])
        ax.set_xticks(xs); ax.set_xticklabels([r["pair"].replace(" vs ", "\nvs ") for r in pr])
        ax.set_ylim(0, 1); ax.set_ylabel(f"Jaccard similarity (top-{pk} features)")
        ax.set_title("Explanation agreement with 95% bootstrap intervals")
        for x, v in zip(xs, vals): ax.text(x, v + 0.04, f"{v:.3f}", ha="center", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
        fig.savefig(os.path.join(res, "fig_jaccard_bootstrap.png"), dpi=200)
        print("\nSaved fig_jaccard_bootstrap.png")
    except ImportError:
        print("\n(matplotlib not installed; skipped the chart. The CSV and JSON are complete.)")


if __name__ == "__main__":
    main()
