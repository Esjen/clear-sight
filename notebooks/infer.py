"""
infer.py

The core prediction + explanation logic for ClearSight. This script
takes a single file's feature vector (2381 numbers) and a model name,
and returns:
  - a prediction (malicious / benign)
  - a confidence score
  - the top contributing features, according to SHAP

This is designed to be called two ways:
  1. Directly from the command line, for testing (see bottom of file)
  2. As a subprocess from the Node.js backend, which will call this
     script and read its JSON output

Run a quick test with:
    python infer.py --test

This loads one real file from your test set and shows what the output
looks like, without needing Node.js at all yet.
"""

import sys
import json
import pickle
import numpy as np
import argparse
import os

import shap

MODEL_DIR = "../models"
DATA_DIR = "../data/ember_2018"

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

MODEL_FILES = {
    "random_forest": "random_forest_baseline_depth20.pkl",
    "xgboost": "xgboost_baseline.pkl",
    "lightgbm": "lightgbm_baseline.pkl",
}

_model_cache = {}
_explainer_cache = {}


def build_feature_names():
    names = []
    for group_name, size in FEATURE_GROUPS:
        for i in range(size):
            names.append(f"{group_name}_{i}")
    return names


def load_model(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]

    filename = MODEL_FILES.get(model_name)
    if filename is None:
        raise ValueError(f"Unknown model: {model_name}")

    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        model = pickle.load(f)

    _model_cache[model_name] = model
    return model


def get_explainer(model_name):
    if model_name in _explainer_cache:
        return _explainer_cache[model_name]

    model = load_model(model_name)
    explainer = shap.TreeExplainer(model)
    _explainer_cache[model_name] = explainer
    return explainer


def predict_and_explain(feature_vector, model_name, top_n=10):
    """
    feature_vector: a list or array of 2381 numbers
    model_name: one of "random_forest", "xgboost", "lightgbm"
    Returns a dict with prediction, confidence, and top features.
    """
    X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

    model = load_model(model_name)
    proba = model.predict_proba(X)[0]
    prediction = "malicious" if proba[1] >= 0.5 else "benign"
    confidence = float(max(proba))

    explainer = get_explainer(model_name)

    if model_name == "random_forest":
        shap_values = explainer.shap_values(X, check_additivity=False)
    else:
        shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    row_shap = shap_values[0]
    feature_names = build_feature_names()

    order = np.argsort(-np.abs(row_shap))[:top_n]
    top_features = [
        {
            "feature": feature_names[i],
            "shap_value": float(row_shap[i]),
            "direction": "pushes toward malicious" if row_shap[i] > 0 else "pushes toward benign",
        }
        for i in order
    ]

    return {
        "model": model_name,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probability_malicious": round(float(proba[1]), 4),
        "top_features": top_features,
    }


def load_test_sample(row_index):
    """Load one row's feature vector and true label from the test set,
    for quick testing without needing to submit a real file."""
    X = np.load(os.path.join(DATA_DIR, "X_test.npy"), mmap_mode="r")
    y = np.load(os.path.join(DATA_DIR, "y_test.npy"), mmap_mode="r")
    return X[row_index].tolist(), int(y[row_index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                         help="Run a quick test using a sample from the test set")
    parser.add_argument("--row", type=int, default=0,
                         help="Which test-set row to use with --test")
    parser.add_argument("--model", type=str, default="lightgbm",
                         choices=["random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--input", type=str,
                         help="Path to a JSON file containing a feature vector "
                              "(used when called from Node.js)")
    args = parser.parse_args()

    if args.test:
        feature_vector, true_label = load_test_sample(args.row)
        result = predict_and_explain(feature_vector, args.model)
        result["true_label"] = "malicious" if true_label == 1 else "benign"
        print(json.dumps(result, indent=2))
        return

    if args.input:
        with open(args.input, "r") as f:
            payload = json.load(f)
        model_name = payload.get("model", "lightgbm")

        if "row" in payload:
            feature_vector, true_label = load_test_sample(payload["row"])
            result = predict_and_explain(feature_vector, model_name)
            result["true_label"] = "malicious" if true_label == 1 else "benign"
        else:
            feature_vector = payload["features"]
            result = predict_and_explain(feature_vector, model_name)

        print(json.dumps(result))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
