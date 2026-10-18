"""
train_random_forest_v2.py

Retrains Random Forest with a bounded max_depth. The original version
used max_depth=None, letting trees grow unrestricted on 600,000 rows --
this almost certainly produced extremely large, deep trees, which is
why SHAP computation kept crashing regardless of how small we made the
sample or background set. SHAP's TreeExplainer cost scales with tree
size, not just how many samples you're explaining.

Bounding tree depth is standard practice specifically to keep SHAP
computation tractable, and is a defensible, documented methodological
choice -- not a compromise on your results. We expect accuracy to be
very similar to (possibly even better than, due to reduced overfitting)
the original unrestricted-depth version.

Run with:
    python train_random_forest_v2.py

This should train notably faster than the original 86-minute run,
since shallower trees are cheaper to build.
"""

import numpy as np
import time
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

DATA_DIR = "../data/ember_2018"
MODEL_DIR = "../models"

MAX_DEPTH = 20  # bounded, unlike the original max_depth=None


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading training data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    print("Loading test data...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    print(f"\nTraining Random Forest with max_depth={MAX_DEPTH}...")
    start = time.time()

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=MAX_DEPTH,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
        verbose=1,
    )
    clf.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed / 60:.1f} minutes.")

    print("\nEvaluating on test set...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("RESULTS (bounded-depth Random Forest)")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"AUC-PR:    {auc_pr:.4f}")

    print("\nConfusion matrix:")
    print("               Predicted Benign  Predicted Malicious")
    print(f"Actual Benign       {cm[0][0]:>10}         {cm[0][1]:>10}")
    print(f"Actual Malicious    {cm[1][0]:>10}         {cm[1][1]:>10}")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

    # Save as a NEW file so we don't lose the original in case we need
    # to compare later.
    model_path = os.path.join(MODEL_DIR, "random_forest_baseline_depth20.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
