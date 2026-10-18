"""
train_random_forest.py

Trains a Random Forest classifier on the flattened EMBER training data
(X_train.npy / y_train.npy) and evaluates it on the held-out test set
(X_test.npy / y_test.npy).

This is your BASELINE model -- the reference point you'll later compare
XGBoost and LightGBM against.

What it does:
-------------
1. Loads the numeric feature matrices you built with flatten_features.py
   and flatten_test_features.py
2. Trains a RandomForestClassifier
3. Evaluates on the test set with several metrics:
     - Accuracy
     - Precision / Recall / F1 (per class)
     - AUC-ROC and AUC-PR (important for imbalanced/real-world settings,
       even though this particular test set is balanced)
     - Confusion matrix
4. Saves the trained model to disk so you don't have to retrain it
   every time you want to use it (e.g. for SHAP later)

Run with:
    python train_random_forest.py

This will take a while to train (hundreds of thousands of rows, 2381
features each) -- expect anywhere from several minutes to ~30+ minutes
depending on your machine. It will NOT need as much memory as the
flattening step, since Random Forest processes data more efficiently,
but keep Task Manager open just in case.
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

    print("\nTraining Random Forest... (this will take a while)")
    start = time.time()

    # n_jobs=-1 uses all CPU cores to speed up training.
    # class_weight="balanced" is a safeguard in case you later train on
    # imbalanced data (your current split is already 50/50, but this
    # keeps the code robust for future experiments).
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
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
    y_proba = clf.predict_proba(X_test)[:, 1]  # probability of "malicious"

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary"
    )
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("RESULTS")
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

    # Save the trained model for later use (e.g. SHAP explainability)
    model_path = os.path.join(MODEL_DIR, "random_forest_baseline.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
