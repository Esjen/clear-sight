"""
train_lightgbm.py

Trains a LightGBM classifier on the same flattened EMBER data used for
your Random Forest and XGBoost baselines, completing the three-model
comparison described in your project objectives.

What it does:
-------------
1. Loads the same X_train/y_train/X_test/y_test .npy files
2. Trains a LightGBM classifier
3. Evaluates with the same metrics as the other two scripts so all
   three models can be directly compared:
     - Accuracy, Precision, Recall, F1
     - AUC-ROC, AUC-PR
     - Confusion matrix
4. Saves the trained model to disk

Run with:
    python train_lightgbm.py

LightGBM is generally the fastest of the three models at this scale
(often faster than XGBoost), so this should complete relatively quickly.
"""

import numpy as np
import time
import pickle
import os

import lightgbm as lgb
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

    print("\nTraining LightGBM...")
    start = time.time()

    clf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=-1,          # -1 means no limit, LightGBM controls via num_leaves
        num_leaves=63,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
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

    model_path = os.path.join(MODEL_DIR, "lightgbm_baseline.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
