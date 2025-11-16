from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features import FeatureExtractor

RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    """Load dataset and validate required columns."""
    df = pd.read_csv(path)
    required_cols = {"system_prompt", "user_prompt", "model_output", "label"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def build_X_y_batch(df: pd.DataFrame, fe: FeatureExtractor, batch_size: int = 32):
    """
    Build feature matrix using the batched feature extractor.

    This is significantly faster than calling `build_features` row by row.
    """
    print("[Train] Building feature matrix (batch mode)...")

    samples = [
        (row["system_prompt"], row["user_prompt"], row["model_output"])
        for _, row in df.iterrows()
    ]

    X = fe.build_features_batch(samples, batch_size=batch_size)
    y = df["label"].values.astype(int)

    return X, y


def train_and_evaluate(args):
    """End-to-end training and validation pipeline."""
    data_path = Path(args.data)
    model_out = Path(args.out)

    print(f"[Train] Loading data from {data_path} ...")
    df = load_dataset(data_path)
    print(f"[Train] Dataset size: {len(df)} samples")
    print(f"[Train] Label distribution:\n{df['label'].value_counts()}")

    # Feature extractor
    fe = FeatureExtractor(embed_model_name=args.embed_model)

    # Build features in batch mode
    X, y = build_X_y_batch(df, fe, batch_size=args.batch_size)
    print(f"[Train] Feature matrix shape: {X.shape}")

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_ratio,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[Train] Train size: {len(X_train)}, Val size: {len(X_val)}")

    # Build classifier pipeline
    print("[Train] Training classifier (Logistic Regression)...")
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=500,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",  # handle class imbalance
                    C=args.reg_strength,       # regularization strength
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    # ========== Validation metrics ==========
    print("\n" + "=" * 60)
    print("[Train] Validation Performance")
    print("=" * 60)

    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, digits=4))

    # ROC-AUC
    roc_auc = roc_auc_score(y_val, y_prob)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # PR-AUC
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")

    # Find best threshold (max F1)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"\nBest threshold (max F1): {best_threshold:.4f}")
    print(f"  Precision: {precision[best_idx]:.4f}")
    print(f"  Recall: {recall[best_idx]:.4f}")
    print(f"  F1: {f1_scores[best_idx]:.4f}")

    # ========== Cross-validation (optional) ==========
    if args.cross_validate:
        print("\n" + "=" * 60)
        print("[Train] Cross-Validation (5-fold)")
        print("=" * 60)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1")
        print(f"F1 scores: {cv_scores}")
        print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # ========== Save model bundle ==========
    model_out.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "feature_extractor_model_name": fe.model_name,
        "classifier": clf,
        "best_threshold": best_threshold,  # store the best threshold
        "feature_names": {
            "embedding_dim": 384 * 3,  # all-MiniLM-L6-v2 (3 segments)
            "numeric_features": [
                "up_len",
                "mo_len",
                "deny_count",
                "sensitive_count",
                "leakage_score",
                "sp_mo_overlap",
                "up_mo_overlap",
                "instruction_fragments",
            ],
        },
    }
    joblib.dump(bundle, model_out)
    print(f"\n[Train] Model saved to: {model_out}")

    # ========== Feature importance (numeric features only) ==========
    if args.show_feature_importance:
        print("\n" + "=" * 60)
        print("[Train] Feature Importance (Top numeric features)")
        print("=" * 60)

        logreg = clf.named_steps["logreg"]
        coefs = logreg.coef_[0]

        # Focus on numeric feature coefficients
        num_feat_names = bundle["feature_names"]["numeric_features"]
        num_feat_start = len(coefs) - len(num_feat_names)

        feat_importance = list(zip(num_feat_names, coefs[num_feat_start:]))
        feat_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        for name, coef in feat_importance:
            direction = "↑ PASSED" if coef > 0 else "↓ FAILED"
            print(f"  {name:25s}: {coef:+.4f}  {direction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM Judge Classifier (Optimized)")
    parser.add_argument("--data", type=str, default="data/train.csv", help="Training data path")
    parser.add_argument("--out", type=str, default="models/judge_clf.joblib", help="Output model path")
    parser.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2", help="Sentence embedding model")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--reg-strength", type=float, default=1.0, help="Regularization strength (C)")
    parser.add_argument("--cross-validate", action="store_true", help="Run 5-fold cross-validation")
    parser.add_argument(
        "--show-feature-importance",
        action="store_true",
        help="Display feature importance for numeric features",
    )

    args = parser.parse_args()
    train_and_evaluate(args)