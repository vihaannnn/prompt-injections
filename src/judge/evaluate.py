from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
import matplotlib.pyplot as plt

from features import FeatureExtractor


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the test dataset and validate required columns."""
    df = pd.read_csv(path)
    required_cols = {"system_prompt", "user_prompt", "model_output", "label"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def plot_roc_curve(y_true, y_prob, save_path: Path):
    """Plot ROC curve and save to disk."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Eval] ROC curve saved to: {save_path}")


def plot_precision_recall_curve(y_true, y_prob, save_path: Path):
    """Plot Precision–Recall curve and save to disk."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Eval] PR curve saved to: {save_path}")


def analyze_errors(df: pd.DataFrame, y_true, y_pred, save_path: Path):
    """Analyze misclassified cases and write them to a CSV file."""
    df["predicted"] = y_pred
    df["correct"] = (y_true == y_pred)

    # Extract all error cases
    errors = df[~df["correct"]].copy()

    # Categorize error types (False Positive vs False Negative)
    errors["error_type"] = errors.apply(
        lambda row: "False Positive (FAILED→PASSED)" if row["label"] == 0 else "False Negative (PASSED→FAILED)",
        axis=1,
    )

    errors.to_csv(save_path, index=False)
    print(f"[Eval] Error cases saved to: {save_path}")


def main(args):
    data_path = Path(args.data)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== Load test data & model ==========
    print(f"[Eval] Loading test data from {data_path} ...")
    df = load_dataset(data_path)
    print(f"[Eval] Test set size: {len(df)}")

    print(f"[Eval] Loading model from {model_path} ...")
    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    embed_model_name = bundle["feature_extractor_model_name"]
    best_threshold = bundle.get("best_threshold", 0.5)

    # ========== Feature extraction ==========
    fe = FeatureExtractor(embed_model_name=embed_model_name)

    print("[Eval] Building features (batch mode)...")
    samples = [
        (row["system_prompt"], row["user_prompt"], row["model_output"])
        for _, row in df.iterrows()
    ]
    X = fe.build_features_batch(samples, batch_size=args.batch_size)
    y_true = df["label"].values.astype(int)

    # ========== Predict ==========
    print("[Eval] Predicting...")
    y_prob = clf.predict_proba(X)[:, 1]

    # Use the best threshold found during training
    y_pred_optimal = (y_prob >= best_threshold).astype(int)

    # Also compute results with the default 0.5 threshold (for comparison)
    y_pred_default = (y_prob >= 0.5).astype(int)

    # ========== Evaluation results ==========
    print("\n" + "=" * 60)
    print(f"[Eval] Results with DEFAULT threshold (0.5)")
    print("=" * 60)
    print("\nConfusion Matrix:")
    cm_default = confusion_matrix(y_true, y_pred_default)
    print(cm_default)
    print(f"  TN={cm_default[0,0]}, FP={cm_default[0,1]}")
    print(f"  FN={cm_default[1,0]}, TP={cm_default[1,1]}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_default, digits=4))

    print("\n" + "=" * 60)
    print(f"[Eval] Results with OPTIMAL threshold ({best_threshold:.4f})")
    print("=" * 60)
    print("\nConfusion Matrix:")
    cm_optimal = confusion_matrix(y_true, y_pred_optimal)
    print(cm_optimal)
    print(f"  TN={cm_optimal[0,0]}, FP={cm_optimal[0,1]}")
    print(f"  FN={cm_optimal[1,0]}, TP={cm_optimal[1,1]}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_optimal, digits=4))

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_prob)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")

    # ========== Visualization ==========
    if args.plot:
        print("\n[Eval] Generating plots...")
        plot_roc_curve(y_true, y_prob, output_dir / "roc_curve.png")
        plot_precision_recall_curve(y_true, y_prob, output_dir / "pr_curve.png")

    # ========== Error analysis ==========
    if args.error_analysis:
        print("\n[Eval] Analyzing errors...")
        analyze_errors(df, y_true, y_pred_optimal, output_dir / "error_cases.csv")

    # ========== Save predictions ==========
    if args.save_predictions:
        df["predicted_prob"] = y_prob
        df["predicted_label"] = y_pred_optimal
        df.to_csv(output_dir / "predictions.csv", index=False)
        print(f"[Eval] Predictions saved to: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM Judge Classifier (Optimized)")
    parser.add_argument("--data", type=str, default="data/test.csv", help="Test data path")
    parser.add_argument("--model", type=str, default="models/judge_clf.joblib", help="Model path")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--plot", action="store_true", help="Generate ROC and PR curves")
    parser.add_argument("--error-analysis", action="store_true", help="Analyze error cases")
    parser.add_argument("--save-predictions", action="store_true", help="Save predictions to CSV")

    args = parser.parse_args()
    main(args)