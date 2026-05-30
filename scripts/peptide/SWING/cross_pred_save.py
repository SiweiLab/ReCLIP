#!/usr/bin/env python3
"""
SWING save wrapper for peptide task.

Runs the original SWING `cross_pred.py` pipeline (kept in the same folder) to
preserve baseline behavior, then merges per-fold prediction files into a
single figure-ready CSV and writes summary metrics.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_FIGURE_DIR = os.path.join(PROJECT_ROOT, "scripts", "peptide", "figure_plot")
ORIGINAL_SWING_SCRIPT = os.path.join(SCRIPT_DIR, "cross_pred.py")


def _dataset_tag(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SWING peptide wrapper with CSV export")
    parser.add_argument("--data_set", required=True, help='Dataset with "Set" column labeling Train and Test sets.')
    parser.add_argument("--output", default=None, help="SWING output tag prefix.")
    parser.add_argument("--cross_pred_set", default=None, help="Label for logs.")
    parser.add_argument("--loops", default=10, type=int, help="Number of stratified folds on Test set.")
    parser.add_argument(
        "--preset",
        default="auto",
        choices=["auto", "classI", "classII", "mixed", "none"],
        help="Hyperparameter preset passed through to cross_pred.py. auto resolves from dataset name.",
    )

    parser.add_argument("--metric", default="polarity", choices=["polarity", "hydrophobicity"])
    parser.add_argument("--k", default=None, type=int)
    parser.add_argument("--padding_score", default=9, type=int)

    parser.add_argument("--w", default=None, type=int)
    parser.add_argument("--dm", default=None, type=int)
    parser.add_argument("--dim", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--min_count", default=None, type=int)
    parser.add_argument("--alpha", default=None, type=float)

    parser.add_argument("--classifier", default="XGBoost", choices=["XGBoost", "LR"])
    parser.add_argument("--n_estimators", default=None, type=int)
    parser.add_argument("--max_depth", default=None, type=int)
    parser.add_argument("--learning_rate", default=None, type=float)
    parser.add_argument(
        "--xgb_device",
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="XGBoost device mode passed through to cross_pred.py",
    )
    parser.add_argument(
        "--xgb_n_jobs",
        default=8,
        type=int,
        help="XGBoost n_jobs passed through to cross_pred.py",
    )
    parser.add_argument("--max_iter", default=10000, type=int)
    parser.add_argument("--l1_ratio", default=0.5, type=float)

    parser.add_argument(
        "--out_csv",
        default=None,
        help="Prediction CSV path. Default: scripts/peptide/figure_plot/SWING_<dataset>_for_figure.csv",
    )
    parser.add_argument(
        "--metric_txt",
        default=None,
        help="Metric TXT path. Default: scripts/peptide/figure_plot/SWING_<dataset>_metrics.txt",
    )
    return parser.parse_args()


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(metrics.roc_auc_score(y_true, y_prob))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = float(metrics.accuracy_score(y_true, y_pred))
    precision = float(metrics.precision_score(y_true, y_pred, zero_division=0))
    recall = float(metrics.recall_score(y_true, y_pred, zero_division=0))
    f1 = float(metrics.f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(metrics.balanced_accuracy_score(y_true, y_pred))
    roc = _safe_roc_auc(y_true, y_prob)
    ap = float(metrics.average_precision_score(y_true, y_prob))
    youden_j = float("nan")
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        youden_j = float(np.max(tpr - fpr))
    t_stat = float("nan")
    p_val = float("nan")
    try:
        t_stat, p_val = stats.ttest_rel(y_true, y_pred)
        t_stat = float(t_stat)
        p_val = float(p_val)
    except Exception:
        pass
    return {
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Balanced_Accuracy": bal_acc,
        "ROC_AUC": roc,
        "PR_AUC(AP)": ap,
        "Youdens_J": youden_j,
        "t_test_t": t_stat,
        "p_value": p_val,
    }


def _save_metrics(metric_txt: str, metric_map: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(metric_txt), exist_ok=True)
    ordered_keys = [
        "ACC",
        "Precision",
        "Recall",
        "F1",
        "Balanced_Accuracy",
        "ROC_AUC",
        "PR_AUC(AP)",
        "Youdens_J",
        "t_test_t",
        "p_value",
    ]
    with open(metric_txt, "w", encoding="utf-8") as f:
        for key in ordered_keys:
            f.write(f"{key}: {metric_map.get(key, float('nan')):.6f}\n")


def _call_original(args: argparse.Namespace, run_tag: str) -> None:
    cmd = [
        sys.executable,
        ORIGINAL_SWING_SCRIPT,
        "--data_set",
        args.data_set,
        "--output",
        run_tag,
        "--cross_pred_set",
        (args.cross_pred_set or _dataset_tag(args.data_set)),
        "--loops",
        str(args.loops),
        "--preset",
        args.preset,
        "--metric",
        args.metric,
        "--padding_score",
        str(args.padding_score),
        "--classifier",
        args.classifier,
        "--xgb_device",
        args.xgb_device,
        "--xgb_n_jobs",
        str(args.xgb_n_jobs),
        "--max_iter",
        str(args.max_iter),
        "--l1_ratio",
        str(args.l1_ratio),
    ]
    optional_args = [
        ("--k", args.k),
        ("--w", args.w),
        ("--dm", args.dm),
        ("--dim", args.dim),
        ("--epochs", args.epochs),
        ("--min_count", args.min_count),
        ("--alpha", args.alpha),
        ("--n_estimators", args.n_estimators),
        ("--max_depth", args.max_depth),
        ("--learning_rate", args.learning_rate),
    ]
    for flag, value in optional_args:
        if value is not None:
            cmd.extend([flag, str(value)])
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def main() -> None:
    args = parse_args()
    if not os.path.exists(ORIGINAL_SWING_SCRIPT):
        raise FileNotFoundError(f"Missing original SWING script: {ORIGINAL_SWING_SCRIPT}")

    args.data_set = os.path.abspath(args.data_set)
    tag = args.output or _dataset_tag(args.data_set)
    out_csv = args.out_csv or os.path.join(DEFAULT_FIGURE_DIR, f"SWING_{tag}_for_figure.csv")
    metric_txt = args.metric_txt or os.path.join(DEFAULT_FIGURE_DIR, f"SWING_{tag}_metrics.txt")
    os.makedirs(DEFAULT_FIGURE_DIR, exist_ok=True)

    # Unique tag to avoid collisions with concurrent or previous runs.
    run_tag = f"{tag}_save_{int(time.time())}"
    print(f"[SWING save] data_set={args.data_set}")
    print(f"[SWING save] run_tag={run_tag}")
    _call_original(args, run_tag)

    pred_glob = os.path.join(SCRIPT_DIR, "output", "cross_pred", "dataframes", f"predictions_{run_tag}_*.csv")
    fold_re = re.compile(rf"predictions_{re.escape(run_tag)}_(\d+)\.csv$")
    pred_files = sorted(
        glob.glob(pred_glob),
        key=lambda p: int(fold_re.search(os.path.basename(p)).group(1)),
    )
    if not pred_files:
        raise RuntimeError(f"No fold prediction files found: {pred_glob}")

    frames = []
    for fp in pred_files:
        fold_df = pd.read_csv(fp)
        fold_df["__source_file"] = os.path.basename(fp)
        frames.append(fold_df)
    merged = pd.concat(frames, ignore_index=True)

    # Sanity check: merged fold rows should match the original test-set size.
    raw_df = pd.read_csv(args.data_set, usecols=["Set"])
    set_norm = raw_df["Set"].astype(str).str.strip().str.lower()
    expected_test_rows = int((set_norm == "test").sum())
    if len(merged) != expected_test_rows:
        raise RuntimeError(
            f"Merged prediction row count mismatch: merged={len(merged)} vs expected_test={expected_test_rows}. "
            "Check cross_pred.py split index order."
        )

    required_pred_cols = ["Epitope", "MHC", "True Y", "Predicted Y", "Predicted Probabilities Y"]
    missing = [c for c in required_pred_cols if c not in merged.columns]
    if missing:
        raise KeyError(f"Merged prediction missing columns: {missing}")

    merged = merged[required_pred_cols]
    merged["True Y"] = pd.to_numeric(merged["True Y"], errors="coerce")
    merged["Predicted Y"] = pd.to_numeric(merged["Predicted Y"], errors="coerce")
    merged["Predicted Probabilities Y"] = pd.to_numeric(merged["Predicted Probabilities Y"], errors="coerce")
    merged = merged.dropna(subset=["True Y", "Predicted Y", "Predicted Probabilities Y"]).copy()
    merged["True Y"] = merged["True Y"].astype(int)
    merged["Predicted Y"] = merged["Predicted Y"].astype(int)

    merged = merged.sort_values(by="Predicted Probabilities Y", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged.to_csv(out_csv, index=False)

    y_true = merged["True Y"].to_numpy()
    y_pred = merged["Predicted Y"].to_numpy()
    y_prob = merged["Predicted Probabilities Y"].to_numpy()
    metric_map = _compute_metrics(y_true, y_pred, y_prob)
    _save_metrics(metric_txt, metric_map)

    print(f"Prediction CSV saved: {out_csv}")
    print(f"Metrics TXT saved: {metric_txt}")
    for k, v in metric_map.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
