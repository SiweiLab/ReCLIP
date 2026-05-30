#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import argparse
import os

# ===================== Configuration =====================

NUM_FOLDS = 10

# Mutation CSV path.
DATASET_PATH = "data/Mutation_perturbation_clean_with_bert.csv"

# Label column.
LABEL_COL = "Y2H_score"

# Output directory and file name.
RESULTS_DIR = "SWING_Baseline_Results"
RESULT_FILE = "result_SWING_baseline_BERT_on_SWING_mutation.txt"

# ===================== Metric computation =====================

def compute_ten_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
    """
    Compute ten evaluation metrics:
    ACC, Precision, Recall, F1, Balanced_Accuracy,
    ROC_AUC, PR_AUC(AP), Youdens_J, t_test_t, p_value
    """
    try:
        from scipy.stats import ttest_ind
        has_scipy = True
    except Exception:
        has_scipy = False

    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    bal = metrics.balanced_accuracy_score(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        roc = metrics.roc_auc_score(y_true, y_proba)
    else:
        roc = float("nan")

    try:
        ap = metrics.average_precision_score(y_true, y_proba)
    except Exception:
        ap = float("nan")

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    youden_j = tpr + tnr - 1

    if has_scipy:
        pos = y_proba[y_true == 1]
        neg = y_proba[y_true == 0]
        if len(pos) > 1 and len(neg) > 1:
            t_stat, p_val = ttest_ind(pos, neg, equal_var=False)
        else:
            t_stat, p_val = float("nan"), float("nan")
    else:
        t_stat, p_val = float("nan"), float("nan")

    return {
        "ACC": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Balanced_Accuracy": bal,
        "ROC_AUC": roc,
        "PR_AUC(AP)": ap,
        "Youdens_J": youden_j,
        "t_test_t": t_stat,
        "p_value": p_val,
    }


def aggregate_metrics(metric_dicts):
    """
    Aggregate per-fold metrics as mean and standard deviation.
    """
    metric_list = list(metric_dicts)
    if not metric_list:
        return {}
    agg = {}
    first = metric_list[0]
    for key in first.keys():
        vals = [
            float(md[key])
            for md in metric_list
            if isinstance(md.get(key), (int, float, np.floating))
        ]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        agg[f"{key}_mean"] = float(np.nanmean(arr))
        agg[f"{key}_std"] = float(np.nanstd(arr))
    return agg

# ===================== Command-line arguments =====================

parser = argparse.ArgumentParser()

parser.add_argument("--n_estimators", type=int, help="The number of trees for the XGBoost model")
parser.add_argument("--max_depth", type=int, help="The maximum depth of the trees")
parser.add_argument("--learning_rate", type=float, help="The learning rate")
parser.add_argument("--run_name", type=str, help="name of model training run")
parser.add_argument("--project_name", type=str, help="wandb project name (ignored here)")

# ===================== Main workflow =====================

def main():
    args = parser.parse_args()

    print("BERT baseline on Mutation_perturbation - Ten Fold Evaluation")
    print("=" * 80)

    # ---------- 1. Load data ----------
    df_raw = pd.read_csv(DATASET_PATH)
    if LABEL_COL not in df_raw.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in {DATASET_PATH}")

    # Keep rows with valid labels; feature columns are validated downstream.
    df = df_raw.dropna(subset=[LABEL_COL]).copy()
    print(f"Loading dataset: {DATASET_PATH}")
    print(f"  rows: {len(df_raw)} -> {len(df)} | dropped: {len(df_raw) - len(df)}")

    # ---------- 2. Build y ----------
    y = df[LABEL_COL].to_numpy().astype(np.float32)
    # Uncomment for datasets that encode binary labels as -1/1.
    # y = (y > 0).astype(int)

    # ---------- 3. Build X ----------
    drop_cols = [
        LABEL_COL,                 # Y2H_score
        "Target_UPID",
        "Mutation",
        "Before_AA",
        "Position",
        "After_AA",
        "Interactor_UPID",
        "Target_Seq",
        "Interactor_Seq",
        "Mutated_Seq (unless WT)",
        "Data",
        "Type",
        "Category",
    ]
    # Drop metadata columns that are present in the current table.
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Remove metadata columns before selecting numeric embeddings.
    df_features = df.drop(columns=drop_cols)

    # Keep only numeric embedding dimensions.
    feature_df = df_features.select_dtypes(include=[np.number])

    print("All columns:", df.columns.tolist())
    print("Feature columns:", feature_df.columns.tolist())
    print(f"Number of feature columns: {feature_df.shape[1]}")

    X = feature_df.to_numpy(dtype=np.float32)

    # ---------- 3.5 Sanity check ----------
    bad_mask = ~np.isfinite(X)
    print("Has NaN/Inf:", bool(bad_mask.any()))
    row_zero = np.all(np.isclose(X, 0.0), axis=1)
    print("Num all-zero rows:", int(row_zero.sum()))

    # ---------- 4. 10-fold Stratified CV + XGBoost ----------
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    fold_metrics = {}
    collected = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n=== Fold {fold_idx} / {NUM_FOLDS - 1} (XGBoost) ===")

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # XGBoost hyperparameters used by the baseline.
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        r = neg / max(1.0, pos)

        base_params = dict(
            n_estimators=3500,
            learning_rate=0.02,
            max_depth=4,
            min_child_weight=12,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=1.0,
            reg_lambda=10.0,
            reg_alpha=0.1,
            max_delta_step=1,
            scale_pos_weight=float(1.3 * r),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            device="cuda",
        )
        params = dict(base_params, tree_method="hist", device="cuda")

        clf = XGBClassifier(**params)

        print(type(X_tr))
        print(type(y_tr))

        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(np.int32)

        metrics_fold = compute_ten_metrics(y_te, pred, proba)
        fold_metrics[fold_idx] = metrics_fold
        collected.append(metrics_fold)

        for k, v in metrics_fold.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

    # ---------- 5. Aggregate mean/std ----------
    summary = aggregate_metrics(collected)
    print("\n=== Aggregated (mean/std) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    # ---------- 6. Write result file ----------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metric_order = [
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

    result_path = os.path.join(RESULTS_DIR, RESULT_FILE)
    with open(result_path, "w", encoding="utf-8") as f:
        for fold_idx in range(NUM_FOLDS):
            f.write(f"=== Fold {fold_idx} ===\n")
            metrics_fold = fold_metrics[fold_idx]
            for key in metric_order:
                val = metrics_fold.get(key, float("nan"))
                if isinstance(val, (int, float, np.floating)):
                    f.write(f"{key}: {val:.6f}\n")
                else:
                    f.write(f"{key}: {val}\n")
            f.write("\n")

        f.write("=== Aggregated ===\n")
        for key in metric_order:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in summary:
                f.write(f"{mean_key}: {summary[mean_key]:.6f}\n")
            if std_key in summary:
                f.write(f"{std_key}: {summary[std_key]:.6f}\n")

    print(f"\nResults written to: {result_path}")
    print("Ten-fold evaluation complete.")


if __name__ == "__main__":
    main()
