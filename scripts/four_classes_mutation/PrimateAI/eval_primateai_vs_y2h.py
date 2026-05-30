#!/usr/bin/env python3
"""
Evaluate PrimateAI-3D against mutation-level Y2H labels.

Files:
  PrimateAI-3D_hg38_filtered.csv
  Mutation_IMEx_IntAct_clean_with_refseq.csv
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from scipy.stats import ttest_ind
import numpy as np

PAI_PATH = "PrimateAI-3D_hg38_filtered.csv"
MUT_PATH = "Mutation_IMEx_IntAct_clean_with_refseq.csv"

def main():
    # 1) Read inputs.
    pai = pd.read_csv(PAI_PATH)
    mut = pd.read_csv(MUT_PATH)

    # 2) Rename score columns to avoid merge conflicts.
    pai = pai.rename(columns={
        "Y2H_score": "Y2H_pred"
    })
    mut = mut.rename(columns={
        "Y2H_score": "Y2H_true"
    })

    # 3) Keep required PrimateAI columns and deduplicate by mutation key.
    key_cols = ["refseq", "change_position_1based", "ref_aa", "alt_aa"]
    required_pai_cols = key_cols + [
        "score_PAI3D", "percentile_PAI3D", "prediction", "Y2H_pred"
    ]
    pai = pai[required_pai_cols].drop_duplicates(subset=key_cols)

    # 4) Validate required mutation-table columns.
    required_mut_cols = ["refseq", "Position", "Before_AA", "After_AA", "Y2H_true"]
    for c in required_mut_cols:
        if c not in mut.columns:
            raise ValueError(f"Column {c} missing in {MUT_PATH}")

    total_mut_rows = len(mut)

    # 5) Left-join with the mutation table as the anchor.
    merged = mut.merge(
        pai,
        left_on=["refseq", "Position", "Before_AA", "After_AA"],
        right_on=["refseq", "change_position_1based", "ref_aa", "alt_aa"],
        how="left",
        suffixes=("_mut", "_pai")
    )

    # 6) Mark mutations matched to PrimateAI.
    hit_mask = merged["score_PAI3D"].notna()
    hit_mut_rows = hit_mask.sum()
    miss_mut_rows = total_mut_rows - hit_mut_rows

    print(f"Original mutation rows         : {total_mut_rows}")
    print(f"Mutations matched to PrimateAI : {hit_mut_rows}")
    print(f"Mutations without PrimateAI    : {miss_mut_rows}")

    # 7) Compute metrics on mutations matched to PrimateAI.
    matched = merged[hit_mask].copy()

    # Map the original four-class Y2H labels to binary labels.
    # Original Y2H_true may be 0,1,2,3:
    #   0 -> negative class 0
    #   1,2,3 -> positive class 1
    matched["Y2H_true_raw"] = matched["Y2H_true"]
    matched["Y2H_true_bin"] = matched["Y2H_true_raw"].apply(
        lambda x: 0 if x == 0 else 1
    )

    # PrimateAI predictions are already 0/1; cast defensively.
    matched["Y2H_pred_bin"] = matched["Y2H_pred"].astype(int)

    y_true = matched["Y2H_true_bin"].astype(int).values
    y_pred = matched["Y2H_pred_bin"].astype(int).values
    y_score = matched["score_PAI3D"].astype(float).values

    # 8) Classification metrics on binarized labels.
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    youden_j = sens + spec - 1 if (not np.isnan(sens) and not np.isnan(spec)) else np.nan

    # 9) AUC / PR-AUC using score_PAI3D and binarized labels.
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = np.nan

    # 10) t-test on score_PAI3D grouped by binarized labels.
    pos_scores = matched.loc[matched["Y2H_true_bin"] == 1, "score_PAI3D"].astype(float)
    neg_scores = matched.loc[matched["Y2H_true_bin"] == 0, "score_PAI3D"].astype(float)

    if len(pos_scores) > 1 and len(neg_scores) > 1:
        t_stat, p_val = ttest_ind(pos_scores, neg_scores, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan

    # 11) Count matched binary predictions.
    same_y2h = (matched["Y2H_true_bin"] == matched["Y2H_pred_bin"]).sum()

    print("\n=== Metrics on matched mutations ===")
    print(f"Exact label matches (Y2H_true_bin == Y2H_pred_bin): {same_y2h}")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision         : {prec:.4f}")
    print(f"Recall            : {rec:.4f}")
    print(f"F1                : {f1:.4f}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"AUC               : {auc:.4f}" if not np.isnan(auc) else "AUC               : NaN")
    print(f"PR_AUC            : {pr_auc:.4f}" if not np.isnan(pr_auc) else "PR_AUC            : NaN")
    print(f"Youden's J        : {youden_j:.4f}" if not np.isnan(youden_j) else "Youden's J        : NaN")
    print(f"t_stat            : {t_stat:.4f}" if not np.isnan(t_stat) else "t_stat            : NaN")
    print(f"p-value           : {p_val:.4e}" if not np.isnan(p_val) else "p-value           : NaN")

    # 12) Save metrics.
    metrics = {
        "Total_mutations": total_mut_rows,
        "Matched_mutations": hit_mut_rows,
        "Unmatched_mutations": miss_mut_rows,
        "Exact_Y2H_matches": same_y2h,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Balanced_Accuracy": bal_acc,
        "AUC": auc,
        "PR_AUC": pr_auc,
        "Youden_J": youden_j,
        "t_stat": t_stat,
        "p_value": p_val,
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv("PrimateAI_vs_Y2H_metrics.csv", index=False)
    print("\nSaved metrics to PrimateAI_vs_Y2H_metrics.csv")

if __name__ == "__main__":
    main()
