#!/usr/bin/env python3
"""Evaluate PrimateAI on PTM and save per-sample figure CSV + metrics txt."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PrimateAI on PTM and export figure-compatible CSV."
    )
    parser.add_argument(
        "--pai-csv",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "PrimateAI", "PrimateAI-3D_hg38_filtered.csv"),
        help="Filtered PrimateAI CSV path.",
    )
    parser.add_argument(
        "--mutation-csv",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "PrimateAI", "Mutation_ptm_with_refseq.csv"),
        help="PTM mutation CSV with refseq column.",
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.join(
            PROJECT_ROOT,
            "scripts",
            "ptm",
            "figure_plot",
            "PrimateAI_baseline_ptm_for_figure.csv",
        ),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--metric-txt",
        default=os.path.join(
            PROJECT_ROOT,
            "Results",
            "result_PrimateAI_baseline_on_ptm_with_predictions.txt",
        ),
        help="Output metrics txt path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fallback threshold for score_PAI3D when prediction label is missing.",
    )
    parser.add_argument(
        "--pai-chunksize",
        type=int,
        default=1_000_000,
        help="Chunk size when scanning large PrimateAI CSV.",
    )
    return parser.parse_args()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if len(y_true) == 0:
        return {
            k: float("nan")
            for k in [
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
        }

    out["ACC"] = float(accuracy_score(y_true, y_pred))
    out["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["Balanced_Accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    try:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["ROC_AUC"] = float("nan")
    try:
        out["PR_AUC(AP)"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        out["PR_AUC(AP)"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    out["Youdens_J"] = float(sens + spec - 1) if not np.isnan(sens) and not np.isnan(spec) else float("nan")

    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]
    if len(pos_scores) > 1 and len(neg_scores) > 1:
        t_stat, p_val = ttest_ind(pos_scores, neg_scores, equal_var=False)
        out["t_test_t"] = float(t_stat)
        out["p_value"] = float(p_val)
    else:
        out["t_test_t"] = float("nan")
        out["p_value"] = float("nan")
    return out


def _metric_lines(prefix: str, metrics: Dict[str, float]) -> list[str]:
    keys = [
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
    out = [f"{prefix}:"]
    for k in keys:
        v = metrics.get(k, float("nan"))
        out.append(f"  {k}: {v:.6f}" if np.isfinite(v) else f"  {k}: NaN")
    return out


def _prepare_mutation_df(mut: pd.DataFrame) -> Tuple[pd.DataFrame, set, set, set, set]:
    required_mut = [
        "Target_UPID",
        "Position",
        "Before_AA",
        "After_AA",
        "Y2H_score",
        "Interactor_UPID",
        "residue_id",
        "ppi_id",
        "refseq",
    ]
    for c in required_mut:
        if c not in mut.columns:
            raise ValueError(f"Mutation CSV missing required column: {c}")

    m = mut.copy()
    m["Position"] = pd.to_numeric(m["Position"], errors="coerce")
    m["Y2H_score"] = pd.to_numeric(m["Y2H_score"], errors="coerce").fillna(0).astype(int)
    if "Y2H_score_two_classes" not in m.columns:
        m["Y2H_score_two_classes"] = (m["Y2H_score"] != 0).astype(int)
    else:
        m["Y2H_score_two_classes"] = (
            pd.to_numeric(m["Y2H_score_two_classes"], errors="coerce").fillna(0).astype(int)
        )
    m["Before_AA"] = m["Before_AA"].fillna("").astype(str).str.upper().str.strip()
    m["After_AA"] = m["After_AA"].fillna("").astype(str).str.upper().str.strip()
    m["refseq"] = m["refseq"].fillna("").astype(str).str.strip()
    m["refseq_trim"] = m["refseq"].str.split(".").str[0]

    needed_full = set(
        zip(
            m["refseq"].astype(str),
            m["Position"].fillna(-1).astype(int),
            m["Before_AA"].astype(str),
            m["After_AA"].astype(str),
        )
    )
    needed_trim = set(
        zip(
            m["refseq_trim"].astype(str),
            m["Position"].fillna(-1).astype(int),
            m["Before_AA"].astype(str),
            m["After_AA"].astype(str),
        )
    )
    needed_refseq_full = set(m["refseq"].astype(str).tolist())
    needed_refseq_trim = set(m["refseq_trim"].astype(str).tolist())

    needed_full.discard(("", -1, "", ""))
    needed_trim.discard(("", -1, "", ""))
    needed_refseq_full.discard("")
    needed_refseq_trim.discard("")
    return m, needed_full, needed_trim, needed_refseq_full, needed_refseq_trim


def _build_primate_maps_from_file(
    pai_csv: str,
    threshold: float,
    chunksize: int,
    needed_full: set,
    needed_trim: set,
    needed_refseq_full: set,
    needed_refseq_trim: set,
) -> Tuple[Dict[tuple, tuple], Dict[tuple, tuple], Dict[tuple, tuple], Dict[tuple, tuple]]:
    """
    Stream PrimateAI CSV and keep only records needed by PTM mutations.

    Returns:
      full_exact_map: (refseq_full, pos, ref, alt) -> (score, pred)
      trim_exact_map: (refseq_trim, pos, ref, alt) -> (score, pred)
      full_ref_map:   (refseq_full, pos, ref)      -> (score, pred)
      trim_ref_map:   (refseq_trim, pos, ref)      -> (score, pred)
    """
    header = pd.read_csv(pai_csv, nrows=0)
    required = ["refseq", "change_position_1based", "ref_aa", "alt_aa", "score_PAI3D"]
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise ValueError(f"Missing required columns in PrimateAI CSV: {missing}")

    usecols = [c for c in ["refseq", "change_position_1based", "ref_aa", "alt_aa", "score_PAI3D", "prediction", "Y2H_score"] if c in header.columns]

    full_exact_map: Dict[tuple, tuple] = {}
    trim_exact_map: Dict[tuple, tuple] = {}
    full_ref_map: Dict[tuple, tuple] = {}
    trim_ref_map: Dict[tuple, tuple] = {}

    needed_ref_full = {(k[0], k[1], k[2]) for k in needed_full}
    needed_ref_trim = {(k[0], k[1], k[2]) for k in needed_trim}
    scanned_rows = 0
    prefiltered_rows = 0
    candidate_rows_exact = 0
    candidate_rows_ref = 0

    reader = pd.read_csv(
        pai_csv,
        usecols=usecols,
        chunksize=max(1, int(chunksize)),
        low_memory=False,
    )
    for chunk_idx, chunk in enumerate(reader):
        scanned_rows += len(chunk)

        chunk["refseq"] = chunk["refseq"].fillna("").astype(str).str.strip()
        chunk["refseq_trim"] = chunk["refseq"].str.split(".").str[0]
        chunk["change_position_1based"] = pd.to_numeric(chunk["change_position_1based"], errors="coerce")
        chunk["ref_aa"] = chunk["ref_aa"].fillna("").astype(str).str.upper().str.strip()
        chunk["alt_aa"] = chunk["alt_aa"].fillna("").astype(str).str.upper().str.strip()
        chunk["score_PAI3D"] = pd.to_numeric(chunk["score_PAI3D"], errors="coerce")

        if "Y2H_score" in chunk.columns:
            chunk["Y2H_pred"] = pd.to_numeric(chunk["Y2H_score"], errors="coerce")
        elif "prediction" in chunk.columns:
            chunk["Y2H_pred"] = chunk["prediction"].map({"pathogenic": 1, "benign": 0})
        else:
            chunk["Y2H_pred"] = np.nan

        chunk = chunk.dropna(subset=["change_position_1based", "score_PAI3D"]).copy()
        if chunk.empty:
            continue
        chunk["change_position_1based"] = chunk["change_position_1based"].astype(int)
        chunk["Y2H_pred"] = chunk["Y2H_pred"].where(
            ~chunk["Y2H_pred"].isna(), (chunk["score_PAI3D"] >= threshold).astype(int)
        )
        chunk["Y2H_pred"] = chunk["Y2H_pred"].astype(int)

        mask_refseq = chunk["refseq"].isin(needed_refseq_full) | chunk["refseq_trim"].isin(needed_refseq_trim)
        chunk = chunk[mask_refseq]
        prefiltered_rows += len(chunk)
        if chunk.empty:
            if (chunk_idx + 1) % 10 == 0:
                print(
                    f"Scanned chunks={chunk_idx + 1}, rows={scanned_rows}, "
                    f"prefiltered={prefiltered_rows}, "
                    f"candidates_exact={candidate_rows_exact}, candidates_ref={candidate_rows_ref}"
                )
            continue

        for row in chunk.itertuples(index=False):
            pos = int(row.change_position_1based)
            ref = str(row.ref_aa)
            alt = str(row.alt_aa)
            r_full = str(row.refseq)
            r_trim = str(row.refseq_trim)
            score = float(row.score_PAI3D)
            pred = int(row.Y2H_pred)
            rec = (score, pred)

            k_full = (r_full, pos, ref, alt)
            if k_full in needed_full:
                candidate_rows_exact += 1
                if k_full not in full_exact_map or score > full_exact_map[k_full][0]:
                    full_exact_map[k_full] = rec

            k_trim = (r_trim, pos, ref, alt)
            if k_trim in needed_trim:
                candidate_rows_exact += 1
                if k_trim not in trim_exact_map or score > trim_exact_map[k_trim][0]:
                    trim_exact_map[k_trim] = rec

            k_full_ref = (r_full, pos, ref)
            if k_full_ref in needed_ref_full:
                candidate_rows_ref += 1
                if k_full_ref not in full_ref_map or score > full_ref_map[k_full_ref][0]:
                    full_ref_map[k_full_ref] = rec

            k_trim_ref = (r_trim, pos, ref)
            if k_trim_ref in needed_ref_trim:
                candidate_rows_ref += 1
                if k_trim_ref not in trim_ref_map or score > trim_ref_map[k_trim_ref][0]:
                    trim_ref_map[k_trim_ref] = rec

        if (chunk_idx + 1) % 10 == 0:
            print(
                f"Scanned chunks={chunk_idx + 1}, rows={scanned_rows}, "
                f"prefiltered={prefiltered_rows}, "
                f"candidates_exact={candidate_rows_exact}, candidates_ref={candidate_rows_ref}"
            )

    print(
        f"Finished scanning PrimateAI: scanned_rows={scanned_rows}, "
        f"prefiltered_rows={prefiltered_rows}, "
        f"candidates_exact={candidate_rows_exact}, candidates_ref={candidate_rows_ref}"
    )
    return full_exact_map, trim_exact_map, full_ref_map, trim_ref_map


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.pai_csv):
        raise FileNotFoundError(
            f"PrimateAI filtered CSV not found: {args.pai_csv}\n"
            "Run filter_primateai.py first (and provide the original PrimateAI-3D input file if needed)."
        )
    if not os.path.exists(args.mutation_csv):
        raise FileNotFoundError(
            f"PTM mutation+refseq CSV not found: {args.mutation_csv}\n"
            "Run add_refseq_column.py first."
        )

    mut = pd.read_csv(args.mutation_csv, low_memory=False)
    print(f"Loaded PTM mutation rows: {len(mut)}")
    m, needed_full, needed_trim, needed_refseq_full, needed_refseq_trim = _prepare_mutation_df(mut)
    print(
        f"Need PrimateAI keys: full={len(needed_full)}, trim={len(needed_trim)}, "
        f"refseq_full={len(needed_refseq_full)}, refseq_trim={len(needed_refseq_trim)}"
    )

    full_exact_map, trim_exact_map, full_ref_map, trim_ref_map = _build_primate_maps_from_file(
        pai_csv=args.pai_csv,
        threshold=args.threshold,
        chunksize=args.pai_chunksize,
        needed_full=needed_full,
        needed_trim=needed_trim,
        needed_refseq_full=needed_refseq_full,
        needed_refseq_trim=needed_refseq_trim,
    )
    print(
        "PrimateAI maps: "
        f"full_exact={len(full_exact_map)}, trim_exact={len(trim_exact_map)}, "
        f"full_ref={len(full_ref_map)}, trim_ref={len(trim_ref_map)}"
    )

    score_list = []
    pred_list = []
    match_type = []
    matched = 0
    for row in m.itertuples(index=False):
        score = 0.0
        pred = 0
        mtype = "none"

        if not np.isnan(row.Position):
            pos = int(row.Position)
            key_full = (str(row.refseq), pos, str(row.Before_AA), str(row.After_AA))
            key_trim = (str(row.refseq_trim), pos, str(row.Before_AA), str(row.After_AA))
            key_full_ref = (str(row.refseq), pos, str(row.Before_AA))
            key_trim_ref = (str(row.refseq_trim), pos, str(row.Before_AA))
            if key_full in full_exact_map:
                score, pred = full_exact_map[key_full]
                mtype = "full_exact"
            elif key_trim in trim_exact_map:
                score, pred = trim_exact_map[key_trim]
                mtype = "trim_exact"
            elif key_full_ref in full_ref_map:
                score, pred = full_ref_map[key_full_ref]
                mtype = "full_ref"
            elif key_trim_ref in trim_ref_map:
                score, pred = trim_ref_map[key_trim_ref]
                mtype = "trim_ref"

        if mtype != "none":
            matched += 1
        score_list.append(float(score))
        pred_list.append(int(pred))
        match_type.append(mtype)

    out_df = m.copy()
    out_df["score_PAI3D"] = np.asarray(score_list, dtype=float)
    out_df["PrimateAI_match_type"] = match_type
    out_df["PrimateAI_baseline_Y_score_1"] = out_df["score_PAI3D"].clip(lower=0.0, upper=1.0)
    out_df["PrimateAI_baseline_Y_score_0"] = 1.0 - out_df["PrimateAI_baseline_Y_score_1"]
    out_df["PrimateAI_baseline_Y_predict"] = np.asarray(pred_list, dtype=int)

    keep_cols = [
        "Target_UPID",
        "Position",
        "Before_AA",
        "After_AA",
        "Y2H_score",
        "Y2H_score_two_classes",
        "Interactor_UPID",
        "residue_id",
        "ppi_id",
        "refseq",
        "score_PAI3D",
        "PrimateAI_baseline_Y_score_0",
        "PrimateAI_baseline_Y_score_1",
        "PrimateAI_baseline_Y_predict",
    ]
    keep_cols = [c for c in keep_cols if c in out_df.columns]
    out_df = out_df[keep_cols].copy()

    id_cols = [
        c
        for c in keep_cols
        if c
        not in {
            "score_PAI3D",
            "PrimateAI_baseline_Y_score_0",
            "PrimateAI_baseline_Y_score_1",
            "PrimateAI_baseline_Y_predict",
        }
    ]
    out_df = (
        out_df.groupby(id_cols, as_index=False)[
            ["score_PAI3D", "PrimateAI_baseline_Y_score_0", "PrimateAI_baseline_Y_score_1"]
        ]
        .mean()
        .reset_index(drop=True)
    )
    out_df["PrimateAI_baseline_Y_predict"] = np.argmax(
        out_df[["PrimateAI_baseline_Y_score_0", "PrimateAI_baseline_Y_score_1"]].values, axis=1
    ).astype(int)

    sort_cols = [c for c in ["Target_UPID", "Position", "Interactor_UPID"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    y_true_all = out_df["Y2H_score_two_classes"].astype(int).values
    y_pred_all = out_df["PrimateAI_baseline_Y_predict"].astype(int).values
    y_score_all = out_df["score_PAI3D"].astype(float).values
    metrics_all = _compute_metrics(y_true_all, y_pred_all, y_score_all)

    matched_mask = out_df["score_PAI3D"] > 0
    y_true_matched = out_df.loc[matched_mask, "Y2H_score_two_classes"].astype(int).values
    y_pred_matched = out_df.loc[matched_mask, "PrimateAI_baseline_Y_predict"].astype(int).values
    y_score_matched = out_df.loc[matched_mask, "score_PAI3D"].astype(float).values
    metrics_matched = _compute_metrics(y_true_matched, y_pred_matched, y_score_matched)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    os.makedirs(os.path.dirname(args.metric_txt), exist_ok=True)
    with open(args.metric_txt, "w", encoding="utf-8") as f:
        f.write("=== PrimateAI on PTM (with per-sample outputs) ===\n")
        f.write(f"Total_mutations: {len(out_df)}\n")
        f.write(f"Matched_mutations(score>0): {int(matched_mask.sum())}\n")
        f.write(f"Unmatched_mutations(score=0): {int((~matched_mask).sum())}\n")
        f.write("\n")
        f.write("\n".join(_metric_lines("All_rows_metrics", metrics_all)))
        f.write("\n\n")
        f.write("\n".join(_metric_lines("Matched_rows_metrics", metrics_matched)))
        f.write("\n")

    print(f"Saved figure CSV: {args.out_csv}")
    print(f"Saved metrics: {args.metric_txt}")
    print(f"Total rows: {len(out_df)} | matched(score>0): {int(matched_mask.sum())}")
    for line in _metric_lines("All rows", metrics_all):
        print(line)
    for line in _metric_lines("Matched rows", metrics_matched):
        print(line)


if __name__ == "__main__":
    main()
