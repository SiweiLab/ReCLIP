#!/usr/bin/env python3
"""Evaluate AlphaMissense on PTM folds and save per-sample figure CSV."""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

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


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize AlphaMissense predictions on PTM ten-fold test rows."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "ten_folds"),
        help="Input file/dir/glob for PTM rows. Default: data/ptm/ten_folds",
    )
    parser.add_argument(
        "--pdb-dir",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "alpha_missense_data"),
        help="Primary AlphaMissense PDB directory.",
    )
    parser.add_argument(
        "--fallback-pdb-dir",
        default=os.path.join(PROJECT_ROOT, "data", "four_classes_mutation", "alpha_missense_data"),
        help="Fallback PDB directory if file is missing in --pdb-dir.",
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.join(
            PROJECT_ROOT,
            "scripts",
            "ptm",
            "figure_plot",
            "AlphaMissense_baseline_ptm_for_figure.csv",
        ),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--metric-txt",
        default=os.path.join(
            PROJECT_ROOT,
            "Results",
            "result_AlphaMissense_baseline_on_ptm_with_predictions.txt",
        ),
        help="Output metrics txt path.",
    )
    parser.add_argument("--threshold", type=float, default=0.56, help="AlphaMissense score threshold.")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow scoring even if Seq-derived Before_AA mismatches PDB residue.",
    )
    return parser.parse_args()


def _collect_input_files(path_or_glob: str) -> List[str]:
    if os.path.isdir(path_or_glob):
        # Prefer PTM fold test files when present.
        test_files = sorted(glob.glob(os.path.join(path_or_glob, "fold_*_test.csv")))
        if test_files:
            return test_files
        files = sorted(glob.glob(os.path.join(path_or_glob, "*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(path_or_glob, "*.tsv")))
        if not files:
            raise FileNotFoundError(f"No CSV/TSV files found in directory: {path_or_glob}")
        return files

    hits = sorted(glob.glob(path_or_glob))
    if hits:
        return hits
    if os.path.exists(path_or_glob):
        return [path_or_glob]
    raise FileNotFoundError(f"Input path not found: {path_or_glob}")


def _read_one_table(path: str) -> pd.DataFrame:
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _derive_before_after(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    seq_col = "Seq" if "Seq" in df.columns else "Target_Seq"
    mut_col = "Mutated_Seq (unless WT)" if "Mutated_Seq (unless WT)" in df.columns else "Mutated_Seq"
    seqs = df[seq_col].fillna("").astype(str).tolist()
    mut_seqs = df[mut_col].fillna("").astype(str).tolist()
    pos_list = pd.to_numeric(df["Position"], errors="coerce").fillna(-1).astype(int).tolist()

    before: List[str] = []
    after: List[str] = []
    for s, ms, pos in zip(seqs, mut_seqs, pos_list):
        idx = pos - 1
        b = s[idx] if 0 <= idx < len(s) else ""
        a = ms[idx] if 0 <= idx < len(ms) else b
        before.append(b)
        after.append(a)
    return before, after


def _normalize_ptm_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Target_UPID" not in d.columns and "Uniprot" in d.columns:
        d["Target_UPID"] = d["Uniprot"]
    if "Interactor_UPID" not in d.columns and "Int_uniprot" in d.columns:
        d["Interactor_UPID"] = d["Int_uniprot"]

    required = ["Target_UPID", "Interactor_UPID", "Position", "Y2H_score"]
    for col in required:
        if col not in d.columns:
            raise ValueError(f"Missing required column '{col}' in PTM input.")

    before, after = _derive_before_after(d)
    d["Before_AA"] = before
    d["After_AA"] = after
    d["Y2H_score"] = pd.to_numeric(d["Y2H_score"], errors="coerce").fillna(0).astype(int)
    d["Y2H_score_two_classes"] = (d["Y2H_score"] != 0).astype(int)

    if "residue_id" not in d.columns:
        d["residue_id"] = d["Target_UPID"].astype(str) + "-" + d["Position"].astype(str)
    if "ppi_id" not in d.columns:
        d["ppi_id"] = ""
    return d


def parse_pdb_ca_map(pdb_path: str) -> Dict[int, Tuple[str, float]]:
    """Return residue position -> (one-letter AA, B-factor) using CA atoms."""
    chain_maps: Dict[str, Dict[int, Tuple[str, float]]] = defaultdict(dict)
    chain_counts: Counter = Counter()

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            resname = line[17:20].strip()
            chain = line[21].strip() or "?"
            resseq = line[22:26].strip()
            if not resseq:
                continue
            try:
                pos = int(resseq)
                bfactor = float(line[60:66])
            except ValueError:
                continue
            aa1 = AA3_TO_1.get(resname, "X")
            if pos not in chain_maps[chain]:
                chain_maps[chain][pos] = (aa1, bfactor)
                chain_counts[chain] += 1

    if not chain_maps:
        return {}
    if "A" in chain_maps:
        return chain_maps["A"]
    best_chain = max(chain_counts.items(), key=lambda x: x[1])[0]
    return chain_maps[best_chain]


def load_pdb_map(upid: str, pdb_dirs: Sequence[str]) -> Optional[Dict[int, Tuple[str, float]]]:
    for d in pdb_dirs:
        if not d:
            continue
        pdb_path = os.path.join(d, f"AF-{upid}-F1-AM_v4.pdb")
        if os.path.exists(pdb_path):
            return parse_pdb_ca_map(pdb_path)
    return None


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
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


def _write_metric_txt(path: str, metrics: Dict[str, float], stats: Counter) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== AlphaMissense on PTM (with per-sample outputs) ===\n")
        for k in (
            "total_rows",
            "scored_rows",
            "missing_pdb",
            "missing_position",
            "aa_mismatch",
            "bad_position",
            "final_rows",
        ):
            f.write(f"{k}: {stats.get(k, 0)}\n")
        f.write("\n")
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
        for k in metric_order:
            v = metrics.get(k, float("nan"))
            f.write(f"{k}: {v:.6f}\n")


def main() -> int:
    args = parse_args()
    input_files = _collect_input_files(args.input)
    print(f"Input files: {len(input_files)}")
    for p in input_files:
        print(f"  - {p}")

    all_df = pd.concat([_normalize_ptm_df(_read_one_table(p)) for p in input_files], ignore_index=True)
    stats: Counter = Counter()
    stats["total_rows"] = int(len(all_df))

    pdb_dirs: List[str] = [args.pdb_dir]
    if args.fallback_pdb_dir and args.fallback_pdb_dir != args.pdb_dir:
        pdb_dirs.append(args.fallback_pdb_dir)
    pdb_cache: Dict[str, Optional[Dict[int, Tuple[str, float]]]] = {}

    scores: List[float] = []
    preds: List[int] = []
    for row in all_df.itertuples(index=False):
        upid = str(getattr(row, "Target_UPID", "")).strip()
        before = str(getattr(row, "Before_AA", "")).strip()
        pos = getattr(row, "Position", None)

        score = 0.0
        pred = 0
        try:
            pos_int = int(pos)
        except Exception:
            stats["bad_position"] += 1
            scores.append(score)
            preds.append(pred)
            continue

        if upid not in pdb_cache:
            pdb_cache[upid] = load_pdb_map(upid, pdb_dirs)
        pdb_map = pdb_cache[upid]
        if not pdb_map:
            stats["missing_pdb"] += 1
        elif pos_int not in pdb_map:
            stats["missing_position"] += 1
        else:
            aa1, bfactor = pdb_map[pos_int]
            if before and aa1 != before and not args.allow_mismatch:
                stats["aa_mismatch"] += 1
            else:
                score = float(bfactor)
                pred = 1 if score > args.threshold else 0
                stats["scored_rows"] += 1

        scores.append(score)
        preds.append(pred)

    out_df = all_df.copy()
    out_df["score_alphamissense"] = np.asarray(scores, dtype=float)
    out_df["AlphaMissense_baseline_Y_score_1"] = out_df["score_alphamissense"].clip(lower=0.0, upper=1.0)
    out_df["AlphaMissense_baseline_Y_score_0"] = 1.0 - out_df["AlphaMissense_baseline_Y_score_1"]
    out_df["AlphaMissense_baseline_Y_predict"] = np.asarray(preds, dtype=int)

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
        "score_alphamissense",
        "AlphaMissense_baseline_Y_score_0",
        "AlphaMissense_baseline_Y_score_1",
        "AlphaMissense_baseline_Y_predict",
    ]
    keep_cols = [c for c in keep_cols if c in out_df.columns]
    out_df = out_df[keep_cols].copy()

    id_cols = [
        c
        for c in keep_cols
        if c
        not in {
            "score_alphamissense",
            "AlphaMissense_baseline_Y_score_0",
            "AlphaMissense_baseline_Y_score_1",
            "AlphaMissense_baseline_Y_predict",
        }
    ]
    out_df = (
        out_df.groupby(id_cols, as_index=False)[
            ["score_alphamissense", "AlphaMissense_baseline_Y_score_0", "AlphaMissense_baseline_Y_score_1"]
        ]
        .mean()
        .reset_index(drop=True)
    )
    out_df["AlphaMissense_baseline_Y_predict"] = (
        out_df["score_alphamissense"] > args.threshold
    ).astype(int)

    sort_cols = [c for c in ["Target_UPID", "Position", "Interactor_UPID"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    y_true = out_df["Y2H_score_two_classes"].astype(int).values
    y_pred = out_df["AlphaMissense_baseline_Y_predict"].astype(int).values
    y_score = out_df["score_alphamissense"].astype(float).values
    metrics = _compute_metrics(y_true, y_pred, y_score)
    stats["final_rows"] = int(len(out_df))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    _write_metric_txt(args.metric_txt, metrics, stats)

    print(f"Saved CSV: {args.out_csv}")
    print(f"Saved metrics: {args.metric_txt}")
    print("Stats:", ", ".join(f"{k}={v}" for k, v in stats.items()))
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if np.isfinite(v) else f"{k}: NaN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
