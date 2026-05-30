#!/usr/bin/env python3
"""
Run SWING ten-fold evaluation on PTM and save per-sample outputs for plotting.

This script keeps the SWING-style XGBoost evaluation and additionally exports:
1) Per-sample OOF predictions CSV for figure plotting.
2) Fold metrics text file.
3) Merged table update into scripts/ptm/figure_plot/PTM_for_figure.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

FOLD_COUNT = 10
MUT_COL_RAW = "Mutated_Seq (unless WT)"
MUT_COL_ALT = "Mutated_Seq_unless_WT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PTM SWING baseline and save per-sample outputs."
    )
    parser.add_argument(
        "--feature-source",
        choices=["pkl", "esm"],
        default="pkl",
        help="Feature source for SWING: pkl vectors (recommended) or ESM embeddings.",
    )
    parser.add_argument(
        "--fold-dir",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "ten_folds"),
        help="Directory containing fold_0_train/test.csv ... fold_9_train/test.csv.",
    )
    parser.add_argument(
        "--vector-pkl",
        default=os.path.join(SCRIPT_DIR, "ptm_phos_dedup_processed_with_vectors.pkl"),
        help="PTM vector PKL used by original SWING pipeline.",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=os.path.join(PROJECT_ROOT, "esm2_t36_3B_UR50D_embeddings_ptm"),
        help="Directory containing SHA1-hashed per-sequence ESM embedding files.",
    )
    parser.add_argument(
        "--esm-layer",
        type=int,
        default=36,
        help="Layer key used under state['mean_representations'][layer].",
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.join(
            PROJECT_ROOT,
            "scripts",
            "ptm",
            "figure_plot",
            "SWING_baseline_ptm_for_figure.csv",
        ),
        help="Output CSV path with per-sample SWING predictions.",
    )
    parser.add_argument(
        "--metric-txt",
        default=os.path.join(
            PROJECT_ROOT,
            "Results",
            "result_SWING_baseline_on_ptm_with_predictions.txt",
        ),
        help="Output path for fold-level metrics text.",
    )
    parser.add_argument(
        "--merge-table",
        default=os.path.join(
            PROJECT_ROOT,
            "scripts",
            "ptm",
            "figure_plot",
            "PTM_for_figure.csv",
        ),
        help="If provided and exists, update SWING columns in this merged table.",
    )
    parser.add_argument(
        "--no-merge-table",
        action="store_true",
        help="Disable merge into PTM_for_figure.csv.",
    )
    return parser.parse_args()


def _sample_key(row: pd.Series) -> str:
    if "ppi_id" in row and "residue_id" in row:
        return f"{row['ppi_id']}||{row['residue_id']}"
    return f"{row.get('Uniprot', '')}||{row.get('Int_uniprot', '')}||{row.get('Position', '')}"


def _normalize_fold_df(df: pd.DataFrame) -> pd.DataFrame:
    if MUT_COL_ALT in df.columns:
        mut_col = MUT_COL_ALT
    elif MUT_COL_RAW in df.columns:
        mut_col = MUT_COL_RAW
    else:
        raise KeyError(f"Missing mutated sequence column: {MUT_COL_RAW}/{MUT_COL_ALT}")

    required = ["Uniprot", "Int_uniprot", "Position", "Y2H_score", "Seq", "Interactor_Seq", mut_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Fold CSV missing columns: {missing}")

    out = df.copy()
    out["__mut_col__"] = mut_col
    out["__sample_key__"] = out.apply(_sample_key, axis=1)
    out["Y2H_score"] = out["Y2H_score"].astype(int)
    return out


def _load_folds(fold_dir: str) -> Tuple[pd.DataFrame, Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]]:
    folds: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_parts: List[pd.DataFrame] = []

    for fold_idx in range(FOLD_COUNT):
        train_path = os.path.join(fold_dir, f"fold_{fold_idx}_train.csv")
        test_path = os.path.join(fold_dir, f"fold_{fold_idx}_test.csv")
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing fold file(s): {train_path}, {test_path}")

        train_df = _normalize_fold_df(pd.read_csv(train_path))
        test_df = _normalize_fold_df(pd.read_csv(test_path))
        folds[fold_idx] = (train_df.reset_index(drop=True), test_df.reset_index(drop=True))
        all_parts.extend([train_df, test_df])

    df_all = pd.concat(all_parts, ignore_index=True)
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["__sample_key__"]).reset_index(drop=True)
    print(f"Union rows: {before} -> unique by key: {len(df_all)}")
    return df_all, folds


def _seq_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


def _embedding_path(embed_dir: str, seq: str) -> str:
    return os.path.join(embed_dir, _seq_hash(seq) + ".pt")


def _load_embedding(path: str, layer: int) -> np.ndarray:
    state = torch.load(path, map_location="cpu")
    reps = state.get("mean_representations")
    if reps is None:
        raise KeyError(f"'mean_representations' not found in {path}")

    if layer in reps:
        vec = reps[layer]
    else:
        keys = list(reps.keys())
        if len(keys) == 1:
            vec = reps[keys[0]]
        else:
            raise KeyError(f"Layer {layer} not found in {path}; available layers: {keys}")
    return np.asarray(vec, dtype=np.float32)


def _build_union_features(
    df_all: pd.DataFrame,
    embeddings_dir: str,
    esm_layer: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, int]]:
    vec_cache: Dict[str, np.ndarray] = {}
    keep_rows: List[pd.Series] = []
    feats: List[np.ndarray] = []
    labels: List[int] = []
    missing_count = 0

    for _, row in df_all.iterrows():
        mut_col = row["__mut_col__"]
        mut_seq = row.get(mut_col)
        int_seq = row.get("Interactor_Seq")
        if pd.isna(mut_seq) or pd.isna(int_seq):
            missing_count += 1
            continue

        mut_seq = str(mut_seq)
        int_seq = str(int_seq)
        mut_path = _embedding_path(embeddings_dir, mut_seq)
        int_path = _embedding_path(embeddings_dir, int_seq)
        if not os.path.exists(mut_path) or not os.path.exists(int_path):
            missing_count += 1
            continue

        if mut_path not in vec_cache:
            vec_cache[mut_path] = _load_embedding(mut_path, esm_layer)
        if int_path not in vec_cache:
            vec_cache[int_path] = _load_embedding(int_path, esm_layer)

        v_mut = vec_cache[mut_path]
        v_int = vec_cache[int_path]
        if v_mut.shape != v_int.shape:
            raise ValueError(f"Embedding shape mismatch: {v_mut.shape} vs {v_int.shape}")

        keep_rows.append(row)
        feats.append((v_mut + v_int).astype(np.float32))
        labels.append(int(row["Y2H_score"]))

    kept_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    X = np.stack(feats, axis=0).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.float32)
    key_to_idx = {k: i for i, k in enumerate(kept_df["__sample_key__"].tolist())}

    print(f"Union features built: {X.shape} | dropped rows (missing seq/embedding): {missing_count}")
    return kept_df, X, y, key_to_idx


def _build_union_features_from_vector_pkl(
    df_all: pd.DataFrame,
    vector_pkl: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, int]]:
    if not os.path.exists(vector_pkl):
        raise FileNotFoundError(f"Vector PKL not found: {vector_pkl}")

    vec_df = pd.read_pickle(vector_pkl)
    need_cols = ["Uniprot", "Int_uniprot", "Site", "Effect", "Vectors"]
    miss = [c for c in need_cols if c not in vec_df.columns]
    if miss:
        raise KeyError(f"Vector PKL missing columns: {miss}")

    vec_df = vec_df.copy()
    vec_df["Uniprot"] = vec_df["Uniprot"].astype(str)
    vec_df["Int_uniprot"] = vec_df["Int_uniprot"].astype(str)
    vec_df["Site"] = pd.to_numeric(vec_df["Site"], errors="coerce").fillna(-1).astype(int)
    vec_df["Effect_l"] = vec_df["Effect"].astype(str).str.lower()

    vec_map: Dict[Tuple[str, str, int, str], np.ndarray] = {}
    for _, row in vec_df.iterrows():
        vec_raw = row["Vectors"]
        if vec_raw is None:
            continue
        key = (row["Uniprot"], row["Int_uniprot"], int(row["Site"]), row["Effect_l"])
        vec = np.asarray(vec_raw, dtype=np.float32)
        if vec.size == 0:
            continue
        vec_map[key] = vec

    keep_rows: List[pd.Series] = []
    feats: List[np.ndarray] = []
    labels: List[int] = []
    missing_count = 0

    for _, row in df_all.iterrows():
        effect = str(row.get("Effect", "")).lower()
        key = (
            str(row.get("Uniprot", "")),
            str(row.get("Int_uniprot", "")),
            int(row.get("Position", -1)),
            effect,
        )
        vec = vec_map.get(key)
        if vec is None:
            missing_count += 1
            continue
        keep_rows.append(row)
        feats.append(vec.astype(np.float32))
        labels.append(int(row["Y2H_score"]))

    if not feats:
        raise RuntimeError("No features matched from vector PKL. Check key mapping or input data.")

    kept_df = pd.DataFrame(keep_rows).reset_index(drop=True)
    X = np.stack(feats, axis=0).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.float32)
    key_to_idx = {k: i for i, k in enumerate(kept_df["__sample_key__"].tolist())}

    print(
        f"Union features built from vector PKL: {X.shape} | "
        f"matched={len(kept_df)} dropped={missing_count}"
    )
    return kept_df, X, y, key_to_idx


def _derive_before_after(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    before: List[str] = []
    after: List[str] = []
    for _, row in df.iterrows():
        mut_col = row["__mut_col__"]
        seq = str(row["Seq"]) if pd.notna(row["Seq"]) else ""
        mut_seq = str(row[mut_col]) if pd.notna(row[mut_col]) else seq
        pos = int(row["Position"]) if pd.notna(row["Position"]) else -1
        idx = pos - 1
        b = seq[idx] if 0 <= idx < len(seq) else ""
        a = mut_seq[idx] if 0 <= idx < len(mut_seq) else b
        before.append(b)
        after.append(a)
    return before, after


def _train_xgb(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    params = {
        "n_estimators": 375,
        "max_depth": 6,
        "learning_rate": 0.08966,
        "random_state": 42,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "tree_method": "hist",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    youdens_j = sensitivity + specificity - 1.0
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) > 1 and len(neg_probs) > 1:
        t_stat, p_value = stats.ttest_ind(pos_probs, neg_probs)
    else:
        t_stat, p_value = np.nan, np.nan

    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = float(auc(r_curve, p_curve))

    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ROC_AUC": _safe_auc(y_true, y_prob),
        "PR_AUC(AP)": pr_auc,
        "Youdens_J": float(youdens_j),
        "t_test_t": float(t_stat),
        "p_value": float(p_value),
    }


def _summarize_metrics(per_fold: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
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
    for k in keys:
        vals = [per_fold[i][k] for i in sorted(per_fold.keys())]
        arr = np.asarray(vals, dtype=np.float64)
        summary[f"{k}_mean"] = float(np.nanmean(arr))
        summary[f"{k}_std"] = float(np.nanstd(arr))
    return summary


def _write_metrics(path: str, per_fold: Dict[int, Dict[str, float]], summary: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    with open(path, "w", encoding="utf-8") as f:
        for i in range(FOLD_COUNT):
            f.write(f"=== Fold {i} ===\n")
            for k in metric_order:
                f.write(f"{k}: {per_fold[i][k]:.6f}\n")
            f.write("\n")
        f.write("=== Aggregated ===\n")
        for k in metric_order:
            f.write(f"{k}_mean: {summary[f'{k}_mean']:.6f}\n")
            f.write(f"{k}_std: {summary[f'{k}_std']:.6f}\n")


def _merge_into_table(merge_path: str, swing_df: pd.DataFrame) -> None:
    if not os.path.exists(merge_path):
        print(f"[warn] Merge table not found, skipped: {merge_path}")
        return

    base = pd.read_csv(merge_path)
    id_cols = [
        "Target_UPID",
        "Position",
        "Before_AA",
        "After_AA",
        "Y2H_score",
        "Interactor_UPID",
        "residue_id",
        "ppi_id",
    ]
    pred_cols = [
        "SWING_baseline_Y_score_0",
        "SWING_baseline_Y_score_1",
        "SWING_baseline_Y_predict",
    ]

    for c in pred_cols:
        if c in base.columns:
            base = base.drop(columns=[c])

    add_df = swing_df[id_cols + pred_cols].copy()
    merged = base.merge(add_df, on=id_cols, how="left")
    merged.to_csv(merge_path, index=False)
    print(f"Merged SWING columns into: {merge_path}")


def main() -> None:
    args = parse_args()
    print("PTM SWING baseline (save per-sample outputs)")
    print("=" * 80)
    print(f"Feature source: {args.feature_source}")

    df_all, folds = _load_folds(args.fold_dir)
    if args.feature_source == "pkl":
        kept_df, X_union, y_union, key_to_idx = _build_union_features_from_vector_pkl(
            df_all,
            args.vector_pkl,
        )
    else:
        kept_df, X_union, y_union, key_to_idx = _build_union_features(
            df_all,
            args.embeddings_dir,
            args.esm_layer,
        )
    print(f"Usable union rows: {len(kept_df)}")

    fold_metrics: Dict[int, Dict[str, float]] = {}
    oof_parts: List[pd.DataFrame] = []

    for fold_idx in range(FOLD_COUNT):
        train_df, test_df = folds[fold_idx]
        train_mask = train_df["__sample_key__"].isin(key_to_idx)
        test_mask = test_df["__sample_key__"].isin(key_to_idx)
        train_use = train_df.loc[train_mask].reset_index(drop=True)
        test_use = test_df.loc[test_mask].reset_index(drop=True)

        train_idx = np.array([key_to_idx[k] for k in train_use["__sample_key__"].tolist()], dtype=np.int64)
        test_idx = np.array([key_to_idx[k] for k in test_use["__sample_key__"].tolist()], dtype=np.int64)

        X_train = X_union[train_idx]
        y_train = y_union[train_idx]
        X_test = X_union[test_idx]
        y_test = y_union[test_idx]

        print(
            f"Fold {fold_idx}: train={len(train_use)} test={len(test_use)} "
            f"(dropped train/test={len(train_df)-len(train_use)}/{len(test_df)-len(test_use)})"
        )

        model = _train_xgb(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1].astype(np.float32)
        pred = (proba >= 0.5).astype(np.int32)

        m = _fold_metrics(y_test, pred, proba)
        fold_metrics[fold_idx] = m
        print(
            f"  ACC={m['ACC']:.4f} F1={m['F1']:.4f} "
            f"ROC_AUC={m['ROC_AUC']:.4f} PR_AUC={m['PR_AUC(AP)']:.4f}"
        )

        fold_out = test_use.copy()
        before, after = _derive_before_after(fold_out)
        fold_out["Target_UPID"] = fold_out["Uniprot"]
        fold_out["Interactor_UPID"] = fold_out["Int_uniprot"]
        fold_out["Before_AA"] = before
        fold_out["After_AA"] = after
        fold_out["SWING_baseline_Y_score_0"] = 1.0 - proba
        fold_out["SWING_baseline_Y_score_1"] = proba
        fold_out["SWING_baseline_Y_predict"] = pred
        oof_parts.append(fold_out)

    summary = _summarize_metrics(fold_metrics)
    print("\nAggregated metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")

    out_df = pd.concat(oof_parts, ignore_index=True)
    keep_cols = [
        "Target_UPID",
        "Position",
        "Before_AA",
        "After_AA",
        "Y2H_score",
        "Interactor_UPID",
        "residue_id",
        "ppi_id",
        "SWING_baseline_Y_score_0",
        "SWING_baseline_Y_score_1",
        "SWING_baseline_Y_predict",
    ]
    out_df = out_df[keep_cols].copy()

    id_cols = [
        "Target_UPID",
        "Position",
        "Before_AA",
        "After_AA",
        "Y2H_score",
        "Interactor_UPID",
        "residue_id",
        "ppi_id",
    ]
    out_df = (
        out_df.groupby(id_cols, as_index=False)[["SWING_baseline_Y_score_0", "SWING_baseline_Y_score_1"]]
        .mean()
        .reset_index(drop=True)
    )
    out_df["SWING_baseline_Y_predict"] = np.argmax(
        out_df[["SWING_baseline_Y_score_0", "SWING_baseline_Y_score_1"]].values,
        axis=1,
    ).astype(int)

    out_df = out_df.sort_values(["Target_UPID", "Position", "Interactor_UPID"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-sample CSV: {args.out_csv} (rows={len(out_df)})")

    _write_metrics(args.metric_txt, fold_metrics, summary)
    print(f"Saved metrics: {args.metric_txt}")

    if not args.no_merge_table:
        _merge_into_table(args.merge_table, out_df)


if __name__ == "__main__":
    main()
