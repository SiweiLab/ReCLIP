#!/usr/bin/env python3
"""
Train once on Set!=Test and evaluate once on Set==Test for peptide ReCLIP.
Save per-sample predictions to CSV for figure plotting.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_FIGURE_DIR = os.path.join(PROJECT_ROOT, "scripts", "peptide", "figure_plot")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import peptide_reclip_support as base

METRIC_ORDER = [
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


def _dataset_tag(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run peptide ReCLIP baseline and save Set==Test predictions."
    )
    parser.add_argument(
        "--data-set",
        required=True,
        help="CSV with Epitope/Sequence/Hit/Set columns.",
    )
    parser.add_argument(
        "--classifier",
        choices=["xgb", "mlp"],
        default="xgb",
        help="Classifier head for train/test evaluation.",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Feature cache root; default is dataset-specific under peptide_result_reclip.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute all features and overwrite cache.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for MLP.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for MLP.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for MLP.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for MLP.")
    parser.add_argument("--mlp-hidden", type=int, default=1024, help="Hidden dimension for MLP.")
    parser.add_argument(
        "--mlp-layers",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of linear layers (including output).",
    )
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="Dropout for MLP.")
    parser.add_argument(
        "--mint-layer",
        type=int,
        default=base.MINT_SELECT_LAYER,
        help="MINT cross-attention layer index (0-based, -1 for last).",
    )
    parser.add_argument(
        "--hf-layer",
        type=int,
        default=base.SELECT_LAYER,
        help="HuggingFace ESM2 layer index (0-based, -1 for last).",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Prediction CSV path. Default: scripts/peptide/figure_plot/ReCLIP_<dataset>_for_figure.csv",
    )
    parser.add_argument(
        "--metric-txt",
        default=None,
        help="Metric TXT path. Default: scripts/peptide/figure_plot/ReCLIP_<dataset>_metrics.txt",
    )
    parser.add_argument(
        "--xgb-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Execution device for XGBoost. 'auto' tries GPU first and falls back to CPU on failure.",
    )
    return parser.parse_args()


def _train_xgb_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    xgb_device: str = "auto",
) -> np.ndarray:
    if base.XGBClassifier is None:
        raise ImportError("xgboost is required.")

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    r = neg / max(1.0, pos)
    params = dict(
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
    )

    requested_gpu = xgb_device == "gpu" or (xgb_device == "auto" and base.DEVICE == "cuda")
    if requested_gpu and not base.CUPY_AVAILABLE:
        if xgb_device == "gpu":
            raise RuntimeError("Requested GPU XGBoost, but CuPy is unavailable.")
        print("[warn] CUDA available but CuPy missing, fallback to CPU XGBoost.")
        requested_gpu = False

    def _fit_predict(use_gpu: bool) -> np.ndarray:
        clf = None
        X_train_dev = X_train
        y_train_dev = y_train
        X_test_dev = X_test
        local_params = dict(params)
        if use_gpu:
            local_params.update(tree_method="gpu_hist", predictor="gpu_predictor")
        else:
            local_params.update(tree_method="hist", predictor="auto")

        try:
            clf = base.XGBClassifier(**local_params)
            if use_gpu:
                X_train_dev = base.cp.asarray(X_train)
                y_train_dev = base.cp.asarray(y_train)
                X_test_dev = base.cp.asarray(X_test)
            clf.fit(X_train_dev, y_train_dev, verbose=False)
            proba_local = clf.predict_proba(X_test_dev)[:, 1]
            if use_gpu and isinstance(proba_local, base.cp.ndarray):
                proba_local = base.cp.asnumpy(proba_local)
            return np.asarray(proba_local, dtype=np.float32)
        finally:
            if clf is not None:
                del clf
            if use_gpu:
                del X_train_dev, y_train_dev, X_test_dev
            base._release_gpu_memory()

    if requested_gpu:
        try:
            print("XGBoost device: GPU")
            return _fit_predict(use_gpu=True)
        except Exception as exc:
            if xgb_device == "gpu":
                raise
            print(f"[warn] GPU XGBoost failed ({type(exc).__name__}: {exc}). Falling back to CPU.")

    print("XGBoost device: CPU")
    return _fit_predict(use_gpu=False)


def _train_mlp_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test).float()),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = base.MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dim=args.mlp_hidden,
        num_layers=args.mlp_layers,
        dropout=args.mlp_dropout,
    ).to(base.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        loss_epoch = base._mlp_epoch(model, train_loader, criterion, optimizer, torch.device(base.DEVICE))
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"  Epoch {epoch + 1}/{args.epochs} - loss: {loss_epoch:.4f}")

    proba = base._evaluate_model(model, eval_loader, torch.device(base.DEVICE))
    del model
    base._release_gpu_memory()
    return np.asarray(proba, dtype=np.float32)


def _save_metrics(
    metric_path: str,
    metric_map: Dict[str, float],
    has_labeled_test: bool,
    notes: Optional[List[str]] = None,
) -> None:
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    with open(metric_path, "w", encoding="utf-8") as f:
        if not has_labeled_test:
            f.write("Note: Test set labels are missing; metrics are unavailable.\n")
        if notes:
            for note in notes:
                f.write(f"Note: {note}\n")
        for key in METRIC_ORDER:
            val = metric_map.get(key, float("nan"))
            f.write(f"{key}: {float(val):.6f}\n")


def _load_dataset_for_prediction(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading dataset: {data_path}")
    df_raw = pd.read_csv(data_path)
    if base.MUT_SEQ_COL_RAW and base.MUT_SEQ_COL_RAW in df_raw.columns:
        df_raw = df_raw.rename(columns={base.MUT_SEQ_COL_RAW: base.MUT_SEQ_COL})

    required = [base.MUT_SEQ_COL, base.INTERACTOR_SEQ_COL, base.SET_COL]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise KeyError(f"Dataset is missing columns: {missing}")
    if base.LABEL_COL not in df_raw.columns:
        raise KeyError(f"Dataset is missing label column: {base.LABEL_COL}")

    df = df_raw.copy()
    for col in [base.MUT_SEQ_COL, base.INTERACTOR_SEQ_COL, base.SET_COL]:
        df[col] = df[col].astype(str).str.strip()

    non_empty_seq = (df[base.MUT_SEQ_COL] != "") & (df[base.INTERACTOR_SEQ_COL] != "")
    dropped_empty = int((~non_empty_seq).sum())
    if dropped_empty > 0:
        print(f"[warn] Dropping {dropped_empty} rows with empty epitope/interactor sequences.")
    df = df.loc[non_empty_seq].copy()

    if base.POSITION_COL not in df.columns:
        df[base.POSITION_COL] = df[base.MUT_SEQ_COL].apply(base._default_mutation_position)
    else:
        pos_num = pd.to_numeric(df[base.POSITION_COL], errors="coerce")
        fallback = df[base.MUT_SEQ_COL].apply(base._default_mutation_position)
        df[base.POSITION_COL] = pos_num.fillna(fallback).astype(int)

    seq_lengths = df[base.MUT_SEQ_COL].str.len()
    valid_pos = (df[base.POSITION_COL] >= 1) & (df[base.POSITION_COL] <= seq_lengths)
    dropped_pos = int((~valid_pos).sum())
    if dropped_pos > 0:
        print(f"[warn] Dropping {dropped_pos} rows with pseudo positions outside sequence length.")
    df = df.loc[valid_pos].copy()

    set_norm = df[base.SET_COL].astype(str).str.strip().str.lower()
    is_test = set_norm == "test"

    train_df = df.loc[~is_test].copy()
    test_df = df.loc[is_test].copy()

    train_labels = pd.to_numeric(train_df[base.LABEL_COL], errors="coerce")
    valid_train = train_labels.notna()
    dropped_train_label = int((~valid_train).sum())
    if dropped_train_label > 0:
        print(f"[warn] Dropping {dropped_train_label} training rows with missing/non-numeric labels.")
    train_df = train_df.loc[valid_train].copy()
    train_df[base.LABEL_COL] = pd.to_numeric(train_df[base.LABEL_COL], errors="raise").astype(int)

    if not test_df.empty:
        test_df[base.LABEL_COL] = pd.to_numeric(test_df[base.LABEL_COL], errors="coerce")

    out = pd.concat([train_df, test_df], ignore_index=True)
    print(f"  rows: {len(df_raw)} -> {len(out)} | dropped: {len(df_raw) - len(out)}")
    return out.reset_index(drop=True)


def _split_train_test_by_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if base.SET_COL not in df.columns:
        raise KeyError(f"Dataset is missing the {base.SET_COL} column required for Train/Test splitting.")
    set_values = df[base.SET_COL].astype(str).str.strip().str.lower()
    test_mask = set_values == "test"
    train_df = df.loc[~test_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training split (Set != 'Test') is empty.")
    if test_df.empty:
        raise ValueError("Test split (Set == 'Test') is empty.")
    print(f"Train rows (Set!=Test): {len(train_df)} | Test rows (Set==Test): {len(test_df)}")
    return train_df, test_df


def _supported_residue_tokens(alphabet: object) -> Set[str]:
    tok_to_idx = getattr(alphabet, "tok_to_idx", {})
    return {str(tok).upper() for tok in tok_to_idx.keys() if isinstance(tok, str) and len(tok) == 1 and tok.isalpha()}


def _invalid_sequence_chars(seq: str, allowed_tokens: Set[str]) -> List[str]:
    seq_norm = str(seq).strip().upper().replace("J", "L")
    invalid = sorted({ch for ch in seq_norm if ch and ch not in allowed_tokens})
    return invalid


def _partition_supported_rows(
    df: pd.DataFrame,
    allowed_tokens: Set[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    keep_mask: List[bool] = []
    invalid_details: List[str] = []
    for row in df.itertuples(index=False):
        detail_parts: List[str] = []
        for col in [base.MUT_SEQ_COL, base.INTERACTOR_SEQ_COL]:
            invalid_chars = _invalid_sequence_chars(getattr(row, col), allowed_tokens)
            if invalid_chars:
                detail_parts.append(f"{col}:{''.join(invalid_chars)}")
        keep_mask.append(len(detail_parts) == 0)
        invalid_details.append(";".join(detail_parts))

    mask = pd.Series(keep_mask, index=df.index)
    supported_df = df.loc[mask].copy().reset_index(drop=True)
    excluded_df = df.loc[~mask].copy().reset_index(drop=True)
    if not excluded_df.empty:
        excluded_df["Prediction_Status"] = "excluded_invalid_sequence"
        excluded_df["Unsupported_Detail"] = [detail for keep, detail in zip(keep_mask, invalid_details) if not keep]
    return supported_df, excluded_df


def _features_and_optional_labels(
    df: pd.DataFrame,
    features: np.ndarray,
    key_to_idx: Dict[str, int],
    require_labels: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    idxs = []
    for row in df.itertuples(index=False):
        key = base._make_sample_key(row)
        if key not in key_to_idx:
            raise KeyError(f"Feature cache missing key: {key}")
        idxs.append(key_to_idx[key])
    X = features[idxs]

    labels_num = pd.to_numeric(df[base.LABEL_COL], errors="coerce")
    if require_labels:
        if not labels_num.notna().all():
            raise ValueError("Training set contains missing labels.")
        y = labels_num.astype(int).to_numpy(dtype=np.float32)
        return X, y

    if labels_num.notna().all():
        y = labels_num.astype(int).to_numpy(dtype=np.float32)
    else:
        y = None
    return X, y


def main() -> None:
    args = parse_args()
    tag = _dataset_tag(args.data_set)
    clf_tag = args.classifier.upper()

    os.makedirs(DEFAULT_FIGURE_DIR, exist_ok=True)
    out_csv = args.out_csv or os.path.join(
        DEFAULT_FIGURE_DIR, f"Cross_Attention_{tag}_{clf_tag}_for_figure.csv"
    )
    metric_txt = args.metric_txt or os.path.join(
        DEFAULT_FIGURE_DIR, f"Cross_Attention_{tag}_{clf_tag}_metrics.txt"
    )

    cache_root = args.cache_root or os.path.join(PROJECT_ROOT, "peptide_result_reclip", tag)
    base.CACHE_ROOT_PREFIX = cache_root

    print("Peptide ESM2+MINT Cross-Attention (single train/test)")
    print("=" * 80)
    print(f"Dataset: {args.data_set}")
    print(f"Cache root: {cache_root}")

    df_all = _load_dataset_for_prediction(args.data_set)
    if df_all.empty:
        raise RuntimeError("Dataset is empty.")
    train_df_full, test_df_full = _split_train_test_by_set(df_all)

    hf_tok, hf_model, select_layer_idx, layer_tag = base.load_hf_esm2(args.hf_layer)
    mint_model, mint_alphabet, mint_layers = base.load_mint_model()
    mint_select_layer_idx = base._norm_single_layer(args.mint_layer, mint_layers)
    mint_layer_tag = f"M{mint_select_layer_idx}"

    allowed_tokens = _supported_residue_tokens(mint_alphabet)
    train_df, train_excluded = _partition_supported_rows(train_df_full, allowed_tokens)
    test_df, test_excluded = _partition_supported_rows(test_df_full, allowed_tokens)

    metric_notes: List[str] = []
    if not train_excluded.empty:
        msg = f"Excluded {len(train_excluded)} training rows with unsupported sequence characters."
        metric_notes.append(msg)
        print(f"[warn] {msg}")
    if not test_excluded.empty:
        msg = f"Excluded {len(test_excluded)} test rows with unsupported sequence characters; saved with NaN predictions."
        metric_notes.append(msg)
        print(f"[warn] {msg}")

    if train_df.empty:
        raise RuntimeError("No supported training rows remain after sequence validation.")
    if test_df.empty:
        raise RuntimeError("No supported test rows remain after sequence validation.")

    df_all_supported = pd.concat([train_df, test_df], ignore_index=True)

    features, keys, cache_path = base.prepare_feature_cache(
        df_all_supported,
        hf_tok,
        hf_model,
        select_layer_idx,
        layer_tag,
        mint_model,
        mint_alphabet,
        mint_select_layer_idx,
        mint_layer_tag,
        force_rebuild=args.force_rebuild,
    )
    key_to_idx = {k: i for i, k in enumerate(keys)}

    if base.DEVICE == "cuda":
        del hf_model
        del mint_model
        torch.cuda.empty_cache()

    X_train, y_train = _features_and_optional_labels(train_df, features, key_to_idx, require_labels=True)
    X_test, y_test = _features_and_optional_labels(test_df, features, key_to_idx, require_labels=False)

    if args.classifier == "xgb":
        proba = _train_xgb_and_predict(X_train, y_train, X_test, xgb_device=args.xgb_device)
    else:
        proba = _train_mlp_and_predict(X_train, y_train, X_test, args)

    pred = (proba >= 0.5).astype(np.int32)
    has_labeled_test = y_test is not None
    if has_labeled_test:
        metrics_map = base.compute_ten_metrics(y_test, pred, proba)
    else:
        metrics_map = {k: float("nan") for k in METRIC_ORDER}

    out_df = test_df.copy()
    out_df["True Y"] = pd.to_numeric(test_df[base.LABEL_COL], errors="coerce")
    out_df["Predicted Y"] = pred.astype(int)
    out_df["Predicted Probabilities Y"] = proba.astype(np.float32)
    out_df["Prediction_Status"] = "ok"
    if not test_excluded.empty:
        excluded_out = test_excluded.copy()
        excluded_out["True Y"] = pd.to_numeric(excluded_out[base.LABEL_COL], errors="coerce")
        excluded_out["Predicted Y"] = np.nan
        excluded_out["Predicted Probabilities Y"] = np.nan
        out_df = pd.concat([out_df, excluded_out], ignore_index=True, sort=False)
    out_df = out_df.sort_values(
        by=["Predicted Probabilities Y", "Prediction_Status"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    _save_metrics(metric_txt, metrics_map, has_labeled_test, notes=metric_notes)

    print(f"Prediction CSV saved: {out_csv}")
    print(f"Metrics TXT saved: {metric_txt}")
    print(f"Feature cache file: {cache_path}")
    for k in METRIC_ORDER:
        v = metrics_map.get(k, float("nan"))
        if isinstance(v, (int, float, np.floating)):
            print(f"{k}: {float(v):.6f}")
        else:
            print(f"{k}: {v}")
