#!/usr/bin/env python3
"""
Save per-sample PTM predictions for the ReCLIP.

This script keeps the original evaluation logic and additionally exports
OOF (out-of-fold) scores/predictions to a CSV for figure plotting.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import ptm_reclip_support as base


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_CACHE_ROOT = "/home/zhangzec/andrew_ptm/ptm_result_reclip"
if not os.path.exists(DEFAULT_CACHE_ROOT):
    DEFAULT_CACHE_ROOT = os.path.join(PROJECT_ROOT, base.CACHE_ROOT_PREFIX)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PTM ReCLIP and save per-sample outputs."
    )
    parser.add_argument(
        "--classifier",
        choices=["xgb", "mlp"],
        default="xgb",
        help="Classifier head to use for ten-fold evaluation.",
    )
    parser.add_argument(
        "--cache-root",
        default=DEFAULT_CACHE_ROOT,
        help="Root directory containing ReCLIP feature cache.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute features even if the cache already exists.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for the MLP classifier.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for the MLP classifier.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the MLP classifier.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for the MLP classifier.")
    parser.add_argument("--mlp-hidden", type=int, default=1024, help="Hidden dimension for the MLP classifier.")
    parser.add_argument(
        "--mlp-layers",
        type=int,
        choices=[2, 3],
        default=2,
        help="Number of linear layers (including output) for the MLP classifier.",
    )
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="Dropout rate inside the MLP classifier.")
    parser.add_argument(
        "--mint-layer",
        type=int,
        default=base.MINT_SELECT_LAYER,
        help="MINT cross-attention layer index for feature rebuild fallback.",
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.join(
            PROJECT_ROOT,
            "scripts",
            "ptm",
            "figure_plot",
            "ReCLIP_ptm_for_figure.csv",
        ),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--metric-txt",
        default=os.path.join(
            PROJECT_ROOT,
            base.RESULTS_DIR,
            "result_ptm_reclip_XGB_tenfold_with_predictions.txt",
        ),
        help="Where to save fold metrics.",
    )
    return parser.parse_args()


def _derive_before_after(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    mut_col = base.MUT_SEQ_COL_RAW if base.MUT_SEQ_COL_RAW in df.columns else base.TARGET_SEQ_COL
    seqs = df[base.TARGET_SEQ_COL].astype(str).tolist()
    mut_seqs = df[mut_col].astype(str).tolist()
    pos = df[base.POSITION_COL].astype(int).tolist()

    before: List[str] = []
    after: List[str] = []
    for s, ms, p in zip(seqs, mut_seqs, pos):
        idx = p - 1
        b = s[idx] if 0 <= idx < len(s) else ""
        a = ms[idx] if 0 <= idx < len(ms) else b
        before.append(b)
        after.append(a)
    return before, after


def _load_feature_cache_or_build(
    df_unique: pd.DataFrame,
    cache_root: str,
    force_rebuild: bool,
    mint_layer: int,
) -> Tuple[np.ndarray, List[str], str]:
    layer_tag = f"L{base.SELECT_LAYER}"
    cache_path = os.path.join(cache_root, f"layer_{layer_tag}", base.FEATURE_CACHE_SUBDIR, "all_samples.npz")
    if os.path.exists(cache_path) and not force_rebuild:
        arr = np.load(cache_path, allow_pickle=True)
        feats = arr["features"].astype(np.float32, copy=False)
        keys = arr["keys"].tolist()
        print(f"Loaded cached features: {feats.shape} from {cache_path}")
    else:
        print("Feature cache missing or rebuild requested, recomputing features...")
        base.CACHE_ROOT_PREFIX = cache_root
        hf_tok, hf_model, select_layer_idx, layer_tag = base.load_hf_esm2()
        mint_model, mint_alphabet, mint_layers = base.load_mint_model()
        mint_select_layer_idx = base._norm_single_layer(mint_layer, mint_layers)
        feats, keys, cache_path = base.prepare_feature_cache(
            df_unique,
            hf_tok,
            hf_model,
            select_layer_idx,
            layer_tag,
            mint_model,
            mint_alphabet,
            mint_select_layer_idx,
            f"M{mint_select_layer_idx}",
            force_rebuild=force_rebuild,
        )
        del hf_model
        del mint_model
        base._release_gpu_memory()

    key_set = set(keys)
    missing = 0
    for row in df_unique.itertuples(index=False):
        if base._make_sample_key(row) not in key_set:
            missing += 1
    if missing > 0:
        raise KeyError(f"Feature cache still missing {missing} unique samples: {cache_path}")
    return feats, keys, cache_path


def _train_xgb_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    if base.XGBClassifier is None:
        raise ImportError("xgboost>=1.6.0 is required.")

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

    use_gpu = base.DEVICE == "cuda"
    if use_gpu and not base.CUPY_AVAILABLE:
        print("[warn] CUDA is available but CuPy is missing, fallback to CPU XGBoost.")
        use_gpu = False

    if use_gpu:
        params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        params.update(tree_method="hist", predictor="auto")

    clf = base.XGBClassifier(**params)
    if use_gpu:
        X_train_dev = base.cp.asarray(X_train)
        y_train_dev = base.cp.asarray(y_train)
        X_test_dev = base.cp.asarray(X_test)
    else:
        X_train_dev = X_train
        y_train_dev = y_train
        X_test_dev = X_test

    clf.fit(X_train_dev, y_train_dev, verbose=False)
    proba = clf.predict_proba(X_test_dev)[:, 1]
    if use_gpu and isinstance(proba, base.cp.ndarray):
        proba = base.cp.asnumpy(proba)
    del clf
    if use_gpu:
        del X_train_dev, y_train_dev, X_test_dev
    base._release_gpu_memory()
    return np.asarray(proba, dtype=np.float32)


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


def _write_metrics(metric_path: str, fold_metrics: Dict[int, Dict[str, float]], summary: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
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
    with open(metric_path, "w", encoding="utf-8") as f:
        for fold_idx in base.FOLD_INDICES:
            f.write(f"=== Fold {fold_idx} ===\n")
            for key in metric_order:
                val = fold_metrics[fold_idx].get(key, float("nan"))
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


def main() -> None:
    args = parse_args()
    print("PTM Cross-Attention baseline - save per-sample outputs")
    print("=" * 80)

    fold_data = base.load_all_folds()
    df_all = base.concat_all_folds(fold_data)
    if df_all.empty:
        raise RuntimeError("No samples found in PTM folds.")

    if base.PPI_ID_COL in df_all.columns and base.RESIDUE_ID_COL in df_all.columns:
        subset_cols = [base.PPI_ID_COL, base.RESIDUE_ID_COL]
    else:
        subset_cols = [c for c in [base.TARGET_ID_COL, base.INTERACTOR_ID_COL, base.POSITION_COL] if c in df_all.columns]
    df_unique = df_all.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    print(f"Total rows: {len(df_all)} | unique rows for cache: {len(df_unique)}")

    features, keys, cache_path = _load_feature_cache_or_build(
        df_unique=df_unique,
        cache_root=args.cache_root,
        force_rebuild=args.force_rebuild,
        mint_layer=args.mint_layer,
    )
    key_to_idx = {k: i for i, k in enumerate(keys)}

    fold_metrics: Dict[int, Dict[str, float]] = {}
    all_metrics: List[Dict[str, float]] = []
    oof_parts: List[pd.DataFrame] = []

    for fold_idx in base.FOLD_INDICES:
        print(f"\n=== Fold {fold_idx} / {base.FOLD_INDICES[-1]} ({args.classifier.upper()}) ===")
        train_df, test_df = fold_data[fold_idx]
        X_train, y_train = base.features_from_df(train_df, features, key_to_idx)
        X_test, y_test = base.features_from_df(test_df, features, key_to_idx)

        if args.classifier == "xgb":
            proba = _train_xgb_and_predict(X_train, y_train, X_test)
        else:
            proba = _train_mlp_and_predict(X_train, y_train, X_test, args)

        pred = (proba >= 0.5).astype(np.int32)
        metrics_fold = base.compute_ten_metrics(y_test, pred, proba)
        fold_metrics[fold_idx] = metrics_fold
        all_metrics.append(metrics_fold)
        for k, v in metrics_fold.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        score1 = proba
        score0 = 1.0 - proba
        pred_orig = pred.astype(np.int32)

        fold_out = test_df.copy().reset_index(drop=True)
        before, after = _derive_before_after(fold_out)
        fold_out["Target_UPID"] = fold_out[base.TARGET_ID_COL]
        fold_out["Interactor_UPID"] = fold_out[base.INTERACTOR_ID_COL]
        fold_out["Before_AA"] = before
        fold_out["After_AA"] = after
        fold_out["Cross_Attention_baseline_Y_score_0"] = score0
        fold_out["Cross_Attention_baseline_Y_score_1"] = score1
        fold_out["Cross_Attention_baseline_Y_predict"] = pred_orig
        oof_parts.append(fold_out)

    summary = base.aggregate_metrics(all_metrics)
    print("\n=== Aggregated (mean/std) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

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
        "Cross_Attention_baseline_Y_score_0",
        "Cross_Attention_baseline_Y_score_1",
        "Cross_Attention_baseline_Y_predict",
    ]
    keep_cols = [c for c in keep_cols if c in out_df.columns]
    out_df = out_df[keep_cols].copy()

    id_cols = [c for c in keep_cols if c not in {
        "Cross_Attention_baseline_Y_score_0",
        "Cross_Attention_baseline_Y_score_1",
        "Cross_Attention_baseline_Y_predict",
    }]
    out_df = (
        out_df.groupby(id_cols, as_index=False)[
            ["Cross_Attention_baseline_Y_score_0", "Cross_Attention_baseline_Y_score_1"]
        ]
        .mean()
        .reset_index(drop=True)
    )
    out_df["Cross_Attention_baseline_Y_predict"] = np.argmax(
        out_df[["Cross_Attention_baseline_Y_score_0", "Cross_Attention_baseline_Y_score_1"]].values,
        axis=1,
    ).astype(int)

    sort_cols = [c for c in ["Target_UPID", "Position", "Interactor_UPID"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-sample CSV: {args.out_csv}")

    metric_path = args.metric_txt
    if args.classifier == "mlp":
        metric_path = metric_path.replace("_XGB_", "_MLP_")
    _write_metrics(metric_path, fold_metrics, summary)
    print(f"Saved metrics txt: {metric_path}")
    print(f"Feature cache path: {cache_path}")
