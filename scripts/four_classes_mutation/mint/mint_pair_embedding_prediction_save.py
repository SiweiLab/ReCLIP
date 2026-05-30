#!/usr/bin/env python3
"""
MINT pair-embedding prediction with per-sample outputs (IntAct_mutation_for_figure.csv format).
This script does NOT modify the original evaluation script.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

import mint_pair_embedding_prediction as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MINT pair-embedding prediction and save per-sample scores."
    )
    parser.add_argument(
        "--classifier",
        choices=["xgb", "mlp"],
        default="xgb",
        help="Classifier head to use for ten-fold evaluation.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute all features even if the feature cache exists.",
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
        "--out_csv",
        default=os.path.join(
            base.PROJECT_ROOT,
            "scripts",
            "four_classes_mutation",
            "figure_plot",
            "MINT_for_figure.csv",
        ),
        help="Output CSV path (based on IntAct_mutation_for_figure.csv format).",
    )
    parser.add_argument(
        "--reference_csv",
        default=os.path.join(
            base.PROJECT_ROOT,
            "scripts",
            "four_classes_mutation",
            "figure_plot",
            "IntAct_mutation_for_figure.csv",
        ),
        help="Reference CSV to define full sample rows/columns.",
    )
    return parser.parse_args()


def _train_xgb_and_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    if base.XGBClassifier is None:
        raise ImportError("xgboost>=1.6.0 is required.")

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
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        enable_categorical=False,
    )

    use_gpu = base.DEVICE == "cuda"
    if use_gpu:
        tree_method = "gpu_hist"
        predictor = "gpu_predictor"
        device_param = "cuda"
    else:
        tree_method = "hist"
        predictor = "auto"
        device_param = "cpu"

    clf = base.XGBClassifier(
        **base_params,
        tree_method=tree_method,
        predictor=predictor,
        device=device_param,
    )
    clf.fit(X_train, y_train, verbose=False)
    proba = clf.predict_proba(X_test)
    del clf
    return proba


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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        loss_epoch = base._mlp_epoch(model, train_loader, criterion, optimizer, base.DEVICE)
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} - loss: {loss_epoch:.4f}")

    proba = base._evaluate_model(model, eval_loader, base.DEVICE)
    del model
    base._release_gpu_memory()
    return proba


def main():
    args = parse_args()
    print("MINT Pair Embedding Prediction - Save per-sample outputs")
    print("=" * 80)

    df_all = base.load_dataset()
    if df_all.empty:
        raise RuntimeError("Dataset is empty.")
    print(f"Total samples: {len(df_all)}")

    skf = StratifiedKFold(n_splits=base.NUM_FOLDS, shuffle=True, random_state=base.RANDOM_SEED)

    mint_model, mint_alphabet, mint_layers = base.load_mint_model()
    features, keys, cache_path = base.prepare_feature_cache(
        df_all,
        mint_model,
        mint_alphabet,
        mint_layers,
        force_rebuild=args.force_rebuild,
    )
    key_to_idx = {k: i for i, k in enumerate(keys)}

    if base.DEVICE == "cuda":
        print("\nReleasing MINT GPU memory before classifier training...")
        del mint_model
        torch.cuda.empty_cache()

    n_samples = len(df_all)
    proba_all = np.zeros((n_samples, 4), dtype=np.float32)
    pred_all = np.zeros((n_samples,), dtype=np.int64)

    fold_metrics: List[Tuple[int, dict]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df_all, df_all[base.LABEL_COL])):
        print(f"\n=== Fold {fold_idx} / {base.NUM_FOLDS - 1} ({args.classifier.upper()}) ===")
        train_df = df_all.iloc[train_idx]
        test_df = df_all.iloc[test_idx]
        X_train, y_train = base.features_from_df(train_df, features, key_to_idx)
        X_test, y_test = base.features_from_df(test_df, features, key_to_idx)

        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)

        if args.classifier == "xgb":
            proba = _train_xgb_and_predict(X_train, y_train, X_test)
        else:
            proba = _train_mlp_and_predict(X_train, y_train, X_test, args)

        pred = np.argmax(proba, axis=1).astype(np.int64)
        proba_all[test_idx] = proba
        pred_all[test_idx] = pred

        metrics_fold = base.compute_13_metrics(y_test.astype(np.int64), pred, proba, num_classes=4)
        fold_metrics.append((fold_idx, metrics_fold))
        for k, v in metrics_fold.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        if base.DEVICE != "cuda":
            base._release_gpu_memory()

    print(f"\nFeature cache file: {cache_path}")

    key_cols = [
        base.TARGET_ID_COL,
        base.POSITION_COL,
        "Before_AA",
        "After_AA",
        base.INTERACTOR_ID_COL,
    ]

    score_cols = [f"MINT_baseline_Y_score_{i}" for i in range(4)]
    pred_df = df_all[key_cols].copy()
    for i in range(4):
        pred_df[f"MINT_baseline_Y_score_{i}"] = proba_all[:, i]
    pred_df = (
        pred_df.groupby(key_cols, as_index=False)[score_cols]
        .mean()
        .reset_index(drop=True)
    )
    pred_df["MINT_baseline_Y_predict"] = np.argmax(pred_df[score_cols].values, axis=1).astype(int)

    ref_df = pd.read_csv(args.reference_csv)
    for col in key_cols:
        if col not in ref_df.columns:
            raise KeyError(f"Reference CSV missing required column: {col}")

    out_df = ref_df.merge(pred_df, on=key_cols, how="left", validate="many_to_one")
    missing = out_df[score_cols].isna().any(axis=1).sum()
    if missing > 0:
        print(f"[warn] {missing} rows missing MINT scores after merge.")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
