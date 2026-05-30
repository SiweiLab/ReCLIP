#!/usr/bin/env python3
"""
Run PTM ReCLIP and save per-sample outputs for plotting.

This uses the validated ReCLIP feature layout:
  - HuggingFace ESM2 layer 32 per-head context features
  - per-head self-attention top-5 residues around the PTM/mutation position
  - MINT partner residue representations weighted by MINT cross-chain attention
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
MUTATION_RECLIP_DIR = PROJECT_ROOT / "scripts" / "four_classes_mutation" / "ReCLIP"
FIGURE_DIR = PROJECT_ROOT / "scripts" / "ptm" / "figure_plot"
RESULTS_DIR = PROJECT_ROOT / "Results"

for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ptm_reclip_support as base
import ptm_reclip_save_support as old_save


def _load_mutation_reclip_common():
    common_path = MUTATION_RECLIP_DIR / "common.py"
    spec = importlib.util.spec_from_file_location("ptm_reclip_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load mutation ReCLIP common module from {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


new_common = _load_mutation_reclip_common()
new_common.WORK_DIR = SCRIPT_DIR

# Make PTM helpers robust when invoked outside the repository root.
base.FOLD_DIR = str(PROJECT_ROOT / "data" / "ptm" / "ten_folds")
base.MINT_CFG_PATH = str(PROJECT_ROOT / "mint" / "data" / "esm2_t33_650M_UR50D.json")
base.MINT_CHECKPOINT_PATH = str(PROJECT_ROOT / "mint" / "mint.ckpt")

ESM_LAYER = 32
MINT_ATTN_LAYER = -1
PARTNER_MINT_LAYER = 32
K_SEQ = 5
TARGET_FEATURE_SPACE = "context"
USE_ATTENTION_LAYERNORM = True
TARGET_AGGREGATION = "flatten"
METHOD_NAME = "ReCLIP"
SCORE0_COL = "Cross_Attention_baseline_Y_score_0"
SCORE1_COL = "Cross_Attention_baseline_Y_score_1"
PREDICT_COL = "Cross_Attention_baseline_Y_predict"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PTM ReCLIP ten-fold evaluation and save figure-ready outputs. "
            "Defaults: L32, all heads, top5 per head, context+LN, MINT partner representations."
        )
    )
    parser.add_argument("--classifier", choices=["xgb", "mlp"], default="xgb", help="Classifier head.")
    parser.add_argument(
        "--cache-root",
        default=str(PROJECT_ROOT / "ptm_result_reclip"),
        help="Root directory for final feature matrix cache.",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Recompute final feature matrix cache.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for MLP.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs for MLP.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for MLP.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for MLP.")
    parser.add_argument("--mlp-hidden", type=int, default=1024, help="Hidden dimension for MLP.")
    parser.add_argument("--mlp-layers", type=int, choices=[2, 3], default=2, help="MLP linear layer count.")
    parser.add_argument("--mlp-dropout", type=float, default=0.2, help="Dropout for MLP.")
    parser.add_argument("--hf-layer", type=int, default=ESM_LAYER, help="HuggingFace ESM2 layer index.")
    parser.add_argument(
        "--mint-attn-layer",
        type=int,
        default=MINT_ATTN_LAYER,
        help="MINT attention layer index used for partner weighting.",
    )
    parser.add_argument(
        "--partner-mint-layer",
        type=int,
        default=PARTNER_MINT_LAYER,
        help="MINT representation layer index used for partner embeddings.",
    )
    parser.add_argument("--top-k", type=int, default=K_SEQ, help="Top residues selected per attention head.")
    parser.add_argument(
        "--out-csv",
        default=str(FIGURE_DIR / "ReCLIP_ptm_for_figure.csv"),
        help="Figure-ready output CSV path.",
    )
    parser.add_argument(
        "--metric-txt",
        default=str(RESULTS_DIR / "result_ptm_reclip_XGB_tenfold_with_predictions.txt"),
        help="Fold metric text output path.",
    )
    parser.add_argument(
        "--metric-json",
        default=str(RESULTS_DIR / "result_ptm_reclip_XGB_tenfold_metadata.json"),
        help="Metadata JSON output path.",
    )
    parser.add_argument(
        "--xgb-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Execution device for XGBoost.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Override ESM2/MINT model device.",
    )
    return parser.parse_args()


def _release_gpu_memory() -> None:
    base._release_gpu_memory()
    new_common._release_gpu_memory()


def _atomic_save_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as f:
            np.savez(f, **arrays)
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _load_feature_matrix_cache(path: Path) -> Optional[Tuple[np.ndarray, List[str]]]:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        features = np.asarray(data["features"], dtype=np.float32)
        keys = data["keys"].tolist()
        return features, keys
    except (OSError, ValueError, EOFError, KeyError):
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _save_feature_matrix_cache(path: Path, features: np.ndarray, keys: List[str]) -> None:
    _atomic_save_npz(path, features=np.asarray(features, dtype=np.float32), keys=np.asarray(keys, dtype=object))


def _feature_cache_path(cache_root: Path, layer_tag: str, mint_attn_tag: str, partner_mint_tag: str, top_k: int) -> Path:
    setting_tag = (
        f"layer_{layer_tag}_hall_{TARGET_AGGREGATION}_{TARGET_FEATURE_SPACE}_ln_k{top_k}_"
        f"mintattn_{mint_attn_tag}_partner_{partner_mint_tag}_partnerL1v2"
    )
    return cache_root / setting_tag / base.FEATURE_CACHE_SUBDIR / "all_samples.npz"


def build_attention_target_feature(context_aa: np.ndarray, attn_row: np.ndarray, top_k: int) -> np.ndarray:
    num_heads = int(context_aa.shape[0])
    head_dim = int(context_aa.shape[-1])
    slots = np.zeros((num_heads, int(top_k), head_dim), dtype=np.float32)
    seq_len = min(int(context_aa.shape[1]), int(attn_row.shape[1]))
    if seq_len <= 0:
        return slots.reshape(-1).astype(np.float32, copy=False)

    for head_idx in range(num_heads):
        head_scores = attn_row[head_idx, :seq_len]
        top_idx = np.argsort(-head_scores)[: min(int(top_k), seq_len)].astype(np.int64, copy=False)
        take = min(int(top_k), int(top_idx.size))
        if take > 0:
            slots[head_idx, :take, :] = context_aa[head_idx, top_idx[:take], :].astype(np.float32, copy=False)
    return slots.reshape(-1).astype(np.float32, copy=False)


def aggregate_partner_feature(partner_repr: np.ndarray, mint_weights: np.ndarray, partner_dim: int) -> np.ndarray:
    partner_len = min(int(partner_repr.shape[0]), int(mint_weights.shape[0]))
    if partner_len <= 0:
        return np.zeros((partner_dim,), dtype=np.float32)
    weights = mint_weights[:partner_len].astype(np.float32, copy=False)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        weights = np.full((partner_len,), 1.0 / partner_len, dtype=np.float32)
    else:
        weights = weights / total
    return (partner_repr[:partner_len] * weights[:, None]).sum(axis=0).astype(np.float32, copy=False)


def build_features(
    df: pd.DataFrame,
    hf_tok,
    hf_model,
    select_layer_idx: int,
    mint_model,
    alphabet,
    mint_attn_layer_idx: int,
    partner_mint_layer_idx: int,
    top_k: int,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    num_heads = int(hf_model.encoder.layer[select_layer_idx].attention.self.num_attention_heads)
    head_dim = int(hf_model.encoder.layer[select_layer_idx].attention.self.attention_head_size)
    partner_dim = int(getattr(mint_model, "embed_dim", hf_model.config.hidden_size))
    target_dim = int(num_heads * top_k * head_dim)
    feature_dim = int(target_dim + partner_dim)

    features = np.zeros((len(df), feature_dim), dtype=np.float32)
    keys: List[str] = []
    iterator = tqdm(df.itertuples(index=False), total=len(df), desc="Build PTM ReCLIP features")

    for row_idx, row in enumerate(iterator):
        target_seq_full = getattr(row, base.TARGET_SEQ_COL)
        partner_seq_full = getattr(row, base.INTERACTOR_SEQ_COL)
        target_pos_orig = int(getattr(row, base.POSITION_COL))

        target_seq_trim, partner_seq_trim, target_pos_trim = base.truncate_pair_for_models(
            target_seq_full,
            partner_seq_full,
            target_pos_orig,
        )

        context_aa = new_common.load_or_compute_target_residue_features(
            target_seq_trim,
            select_layer_idx,
            hf_tok,
            hf_model,
            feature_space=TARGET_FEATURE_SPACE,
            use_attention_layernorm=USE_ATTENTION_LAYERNORM,
        )
        _release_gpu_memory()

        attn_row = new_common.load_or_compute_attention_row(
            target_seq_trim,
            target_pos_trim,
            select_layer_idx,
            hf_tok,
            hf_model,
            use_attention_layernorm=USE_ATTENTION_LAYERNORM,
        )
        _release_gpu_memory()

        partner_repr, mint_weights = new_common.load_or_compute_mint_partner_representations_and_weights(
            target_seq_trim,
            partner_seq_trim,
            target_pos_trim,
            mint_model,
            alphabet,
            partner_mint_layer_idx=partner_mint_layer_idx,
            mint_attn_layer_idx=mint_attn_layer_idx,
        )
        _release_gpu_memory()

        target_feat = build_attention_target_feature(context_aa, attn_row, top_k=top_k)
        partner_feat = aggregate_partner_feature(partner_repr, mint_weights, partner_dim)
        features[row_idx] = np.concatenate([target_feat, partner_feat], axis=0)
        keys.append(base._make_sample_key(row))

    dims = {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "target_feature_dim": target_dim,
        "partner_feature_dim": partner_dim,
        "feature_dim": feature_dim,
    }
    return features, keys, dims


def prepare_feature_cache(
    df_unique: pd.DataFrame,
    hf_tok,
    hf_model,
    select_layer_idx: int,
    layer_tag: str,
    mint_model,
    alphabet,
    mint_attn_layer_idx: int,
    mint_attn_tag: str,
    partner_mint_layer_idx: int,
    partner_mint_tag: str,
    top_k: int,
    cache_root: Path,
    force_rebuild: bool,
) -> Tuple[np.ndarray, List[str], Path, Dict[str, int]]:
    num_heads = int(hf_model.encoder.layer[select_layer_idx].attention.self.num_attention_heads)
    head_dim = int(hf_model.encoder.layer[select_layer_idx].attention.self.attention_head_size)
    partner_dim = int(getattr(mint_model, "embed_dim", hf_model.config.hidden_size))
    target_dim = int(num_heads * top_k * head_dim)
    feature_dim = int(target_dim + partner_dim)

    cache_path = _feature_cache_path(cache_root, layer_tag, mint_attn_tag, partner_mint_tag, top_k)
    if force_rebuild and cache_path.exists():
        cache_path.unlink()

    expected_keys = [base._make_sample_key(row) for row in df_unique.itertuples(index=False)]
    cached = _load_feature_matrix_cache(cache_path)
    if cached is not None:
        cached_features, cached_keys = cached
        if (
            cached_features.ndim == 2
            and cached_features.shape == (len(df_unique), feature_dim)
            and cached_keys == expected_keys
        ):
            print(f"Feature matrix cache hit: {cache_path}")
            dims = {
                "num_heads": num_heads,
                "head_dim": head_dim,
                "target_feature_dim": target_dim,
                "partner_feature_dim": partner_dim,
                "feature_dim": feature_dim,
            }
            return cached_features, cached_keys, cache_path, dims
        print("Feature matrix cache mismatch detected; rebuilding.")

    features, keys, dims = build_features(
        df=df_unique,
        hf_tok=hf_tok,
        hf_model=hf_model,
        select_layer_idx=select_layer_idx,
        mint_model=mint_model,
        alphabet=alphabet,
        mint_attn_layer_idx=mint_attn_layer_idx,
        partner_mint_layer_idx=partner_mint_layer_idx,
        top_k=top_k,
    )
    _save_feature_matrix_cache(cache_path, features, keys)
    print(f"Feature matrix saved: {cache_path}")
    return features, keys, cache_path, dims


def _train_xgb_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    xgb_device: str,
) -> np.ndarray:
    if base.XGBClassifier is None:
        raise ImportError("xgboost is required for this script.")

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
            proba = clf.predict_proba(X_test_dev)[:, 1]
            if use_gpu and isinstance(proba, base.cp.ndarray):
                proba = base.cp.asnumpy(proba)
            return np.asarray(proba, dtype=np.float32)
        finally:
            if clf is not None:
                del clf
            if use_gpu:
                del X_train_dev, y_train_dev, X_test_dev
            _release_gpu_memory()

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
    return old_save._train_mlp_and_predict(X_train, y_train, X_test, args)


def _build_unique_dataframe(fold_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    df_all = base.concat_all_folds(fold_data)
    if df_all.empty:
        raise RuntimeError("No samples found in PTM folds.")
    if base.PPI_ID_COL in df_all.columns and base.RESIDUE_ID_COL in df_all.columns:
        subset_cols = [base.PPI_ID_COL, base.RESIDUE_ID_COL]
    else:
        subset_cols = [c for c in [base.TARGET_ID_COL, base.INTERACTOR_ID_COL, base.POSITION_COL] if c in df_all.columns]
        if not subset_cols:
            subset_cols = df_all.columns.tolist()
    df_unique = df_all.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    print(f"Total fold rows: {len(df_all)} | unique rows for cache: {len(df_unique)}")
    return df_unique


def _save_figure_csv(oof_parts: List[pd.DataFrame], out_csv: Path) -> None:
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
        SCORE0_COL,
        SCORE1_COL,
        PREDICT_COL,
    ]
    keep_cols = [c for c in keep_cols if c in out_df.columns]
    out_df = out_df[keep_cols].copy()

    id_cols = [c for c in keep_cols if c not in {SCORE0_COL, SCORE1_COL, PREDICT_COL}]
    out_df = (
        out_df.groupby(id_cols, as_index=False)[[SCORE0_COL, SCORE1_COL]]
        .mean()
        .reset_index(drop=True)
    )
    out_df[PREDICT_COL] = np.argmax(out_df[[SCORE0_COL, SCORE1_COL]].values, axis=1).astype(int)
    sort_cols = [c for c in ["Target_UPID", "Position", "Interactor_UPID"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved per-sample CSV: {out_csv}")


def main() -> None:
    args = parse_args()
    if args.device != "auto":
        base.DEVICE = args.device
        new_common.DEVICE = args.device

    print("PTM ReCLIP - save per-sample outputs")
    print("=" * 80)
    print(f"Cache root: {args.cache_root}")
    print(f"Model device: {base.DEVICE}")

    fold_data = base.load_all_folds()
    df_unique = _build_unique_dataframe(fold_data)

    hf_tok, hf_model, select_layer_idx, layer_tag = new_common.load_hf_esm2(args.hf_layer)
    mint_model, mint_alphabet, mint_layers = new_common.load_mint_model()
    mint_attn_layer_idx = new_common._norm_single_layer(args.mint_attn_layer, mint_layers)
    partner_mint_layer_idx = new_common._norm_single_layer(args.partner_mint_layer, mint_layers)
    mint_attn_tag = f"M{mint_attn_layer_idx}"
    partner_mint_tag = f"M{partner_mint_layer_idx}"

    features, keys, cache_path, feature_dims = prepare_feature_cache(
        df_unique=df_unique,
        hf_tok=hf_tok,
        hf_model=hf_model,
        select_layer_idx=select_layer_idx,
        layer_tag=layer_tag,
        mint_model=mint_model,
        alphabet=mint_alphabet,
        mint_attn_layer_idx=mint_attn_layer_idx,
        mint_attn_tag=mint_attn_tag,
        partner_mint_layer_idx=partner_mint_layer_idx,
        partner_mint_tag=partner_mint_tag,
        top_k=args.top_k,
        cache_root=Path(args.cache_root),
        force_rebuild=args.force_rebuild,
    )
    key_to_idx = {k: i for i, k in enumerate(keys)}

    if base.DEVICE == "cuda":
        print("Releasing ESM2/MINT GPU memory before classifier training ...")
        del hf_model
        del mint_model
        torch.cuda.empty_cache()

    fold_metrics: Dict[int, Dict[str, float]] = {}
    all_metrics: List[Dict[str, float]] = []
    oof_parts: List[pd.DataFrame] = []

    for fold_idx in base.FOLD_INDICES:
        print(f"\n=== Fold {fold_idx} / {base.FOLD_INDICES[-1]} ({args.classifier.upper()}) ===")
        train_df, test_df = fold_data[fold_idx]
        X_train, y_train = base.features_from_df(train_df, features, key_to_idx)
        X_test, y_test = base.features_from_df(test_df, features, key_to_idx)

        if args.classifier == "xgb":
            proba = _train_xgb_and_predict(X_train, y_train, X_test, xgb_device=args.xgb_device)
        else:
            proba = _train_mlp_and_predict(X_train, y_train, X_test, args)

        pred = (proba >= 0.5).astype(np.int32)
        metrics_fold = base.compute_ten_metrics(y_test, pred, proba)
        fold_metrics[fold_idx] = metrics_fold
        all_metrics.append(metrics_fold)
        for key in METRIC_ORDER:
            if key in metrics_fold:
                print(f"{key}: {float(metrics_fold[key]):.6f}")

        fold_out = test_df.copy().reset_index(drop=True)
        before, after = old_save._derive_before_after(fold_out)
        fold_out["Target_UPID"] = fold_out[base.TARGET_ID_COL]
        fold_out["Interactor_UPID"] = fold_out[base.INTERACTOR_ID_COL]
        fold_out["Before_AA"] = before
        fold_out["After_AA"] = after
        fold_out[SCORE0_COL] = 1.0 - proba
        fold_out[SCORE1_COL] = proba
        fold_out[PREDICT_COL] = pred.astype(np.int32)
        oof_parts.append(fold_out)

        del X_train, X_test, y_train, y_test, proba, pred
        _release_gpu_memory()

    summary = base.aggregate_metrics(all_metrics)
    print("\n=== Aggregated (mean/std) ===")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")

    out_csv = Path(args.out_csv)
    metric_txt = Path(args.metric_txt)
    metric_json = Path(args.metric_json)
    if args.classifier == "mlp":
        metric_txt = Path(str(metric_txt).replace("_XGB_", "_MLP_"))
        metric_json = Path(str(metric_json).replace("_XGB_", "_MLP_"))

    _save_figure_csv(oof_parts, out_csv)
    old_save._write_metrics(str(metric_txt), fold_metrics, summary)
    print(f"Saved metrics txt: {metric_txt}")

    meta = {
        "task": "ptm",
        "method": "ReCLIP",
        "classifier": args.classifier,
        "device": base.DEVICE,
        "esm_layer_arg": int(args.hf_layer),
        "select_layer_idx": int(select_layer_idx),
        "layer_tag": str(layer_tag),
        "mint_attn_layer_arg": int(args.mint_attn_layer),
        "mint_attn_layer_idx": int(mint_attn_layer_idx),
        "partner_mint_layer_arg": int(args.partner_mint_layer),
        "partner_mint_layer_idx": int(partner_mint_layer_idx),
        "top_k": int(args.top_k),
        "target_feature_space": TARGET_FEATURE_SPACE,
        "use_attention_layernorm": bool(USE_ATTENTION_LAYERNORM),
        "target_aggregation": TARGET_AGGREGATION,
        "use_all_heads": True,
        "feature_dims": feature_dims,
        "feature_cache_path": str(cache_path),
        "figure_csv_path": str(out_csv),
        "metric_txt_path": str(metric_txt),
        "summary": summary,
    }
    metric_json.parent.mkdir(parents=True, exist_ok=True)
    metric_json.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved metadata json: {metric_json}")
    print(f"Feature cache path: {cache_path}")


if __name__ == "__main__":
    main()
