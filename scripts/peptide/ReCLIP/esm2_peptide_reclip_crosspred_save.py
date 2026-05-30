#!/usr/bin/env python3
"""
Train once on Set!=Test and evaluate once on Set==Test for peptide ReCLIP.

This uses the mutation ReCLIP feature layout:
  - HuggingFace ESM2 layer 32 per-head context features
  - per-head self-attention top-5 residues around the pseudo position
  - MINT partner residue representations weighted by MINT cross-chain attention
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
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
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "scripts" / "peptide" / "figure_plot"

for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import peptide_reclip_support as base
import peptide_reclip_save_support as old_save


def _load_mutation_reclip_common():
    common_path = MUTATION_RECLIP_DIR / "common.py"
    spec = importlib.util.spec_from_file_location("mutation_reclip_common", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load mutation ReCLIP common module from {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


new_common = _load_mutation_reclip_common()
new_common.WORK_DIR = SCRIPT_DIR

# Make peptide helpers robust when invoked outside the repository root.
base.MINT_CFG_PATH = str(PROJECT_ROOT / "mint" / "data" / "esm2_t33_650M_UR50D.json")
base.MINT_CHECKPOINT_PATH = str(PROJECT_ROOT / "mint" / "mint.ckpt")

RANDOM_SEED = 42
ESM_LAYER = 32
MINT_ATTN_LAYER = -1
PARTNER_MINT_LAYER = 32
K_SEQ = 5
TARGET_FEATURE_SPACE = "context"
USE_ATTENTION_LAYERNORM = True
TARGET_AGGREGATION = "flatten"
METHOD_NAME = "ReCLIP"

METRIC_ORDER = old_save.METRIC_ORDER


def _dataset_tag(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run peptide ReCLIP and save Set==Test predictions. "
            "Defaults match the validated mutation ReCLIP setting: L32, all heads, "
            "top5 per head, context+LN, and MINT partner representations."
        )
    )
    parser.add_argument("--data-set", required=True, help="CSV with Epitope/Sequence/Hit/Set columns.")
    parser.add_argument("--classifier", choices=["xgb", "mlp"], default="xgb", help="Classifier head.")
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Feature cache root; default is dataset-specific under peptide_result_reclip.",
    )
    parser.add_argument("--force-rebuild", action="store_true", help="Recompute all features and overwrite cache.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for MLP.")
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
    parser.add_argument("--out-csv", default=None, help="Prediction CSV path.")
    parser.add_argument("--metric-txt", default=None, help="Metric TXT path.")
    parser.add_argument("--metric-json", default=None, help="Optional metadata JSON path.")
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


def _feature_cache_path(cache_root: Path, layer_tag: str, mint_attn_tag: str, partner_mint_tag: str, top_k: int) -> Path:
    setting_tag = (
        f"layer_{layer_tag}_hall_{TARGET_AGGREGATION}_{TARGET_FEATURE_SPACE}_ln_k{top_k}_"
        f"mintattn_{mint_attn_tag}_partner_{partner_mint_tag}_partnerL1v2"
    )
    return cache_root / setting_tag / "Feature_cache" / "all_samples.npz"


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

    iterator = tqdm(df.itertuples(index=False), total=len(df), desc="Build peptide ReCLIP features")
    for row_idx, row in enumerate(iterator):
        target_seq_full = getattr(row, base.MUT_SEQ_COL)
        partner_seq_full = getattr(row, base.INTERACTOR_SEQ_COL)
        target_pos_orig = int(getattr(row, base.POSITION_COL))

        target_seq_trim, partner_seq_trim, target_pos_trim = base.truncate_pair_for_models(
            target_seq_full, partner_seq_full, target_pos_orig
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
    df: pd.DataFrame,
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

    expected_keys = [base._make_sample_key(row) for row in df.itertuples(index=False)]
    cached = _load_feature_matrix_cache(cache_path)
    if cached is not None:
        cached_features, cached_keys = cached
        if (
            cached_features.ndim == 2
            and cached_features.shape == (len(df), feature_dim)
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
        df=df,
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


def _features_and_optional_labels(
    df: pd.DataFrame,
    features: np.ndarray,
    key_to_idx: Dict[str, int],
    require_labels: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    return old_save._features_and_optional_labels(df, features, key_to_idx, require_labels=require_labels)


def main() -> None:
    args = parse_args()
    tag = _dataset_tag(args.data_set)
    clf_tag = args.classifier.upper()

    if args.device != "auto":
        base.DEVICE = args.device
        new_common.DEVICE = args.device

    DEFAULT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv else DEFAULT_FIGURE_DIR / f"{METHOD_NAME}_{tag}_{clf_tag}_for_figure.csv"
    metric_txt = (
        Path(args.metric_txt) if args.metric_txt else DEFAULT_FIGURE_DIR / f"{METHOD_NAME}_{tag}_{clf_tag}_metrics.txt"
    )
    metric_json = (
        Path(args.metric_json)
        if args.metric_json
        else DEFAULT_FIGURE_DIR / f"{METHOD_NAME}_{tag}_{clf_tag}_metadata.json"
    )
    cache_root = Path(args.cache_root) if args.cache_root else PROJECT_ROOT / "peptide_result_reclip" / tag

    print("Peptide ReCLIP (single train/test)")
    print("=" * 80)
    print(f"Dataset: {args.data_set}")
    print(f"Cache root: {cache_root}")
    print(f"Model device: {base.DEVICE}")

    df_all = old_save._load_dataset_for_prediction(args.data_set)
    if df_all.empty:
        raise RuntimeError("Dataset is empty.")
    train_df_full, test_df_full = old_save._split_train_test_by_set(df_all)

    hf_tok, hf_model, select_layer_idx, layer_tag = base.load_hf_esm2(args.hf_layer)
    mint_model, mint_alphabet, mint_layers = base.load_mint_model()
    mint_attn_layer_idx = base._norm_single_layer(args.mint_attn_layer, mint_layers)
    partner_mint_layer_idx = base._norm_single_layer(args.partner_mint_layer, mint_layers)
    mint_attn_tag = f"M{mint_attn_layer_idx}"
    partner_mint_tag = f"M{partner_mint_layer_idx}"

    allowed_tokens = old_save._supported_residue_tokens(mint_alphabet)
    train_df, train_excluded = old_save._partition_supported_rows(train_df_full, allowed_tokens)
    test_df, test_excluded = old_save._partition_supported_rows(test_df_full, allowed_tokens)

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
    features, keys, cache_path, feature_dims = prepare_feature_cache(
        df=df_all_supported,
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
        cache_root=cache_root,
        force_rebuild=args.force_rebuild,
    )
    key_to_idx = {k: i for i, k in enumerate(keys)}

    if base.DEVICE == "cuda":
        del hf_model
        del mint_model
        torch.cuda.empty_cache()

    X_train, y_train = _features_and_optional_labels(train_df, features, key_to_idx, require_labels=True)
    X_test, y_test = _features_and_optional_labels(test_df, features, key_to_idx, require_labels=False)
    del features
    del key_to_idx
    _release_gpu_memory()

    if args.classifier == "xgb":
        proba = old_save._train_xgb_and_predict(X_train, y_train, X_test, xgb_device=args.xgb_device)
    else:
        proba = old_save._train_mlp_and_predict(X_train, y_train, X_test, args)

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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metric_txt.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    old_save._save_metrics(str(metric_txt), metrics_map, has_labeled_test, notes=metric_notes)

    meta = {
        "task": "peptide",
        "method": "ReCLIP",
        "classifier": args.classifier,
        "dataset": str(Path(args.data_set)),
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
        "cache_path": str(cache_path),
        "out_csv": str(out_csv),
        "metric_txt": str(metric_txt),
        "metrics": metrics_map,
        "notes": metric_notes,
    }
    metric_json.parent.mkdir(parents=True, exist_ok=True)
    metric_json.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Prediction CSV saved: {out_csv}")
    print(f"Metrics TXT saved: {metric_txt}")
    print(f"Metadata JSON saved: {metric_json}")
    print(f"Feature cache file: {cache_path}")
    for key in METRIC_ORDER:
        val = metrics_map.get(key, float("nan"))
        if isinstance(val, (int, float, np.floating)):
            print(f"{key}: {float(val):.6f}")
        else:
            print(f"{key}: {val}")


if __name__ == "__main__":
    main()
