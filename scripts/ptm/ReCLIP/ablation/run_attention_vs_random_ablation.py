#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ABLATION_DIR = Path(__file__).resolve().parent
NEW_DIR = ABLATION_DIR.parent
PROJECT_ROOT = NEW_DIR.parents[2]
MUTATION_RECLIP_DIR = PROJECT_ROOT / "scripts" / "four_classes_mutation" / "ReCLIP"

for path in [PROJECT_ROOT, NEW_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import ptm_reclip_support as base


def _load_mutation_reclip_common():
    common_path = MUTATION_RECLIP_DIR / "common.py"
    spec = importlib.util.spec_from_file_location("ptm_reclip_common_ablation", common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load mutation ReCLIP common module from {common_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


new_common = _load_mutation_reclip_common()
new_common.WORK_DIR = NEW_DIR

base.FOLD_DIR = str(PROJECT_ROOT / "data" / "ptm" / "ten_folds")
base.MINT_CFG_PATH = str(PROJECT_ROOT / "mint" / "data" / "esm2_t33_650M_UR50D.json")
base.MINT_CHECKPOINT_PATH = str(PROJECT_ROOT / "mint" / "mint.ckpt")

ESM_LAYER = 32
MINT_ATTN_LAYER = -1
PARTNER_MINT_LAYER = 32
TOP_K = 5
TARGET_FEATURE_SPACE = "context"
USE_ATTENTION_LAYERNORM = True
TARGET_AGGREGATION = "flatten"
H6_SELECTED_HEADS = (15, 12, 17, 1, 16, 9)
RANDOM_SEED = 42
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
            "Run attention-vs-random ablation for the PTM ReCLIP setup. "
            "Target stays at context+LN+flatten; partner stays MINT-weighted MINT-L32."
        )
    )
    parser.add_argument("--head-mode", choices=["hall", "h6"], default="hall")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--xgb-device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def _release_gpu_memory() -> None:
    base._release_gpu_memory()
    new_common._release_gpu_memory()


def ablation_cache_root() -> Path:
    root = ABLATION_DIR / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def ablation_results_root() -> Path:
    root = ABLATION_DIR / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifact_prefix(head_mode: str) -> str:
    return (
        f"ptm_reclip_ablation_mintpartnerL{PARTNER_MINT_LAYER}_"
        f"L{ESM_LAYER}_{head_mode}_{TARGET_AGGREGATION}_{TARGET_FEATURE_SPACE}_ln_k{TOP_K}"
    )


def _seed_from_key(key: str, random_seed: int) -> int:
    payload = f"{random_seed}::{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def deterministic_random_indices(length: int, top_k: int, key: str, random_seed: int) -> np.ndarray:
    if top_k <= 0 or length <= 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.default_rng(_seed_from_key(key, random_seed))
    return np.sort(rng.choice(length, size=top_k, replace=False).astype(np.int64, copy=False))


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


def load_dual_feature_cache(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=True)
        X_attention = np.asarray(data["attention_features"], dtype=np.float32)
        X_random = np.asarray(data["random_features"], dtype=np.float32)
        keys = data["keys"].tolist()
        return X_attention, X_random, keys
    except (OSError, ValueError, EOFError, KeyError):
        try:
            path.unlink()
        except OSError:
            pass
        return None


def save_dual_feature_cache(path: Path, X_attention: np.ndarray, X_random: np.ndarray, keys: List[str]) -> None:
    _atomic_save_npz(
        path,
        attention_features=np.asarray(X_attention, dtype=np.float32),
        random_features=np.asarray(X_random, dtype=np.float32),
        keys=np.asarray(keys, dtype=object),
    )


def resolve_selected_heads(head_mode: str, num_heads: int) -> List[int]:
    if head_mode == "hall":
        return list(range(num_heads))
    selected = [int(h) for h in H6_SELECTED_HEADS]
    bad = [h for h in selected if h < 0 or h >= num_heads]
    if bad:
        raise ValueError(f"Invalid selected heads {bad} for num_heads={num_heads}")
    return selected


def flatten_selected_head_features(
    context_aa: np.ndarray,
    selected_heads: Sequence[int],
    selected_indices_by_head: Dict[int, np.ndarray],
    top_k: int,
) -> np.ndarray:
    head_dim = int(context_aa.shape[-1])
    slots = np.zeros((len(selected_heads), int(top_k), head_dim), dtype=np.float32)
    for slot_idx, head_idx in enumerate(selected_heads):
        residue_indices = selected_indices_by_head[int(head_idx)]
        take = min(int(top_k), int(residue_indices.size))
        if take > 0:
            slots[slot_idx, :take, :] = context_aa[int(head_idx), residue_indices[:take], :].astype(
                np.float32, copy=False
            )
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


def _build_unique_dataframe(fold_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    df_all = base.concat_all_folds(fold_data)
    if df_all.empty:
        raise RuntimeError("No PTM samples found.")
    if base.PPI_ID_COL in df_all.columns and base.RESIDUE_ID_COL in df_all.columns:
        subset_cols = [base.PPI_ID_COL, base.RESIDUE_ID_COL]
    else:
        subset_cols = [c for c in [base.TARGET_ID_COL, base.INTERACTOR_ID_COL, base.POSITION_COL] if c in df_all.columns]
        if not subset_cols:
            subset_cols = df_all.columns.tolist()
    df_unique = df_all.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    print(f"Total fold rows: {len(df_all)} | unique rows for cache: {len(df_unique)}")
    return df_unique


def build_dual_feature_matrices(
    df_unique: pd.DataFrame,
    hf_tok,
    hf_model,
    select_layer_idx: int,
    mint_model,
    mint_alphabet,
    mint_attn_layer_idx: int,
    partner_mint_layer_idx: int,
    head_mode: str,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object], Path]:
    num_heads = int(hf_model.encoder.layer[select_layer_idx].attention.self.num_attention_heads)
    head_dim = int(hf_model.encoder.layer[select_layer_idx].attention.self.attention_head_size)
    selected_heads = resolve_selected_heads(head_mode, num_heads)
    partner_dim = int(getattr(mint_model, "embed_dim", hf_model.config.hidden_size))
    target_dim = int(len(selected_heads) * TOP_K * head_dim)
    feature_dim = int(target_dim + partner_dim)

    cache_path = (
        ablation_cache_root()
        / "feature_matrices"
        / f"{artifact_prefix(head_mode)}_partnerL1v2"
        / "all_samples_dual.npz"
    )
    if force_rebuild and cache_path.exists():
        cache_path.unlink()

    expected_keys = [base._make_sample_key(row) for row in df_unique.itertuples(index=False)]
    cached = load_dual_feature_cache(cache_path)
    if cached is not None:
        X_attention, X_random, cached_keys = cached
        if (
            X_attention.shape == (len(df_unique), feature_dim)
            and X_random.shape == (len(df_unique), feature_dim)
            and cached_keys == expected_keys
        ):
            print(f"Dual feature cache hit: {cache_path}")
            dims = {
                "num_heads_total": num_heads,
                "selected_heads": list(selected_heads),
                "num_selected_heads": len(selected_heads),
                "head_dim": head_dim,
                "target_feature_dim": target_dim,
                "partner_feature_dim": partner_dim,
                "feature_dim": feature_dim,
            }
            return X_attention, X_random, cached_keys, dims, cache_path
        print("Dual feature cache mismatch detected; rebuilding.")

    X_attention = np.zeros((len(df_unique), feature_dim), dtype=np.float32)
    X_random = np.zeros((len(df_unique), feature_dim), dtype=np.float32)

    iterator = tqdm(df_unique.itertuples(index=False), total=len(df_unique), desc=f"Build PTM dual features ({head_mode})")
    for row_idx, row in enumerate(iterator):
        sample_key = base._make_sample_key(row)
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
            mint_alphabet,
            partner_mint_layer_idx=partner_mint_layer_idx,
            mint_attn_layer_idx=mint_attn_layer_idx,
        )
        _release_gpu_memory()

        seq_len = min(int(context_aa.shape[1]), int(attn_row.shape[1]))
        attn_indices: Dict[int, np.ndarray] = {}
        rand_indices: Dict[int, np.ndarray] = {}
        for head_idx in selected_heads:
            if seq_len <= 0:
                attn_indices[head_idx] = np.zeros((0,), dtype=np.int64)
                rand_indices[head_idx] = np.zeros((0,), dtype=np.int64)
                continue
            head_scores = attn_row[head_idx, :seq_len]
            actual_k = min(int(TOP_K), seq_len)
            top_idx = np.argsort(-head_scores)[:actual_k].astype(np.int64, copy=False)
            rand_idx = deterministic_random_indices(
                seq_len,
                actual_k,
                f"{sample_key}||head{head_idx}",
                RANDOM_SEED,
            )
            if rand_idx.size > 1:
                rand_idx = rand_idx[np.argsort(-head_scores[rand_idx])]
            attn_indices[head_idx] = top_idx
            rand_indices[head_idx] = rand_idx

        target_attention = flatten_selected_head_features(context_aa, selected_heads, attn_indices, TOP_K)
        target_random = flatten_selected_head_features(context_aa, selected_heads, rand_indices, TOP_K)
        partner_feat = aggregate_partner_feature(partner_repr, mint_weights, partner_dim)

        X_attention[row_idx] = np.concatenate([target_attention, partner_feat], axis=0)
        X_random[row_idx] = np.concatenate([target_random, partner_feat], axis=0)

    save_dual_feature_cache(cache_path, X_attention, X_random, expected_keys)
    print(f"Dual feature cache saved: {cache_path}")
    dims = {
        "num_heads_total": num_heads,
        "selected_heads": list(selected_heads),
        "num_selected_heads": len(selected_heads),
        "head_dim": head_dim,
        "target_feature_dim": target_dim,
        "partner_feature_dim": partner_dim,
        "feature_dim": feature_dim,
    }
    return X_attention, X_random, expected_keys, dims, cache_path


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


def save_oof_csv(
    path: Path,
    oof_parts: List[pd.DataFrame],
) -> None:
    out_df = pd.concat(oof_parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)


def write_summary_txt(
    path: Path,
    head_mode: str,
    dims: Dict[str, object],
    attn_summary: Dict[str, float],
    rand_summary: Dict[str, float],
    delta_summary: Dict[str, float],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"PTM ReCLIP ablation ({head_mode})\n")
        f.write(f"Selected heads: {dims['selected_heads']}\n")
        f.write(f"Target feature space: {TARGET_FEATURE_SPACE}\n")
        f.write(f"Use attention layernorm: {USE_ATTENTION_LAYERNORM}\n")
        f.write(f"Target aggregation: {TARGET_AGGREGATION}\n")
        f.write(f"Top-k per head: {TOP_K}\n\n")

        f.write("[attention-topk]\n")
        for key in METRIC_ORDER:
            mean_key = f"{key}_mean"
            if mean_key in attn_summary:
                f.write(f"{mean_key}: {float(attn_summary[mean_key]):.6f}\n")
        f.write("\n[random-k]\n")
        for key in METRIC_ORDER:
            mean_key = f"{key}_mean"
            if mean_key in rand_summary:
                f.write(f"{mean_key}: {float(rand_summary[mean_key]):.6f}\n")
        f.write("\n[random - attention]\n")
        for key in METRIC_ORDER:
            delta_key = f"{key}_mean_random_minus_attention"
            if delta_key in delta_summary:
                f.write(f"{delta_key}: {float(delta_summary[delta_key]):.6f}\n")


def main() -> None:
    args = parse_args()
    if args.device != "auto":
        base.DEVICE = args.device
        new_common.DEVICE = args.device

    ablation_cache_root()
    ablation_results_root()

    prefix = args.output_prefix or artifact_prefix(args.head_mode)
    result_root = ablation_results_root()
    json_path = result_root / f"{prefix}.json"
    txt_path = result_root / f"{prefix}.txt"
    attn_oof_path = result_root / f"{prefix}_attention_oof.csv"
    rand_oof_path = result_root / f"{prefix}_random_oof.csv"

    fold_data = base.load_all_folds()
    df_unique = _build_unique_dataframe(fold_data)

    hf_tok, hf_model, select_layer_idx, layer_tag = new_common.load_hf_esm2(ESM_LAYER)
    mint_model, mint_alphabet, mint_layers = new_common.load_mint_model()
    mint_attn_layer_idx = new_common._norm_single_layer(MINT_ATTN_LAYER, mint_layers)
    partner_mint_layer_idx = new_common._norm_single_layer(PARTNER_MINT_LAYER, mint_layers)

    X_attention, X_random, keys, dims, dual_cache_path = build_dual_feature_matrices(
        df_unique=df_unique,
        hf_tok=hf_tok,
        hf_model=hf_model,
        select_layer_idx=select_layer_idx,
        mint_model=mint_model,
        mint_alphabet=mint_alphabet,
        mint_attn_layer_idx=mint_attn_layer_idx,
        partner_mint_layer_idx=partner_mint_layer_idx,
        head_mode=args.head_mode,
        force_rebuild=args.force_rebuild,
    )
    key_to_idx = {k: i for i, k in enumerate(keys)}

    if base.DEVICE == "cuda":
        del hf_model
        del mint_model
        torch.cuda.empty_cache()

    attn_fold_metrics: Dict[int, Dict[str, float]] = {}
    rand_fold_metrics: Dict[int, Dict[str, float]] = {}
    attn_metrics_all: List[Dict[str, float]] = []
    rand_metrics_all: List[Dict[str, float]] = []
    attn_oof_parts: List[pd.DataFrame] = []
    rand_oof_parts: List[pd.DataFrame] = []

    for fold_idx in base.FOLD_INDICES:
        print(f"\n=== Fold {fold_idx} / {base.FOLD_INDICES[-1]} ({args.head_mode}) ===")
        train_df, test_df = fold_data[fold_idx]
        X_train_attn, y_train = base.features_from_df(train_df, X_attention, key_to_idx)
        X_test_attn, y_test = base.features_from_df(test_df, X_attention, key_to_idx)
        X_train_rand, _ = base.features_from_df(train_df, X_random, key_to_idx)
        X_test_rand, _ = base.features_from_df(test_df, X_random, key_to_idx)

        attn_proba = _train_xgb_and_predict(X_train_attn, y_train, X_test_attn, xgb_device=args.xgb_device)
        rand_proba = _train_xgb_and_predict(X_train_rand, y_train, X_test_rand, xgb_device=args.xgb_device)
        attn_pred = (attn_proba >= 0.5).astype(np.int32)
        rand_pred = (rand_proba >= 0.5).astype(np.int32)

        attn_metrics = base.compute_ten_metrics(y_test, attn_pred, attn_proba)
        rand_metrics = base.compute_ten_metrics(y_test, rand_pred, rand_proba)
        attn_fold_metrics[fold_idx] = attn_metrics
        rand_fold_metrics[fold_idx] = rand_metrics
        attn_metrics_all.append(attn_metrics)
        rand_metrics_all.append(rand_metrics)

        print("attention-topk:")
        for key in METRIC_ORDER:
            if key in attn_metrics:
                print(f"  {key}: {attn_metrics[key]:.6f}")
        print("random-k:")
        for key in METRIC_ORDER:
            if key in rand_metrics:
                print(f"  {key}: {rand_metrics[key]:.6f}")

        fold_attn = test_df.copy().reset_index(drop=True)
        fold_attn["sample_key"] = [base._make_sample_key(row) for row in fold_attn.itertuples(index=False)]
        fold_attn["fold"] = int(fold_idx)
        fold_attn["score_0"] = 1.0 - attn_proba
        fold_attn["score_1"] = attn_proba
        fold_attn["predict"] = attn_pred.astype(np.int32)
        attn_oof_parts.append(fold_attn)

        fold_rand = test_df.copy().reset_index(drop=True)
        fold_rand["sample_key"] = [base._make_sample_key(row) for row in fold_rand.itertuples(index=False)]
        fold_rand["fold"] = int(fold_idx)
        fold_rand["score_0"] = 1.0 - rand_proba
        fold_rand["score_1"] = rand_proba
        fold_rand["predict"] = rand_pred.astype(np.int32)
        rand_oof_parts.append(fold_rand)

        del X_train_attn, X_test_attn, X_train_rand, X_test_rand, y_train, y_test, attn_proba, rand_proba
        del attn_pred, rand_pred
        _release_gpu_memory()

    attn_summary = base.aggregate_metrics(attn_metrics_all)
    rand_summary = base.aggregate_metrics(rand_metrics_all)
    delta_summary = {}
    for key in METRIC_ORDER:
        mean_key = f"{key}_mean"
        if mean_key in attn_summary and mean_key in rand_summary:
            delta_summary[f"{key}_mean_random_minus_attention"] = float(
                rand_summary[mean_key] - attn_summary[mean_key]
            )

    save_oof_csv(attn_oof_path, attn_oof_parts)
    save_oof_csv(rand_oof_path, rand_oof_parts)
    write_summary_txt(txt_path, args.head_mode, dims, attn_summary, rand_summary, delta_summary)

    meta = {
        "task": "ptm",
        "method_family": "ReCLIP_ablation",
        "head_mode": args.head_mode,
        "esm_layer": ESM_LAYER,
        "select_layer_idx": int(select_layer_idx),
        "layer_tag": str(layer_tag),
        "mint_attn_layer_idx": int(mint_attn_layer_idx),
        "partner_mint_layer_idx": int(partner_mint_layer_idx),
        "target_feature_space": TARGET_FEATURE_SPACE,
        "use_attention_layernorm": bool(USE_ATTENTION_LAYERNORM),
        "target_aggregation": TARGET_AGGREGATION,
        "top_k": TOP_K,
        "dims": dims,
        "dual_feature_cache_path": str(dual_cache_path),
        "attention_oof_csv": str(attn_oof_path),
        "random_oof_csv": str(rand_oof_path),
        "attention_fold_metrics": attn_fold_metrics,
        "random_fold_metrics": rand_fold_metrics,
        "attention_summary": attn_summary,
        "random_summary": rand_summary,
        "summary_delta_random_minus_attention": delta_summary,
    }
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSaved summary txt: {txt_path}")
    print(f"Saved metadata json: {json_path}")


if __name__ == "__main__":
    main()
