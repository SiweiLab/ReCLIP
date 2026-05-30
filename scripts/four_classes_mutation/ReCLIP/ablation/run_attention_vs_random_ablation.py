#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


ABLATION_DIR = Path(__file__).resolve().parent
WORK_DIR = ABLATION_DIR.parent
if str(WORK_DIR) not in sys.path:
    sys.path.insert(0, str(WORK_DIR))

import common as common_module
from common import (
    INTERACTOR_ID_COL,
    K_SEQ,
    LABEL_COL,
    MUT_SEQ_COL,
    NUM_FOLDS,
    POSITION_COL,
    RANDOM_SEED,
    TARGET_ID_COL,
    XGBClassifier,
    _make_sample_key,
    _norm_single_layer,
    _release_gpu_memory,
    aggregate_metrics,
    compute_13_metrics,
    create_tenfold_splits,
    ensure_output_layout,
    load_dataset,
    load_hf_esm2,
    load_mint_model,
    load_or_compute_attention_row,
    load_or_compute_mint_partner_representations_and_weights,
    load_or_compute_target_residue_features,
    truncate_pair_for_models,
)


ESM_LAYER = 32
MINT_ATTN_LAYER = -1
PARTNER_MINT_LAYER = 32
USE_ATTENTION_LAYERNORM = True
TARGET_FEATURE_SPACE = "context"
TARGET_AGGREGATION = "flatten"
KEY_COLS = [TARGET_ID_COL, POSITION_COL, "Before_AA", "After_AA", INTERACTOR_ID_COL]
METRIC_ORDER = [
    "Accuracy",
    "Precision_macro",
    "Precision_weighted",
    "Recall_macro",
    "Recall_weighted",
    "F1_macro",
    "F1_weighted",
    "Balanced_Acc",
    "ROC_AUC_macro_ovr",
    "PR_AUC_macro_ovr",
    "Youdens_J_macro_ovr",
    "t_test_t_mean_ovr",
    "p_value_fisher_ovr",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run attention-vs-random ablation for the fixed mutation ReCLIP setup. "
            "Target features stay at context+LN+flatten; partner features stay MINT-weighted MINT-L32."
        )
    )
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def ablation_cache_root() -> Path:
    root = ABLATION_DIR / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def ablation_results_root() -> Path:
    root = ABLATION_DIR / "results"
    root.mkdir(parents=True, exist_ok=True)
    return root


def artifact_prefix() -> str:
    return (
        f"mutation_reclip_ablation_mintpartnerL{PARTNER_MINT_LAYER}_"
        f"L{ESM_LAYER}_hall_{TARGET_AGGREGATION}_{TARGET_FEATURE_SPACE}_ln_k{K_SEQ}"
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


def load_dual_feature_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]] | None:
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


def save_dual_feature_cache(
    path: Path, X_attention: np.ndarray, X_random: np.ndarray, keys: List[str]
) -> None:
    _atomic_save_npz(
        path,
        attention_features=np.asarray(X_attention, dtype=np.float32),
        random_features=np.asarray(X_random, dtype=np.float32),
        keys=np.asarray(keys, dtype=object),
    )


def flatten_selected_head_features(
    context_aa: np.ndarray,
    selected_heads: Sequence[int],
    selected_indices_by_head: Dict[int, np.ndarray],
    k_seq: int,
) -> np.ndarray:
    head_dim = int(context_aa.shape[-1])
    slots = np.zeros((len(selected_heads), int(k_seq), head_dim), dtype=np.float32)
    for slot_idx, head_idx in enumerate(selected_heads):
        residue_indices = selected_indices_by_head[int(head_idx)]
        take = min(int(k_seq), int(residue_indices.size))
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


def build_dual_feature_matrices(
    df_all: pd.DataFrame,
    hf_tok,
    hf_model,
    select_layer_idx: int,
    mint_model,
    mint_alphabet,
    mint_attn_layer_idx: int,
    partner_mint_layer_idx: int,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object], Path]:
    num_heads = int(hf_model.encoder.layer[select_layer_idx].attention.self.num_attention_heads)
    head_dim = int(hf_model.encoder.layer[select_layer_idx].attention.self.attention_head_size)
    selected_heads = list(range(num_heads))
    partner_dim = int(getattr(mint_model, "embed_dim", hf_model.config.hidden_size))
    target_dim = int(len(selected_heads) * K_SEQ * head_dim)
    feature_dim = int(target_dim + partner_dim)

    cache_path = (
        ablation_cache_root()
        / "feature_matrices"
        / f"{artifact_prefix()}_partnerL1v2"
        / "all_samples_dual.npz"
    )
    if force_rebuild and cache_path.exists():
        cache_path.unlink()

    expected_keys = [_make_sample_key(row) for row in df_all.itertuples(index=False)]
    cached = load_dual_feature_cache(cache_path)
    if cached is not None:
        X_attention, X_random, cached_keys = cached
        if (
            X_attention.shape == (len(df_all), feature_dim)
            and X_random.shape == (len(df_all), feature_dim)
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

    X_attention = np.zeros((len(df_all), feature_dim), dtype=np.float32)
    X_random = np.zeros((len(df_all), feature_dim), dtype=np.float32)

    iterator = tqdm(df_all.itertuples(index=False), total=len(df_all), desc="Build dual features (hall)")
    for row_idx, row in enumerate(iterator):
        sample_key = _make_sample_key(row)
        target_seq_full = getattr(row, MUT_SEQ_COL)
        partner_seq_full = getattr(row, "Interactor_Seq")
        target_pos_orig = int(getattr(row, POSITION_COL))
        target_seq_trim, partner_seq_trim, target_pos_trim = truncate_pair_for_models(
            target_seq_full, partner_seq_full, target_pos_orig
        )

        context_aa = load_or_compute_target_residue_features(
            target_seq_trim,
            select_layer_idx,
            hf_tok,
            hf_model,
            feature_space=TARGET_FEATURE_SPACE,
            use_attention_layernorm=USE_ATTENTION_LAYERNORM,
        )
        _release_gpu_memory()

        attn_row = load_or_compute_attention_row(
            target_seq_trim,
            target_pos_trim,
            select_layer_idx,
            hf_tok,
            hf_model,
            use_attention_layernorm=USE_ATTENTION_LAYERNORM,
        )
        _release_gpu_memory()

        partner_repr, mint_weights = load_or_compute_mint_partner_representations_and_weights(
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
            actual_k = min(int(K_SEQ), seq_len)
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

        target_attention = flatten_selected_head_features(context_aa, selected_heads, attn_indices, K_SEQ)
        target_random = flatten_selected_head_features(context_aa, selected_heads, rand_indices, K_SEQ)
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


def train_xgb_and_predict_proba(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    if XGBClassifier is None:
        raise ImportError("xgboost is required for this script.")

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    full_num_classes = 4
    train_classes = np.asarray(sorted(set(int(v) for v in y_train.tolist())), dtype=np.int64)
    if train_classes.size == 0:
        raise ValueError("Training labels are empty.")
    if train_classes.size == 1:
        only_class = int(train_classes[0])
        out = np.zeros((X_test.shape[0], full_num_classes), dtype=np.float32)
        out[:, only_class] = 1.0
        return out

    class_to_local = {int(cls): idx for idx, cls in enumerate(train_classes.tolist())}
    y_train_local = np.asarray([class_to_local[int(v)] for v in y_train.tolist()], dtype=np.int64)

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
        num_class=int(train_classes.size),
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        enable_categorical=False,
    )
    if common_module.DEVICE == "cuda":
        tree_method = "gpu_hist"
        predictor = "gpu_predictor"
        extra_params = {"gpu_id": 0}
    else:
        tree_method = "hist"
        predictor = "auto"
        extra_params = {}

    clf = XGBClassifier(
        **base_params,
        tree_method=tree_method,
        predictor=predictor,
        **extra_params,
    )
    clf.fit(X_train, y_train_local, verbose=False)
    proba_local = np.asarray(clf.predict_proba(X_test), dtype=np.float32)

    proba = np.zeros((X_test.shape[0], full_num_classes), dtype=np.float32)
    for local_idx, original_cls in enumerate(train_classes.tolist()):
        proba[:, int(original_cls)] = proba_local[:, local_idx]

    del clf
    _release_gpu_memory()
    return proba


def save_oof_csv(
    path: Path,
    df_all: pd.DataFrame,
    fold_ids: np.ndarray,
    proba_all: np.ndarray,
    pred_all: np.ndarray,
) -> None:
    score_cols = [f"score_{i}" for i in range(proba_all.shape[1])]
    out_df = df_all[KEY_COLS + [LABEL_COL]].copy()
    out_df["sample_key"] = [_make_sample_key(row) for row in df_all.itertuples(index=False)]
    out_df["fold"] = fold_ids.astype(int)
    for i, score_col in enumerate(score_cols):
        out_df[score_col] = proba_all[:, i]
    out_df["predict"] = pred_all.astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(path, index=False)


def write_summary_txt(
    path: Path,
    dims: Dict[str, object],
    attn_summary: Dict[str, float],
    rand_summary: Dict[str, float],
    delta_summary: Dict[str, float],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("Mutation ReCLIP ablation (hall)\n")
        f.write(f"Selected heads: {dims['selected_heads']}\n")
        f.write(f"Target feature space: {TARGET_FEATURE_SPACE}\n")
        f.write(f"Use attention layernorm: {USE_ATTENTION_LAYERNORM}\n")
        f.write(f"Target aggregation: {TARGET_AGGREGATION}\n")
        f.write(f"Top-k per head: {K_SEQ}\n\n")

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
    ensure_output_layout()
    ablation_cache_root()
    ablation_results_root()

    if args.device != "auto":
        common_module.DEVICE = args.device

    prefix = args.output_prefix or artifact_prefix()
    result_root = ablation_results_root()
    json_path = result_root / f"{prefix}.json"
    txt_path = result_root / f"{prefix}.txt"
    attn_oof_path = result_root / f"{prefix}_attention_oof.csv"
    rand_oof_path = result_root / f"{prefix}_random_oof.csv"

    df_all = load_dataset()
    fold_data = create_tenfold_splits(df_all)

    hf_tok, hf_model, select_layer_idx, layer_tag = load_hf_esm2(ESM_LAYER)
    mint_model, mint_alphabet, mint_layers = load_mint_model()
    mint_attn_layer_idx = _norm_single_layer(MINT_ATTN_LAYER, mint_layers)
    partner_mint_layer_idx = _norm_single_layer(PARTNER_MINT_LAYER, mint_layers)

    X_attention, X_random, sample_keys, dims, dual_cache_path = build_dual_feature_matrices(
        df_all=df_all,
        hf_tok=hf_tok,
        hf_model=hf_model,
        select_layer_idx=select_layer_idx,
        mint_model=mint_model,
        mint_alphabet=mint_alphabet,
        mint_attn_layer_idx=mint_attn_layer_idx,
        partner_mint_layer_idx=partner_mint_layer_idx,
        force_rebuild=args.force_rebuild,
    )

    if common_module.DEVICE == "cuda":
        del hf_model
        del mint_model
        _release_gpu_memory()

    y_all = df_all[LABEL_COL].to_numpy().astype(np.int64)
    attn_proba_all = np.zeros((len(df_all), 4), dtype=np.float32)
    rand_proba_all = np.zeros((len(df_all), 4), dtype=np.float32)
    attn_pred_all = np.zeros((len(df_all),), dtype=np.int64)
    rand_pred_all = np.zeros((len(df_all),), dtype=np.int64)
    fold_ids = np.full((len(df_all),), -1, dtype=np.int64)

    attn_fold_metrics: Dict[int, Dict[str, float]] = {}
    rand_fold_metrics: Dict[int, Dict[str, float]] = {}
    attn_metrics_all: List[Dict[str, float]] = []
    rand_metrics_all: List[Dict[str, float]] = []

    for fold_idx in range(NUM_FOLDS):
        train_idx, test_idx = fold_data[fold_idx]
        print(f"\n=== Fold {fold_idx} / {NUM_FOLDS - 1} (hall) ===")
        X_train_attn = X_attention[train_idx]
        X_test_attn = X_attention[test_idx]
        X_train_rand = X_random[train_idx]
        X_test_rand = X_random[test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        attn_proba = train_xgb_and_predict_proba(X_train_attn, y_train, X_test_attn)
        rand_proba = train_xgb_and_predict_proba(X_train_rand, y_train, X_test_rand)
        attn_pred = np.argmax(attn_proba, axis=1).astype(np.int64)
        rand_pred = np.argmax(rand_proba, axis=1).astype(np.int64)

        attn_proba_all[test_idx] = attn_proba
        rand_proba_all[test_idx] = rand_proba
        attn_pred_all[test_idx] = attn_pred
        rand_pred_all[test_idx] = rand_pred
        fold_ids[test_idx] = int(fold_idx)

        attn_metrics = compute_13_metrics(y_test.astype(np.int64), attn_pred, attn_proba, num_classes=4)
        rand_metrics = compute_13_metrics(y_test.astype(np.int64), rand_pred, rand_proba, num_classes=4)
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

    attn_summary = aggregate_metrics(attn_metrics_all)
    rand_summary = aggregate_metrics(rand_metrics_all)
    delta_summary = {}
    for key in METRIC_ORDER:
        mean_key = f"{key}_mean"
        if mean_key in attn_summary and mean_key in rand_summary:
            delta_summary[f"{key}_mean_random_minus_attention"] = float(
                rand_summary[mean_key] - attn_summary[mean_key]
            )

    save_oof_csv(attn_oof_path, df_all, fold_ids, attn_proba_all, attn_pred_all)
    save_oof_csv(rand_oof_path, df_all, fold_ids, rand_proba_all, rand_pred_all)
    write_summary_txt(txt_path, dims, attn_summary, rand_summary, delta_summary)

    meta = {
        "task": "four_classes_mutation",
        "method_family": "ReCLIP_ablation",
        "head_mode": "hall",
        "esm_layer": ESM_LAYER,
        "select_layer_idx": int(select_layer_idx),
        "layer_tag": str(layer_tag),
        "mint_attn_layer_idx": int(mint_attn_layer_idx),
        "partner_mint_layer_idx": int(partner_mint_layer_idx),
        "target_feature_space": TARGET_FEATURE_SPACE,
        "use_attention_layernorm": USE_ATTENTION_LAYERNORM,
        "target_aggregation": TARGET_AGGREGATION,
        "top_k": K_SEQ,
        "dims": dims,
        "dual_feature_cache_path": str(dual_cache_path),
        "attention_oof_csv": str(attn_oof_path),
        "random_oof_csv": str(rand_oof_path),
        "attention_summary": attn_summary,
        "random_summary": rand_summary,
        "summary_delta_random_minus_attention": delta_summary,
    }
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nSaved summary txt: {txt_path}")
    print(f"Saved metadata json: {json_path}")


if __name__ == "__main__":
    main()
