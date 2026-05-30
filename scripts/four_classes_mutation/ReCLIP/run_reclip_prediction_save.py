#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

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
    WORK_DIR,
    XGBClassifier,
    _make_sample_key,
    _norm_single_layer,
    _release_gpu_memory,
    aggregate_metrics,
    compute_13_metrics,
    create_tenfold_splits,
    ensure_output_layout,
    load_dataset,
    load_feature_matrix_cache,
    load_hf_esm2,
    load_mint_model,
    load_or_compute_attention_row,
    load_or_compute_mint_partner_representations_and_weights,
    load_or_compute_target_residue_features,
    mutation_cache_root,
    save_feature_matrix_cache,
    truncate_pair_for_models,
)


ESM_LAYER = 32
MINT_ATTN_LAYER = -1
PARTNER_MINT_LAYER = 32
USE_ATTENTION_LAYERNORM = True
TARGET_FEATURE_SPACE = "context"
TARGET_AGGREGATION = "flatten"
SCORE_PREFIX = "Cross_Attention_baseline_Y_score_"
PREDICT_COL = "Cross_Attention_baseline_Y_predict"
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
            "Run the fixed four-class mutation ReCLIP pipeline: "
            "L32, all heads, top5 per head, context+LN, flatten target features, "
            "and MINT partner representations weighted by MINT cross-attention."
        )
    )
    parser.add_argument(
        "--classifier",
        choices=["xgb"],
        default="xgb",
        help="Classifier head. Kept fixed to XGBoost for the validated setting.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the full sample-level feature matrix cache even if it already exists.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(WORK_DIR / "results" / "reclip_mutation_predictions.csv"),
        help="Grouped prediction CSV output path.",
    )
    parser.add_argument(
        "--reference-csv",
        default=None,
        help="Optional reference CSV used to preserve an external row layout.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional custom prefix for result/cache artifact names.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(WORK_DIR / "results"),
        help="Directory for txt/json/oof result artifacts.",
    )
    parser.add_argument(
        "--hf-layer",
        type=int,
        default=ESM_LAYER,
        help="HuggingFace ESM2 layer index for target-side per-head features and self-attention.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=K_SEQ,
        help="Top residues selected per attention head.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Override compute device for model loading and XGBoost training.",
    )
    return parser.parse_args()


def artifact_prefix(hf_layer: int, top_k: int) -> str:
    return (
        f"mutation_reclip_mintpartnerL{PARTNER_MINT_LAYER}_"
        f"L{hf_layer}_hall_{TARGET_AGGREGATION}_{TARGET_FEATURE_SPACE}_ln_k{top_k}"
    )


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


def build_feature_matrix(
    df_all: pd.DataFrame,
    hf_tok,
    hf_model,
    select_layer_idx: int,
    mint_model,
    mint_alphabet,
    mint_attn_layer_idx: int,
    partner_mint_layer_idx: int,
    top_k: int,
    prefix: str,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, List[str], Path, Dict[str, int]]:
    num_heads = int(hf_model.encoder.layer[select_layer_idx].attention.self.num_attention_heads)
    head_dim = int(hf_model.encoder.layer[select_layer_idx].attention.self.attention_head_size)
    partner_dim = int(getattr(mint_model, "embed_dim", hf_model.config.hidden_size))
    target_dim = int(num_heads * top_k * head_dim)
    feature_dim = int(target_dim + partner_dim)

    cache_path = mutation_cache_root() / "feature_matrices" / f"{prefix}_partnerL1v2" / "all_samples.npz"
    if force_rebuild and cache_path.exists():
        cache_path.unlink()

    expected_keys = [_make_sample_key(row) for row in df_all.itertuples(index=False)]
    cached = load_feature_matrix_cache(cache_path)
    if cached is not None:
        cached_features, cached_keys = cached
        if (
            cached_features.ndim == 2
            and cached_features.shape == (len(df_all), feature_dim)
            and cached_keys == expected_keys
        ):
            print(f"Feature matrix cache hit: {cache_path}")
            return cached_features, cached_keys, cache_path, {
                "num_heads": num_heads,
                "head_dim": head_dim,
                "target_feature_dim": target_dim,
                "partner_feature_dim": partner_dim,
                "feature_dim": feature_dim,
            }
        print("Feature matrix cache mismatch detected; rebuilding.")

    features = np.zeros((len(df_all), feature_dim), dtype=np.float32)
    iterator = tqdm(df_all.itertuples(index=False), total=len(df_all), desc="Build ReCLIP features")
    for row_idx, row in enumerate(iterator):
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

        target_feat = build_attention_target_feature(context_aa, attn_row, top_k=top_k)
        partner_feat = aggregate_partner_feature(partner_repr, mint_weights, partner_dim)
        features[row_idx] = np.concatenate([target_feat, partner_feat], axis=0)

    save_feature_matrix_cache(cache_path, features, expected_keys)
    print(f"Feature matrix saved: {cache_path}")
    return features, expected_keys, cache_path, {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "target_feature_dim": target_dim,
        "partner_feature_dim": partner_dim,
        "feature_dim": feature_dim,
    }


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


def write_result_file(path: Path, fold_metrics: Dict[int, Dict[str, float]], summary: Dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for fold_idx in range(NUM_FOLDS):
            f.write(f"=== Fold {fold_idx} ===\n")
            metrics_fold = fold_metrics[fold_idx]
            for key in METRIC_ORDER:
                val = metrics_fold.get(key, float("nan"))
                if isinstance(val, (int, float, np.floating)):
                    f.write(f"{key}: {float(val):.6f}\n")
                else:
                    f.write(f"{key}: {val}\n")
            f.write("\n")
        f.write("=== Aggregated ===\n")
        for key in METRIC_ORDER:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in summary:
                f.write(f"{mean_key}: {float(summary[mean_key]):.6f}\n")
            if std_key in summary:
                f.write(f"{std_key}: {float(summary[std_key]):.6f}\n")


def save_figure_outputs(
    df_all: pd.DataFrame,
    fold_ids: np.ndarray,
    proba_all: np.ndarray,
    pred_all: np.ndarray,
    reference_csv: Path | None,
    out_csv: Path,
    oof_csv: Path,
) -> None:
    score_cols = [f"{SCORE_PREFIX}{i}" for i in range(4)]
    pred_df = df_all[KEY_COLS + [LABEL_COL]].copy()
    pred_df["fold"] = fold_ids.astype(int)
    for i, score_col in enumerate(score_cols):
        pred_df[score_col] = proba_all[:, i]
    pred_df[PREDICT_COL] = pred_all.astype(int)
    pred_df.to_csv(oof_csv, index=False)

    grouped_scores = (
        pred_df.groupby(KEY_COLS, as_index=False)[score_cols]
        .mean()
        .reset_index(drop=True)
    )
    grouped_scores[PREDICT_COL] = np.argmax(grouped_scores[score_cols].to_numpy(), axis=1).astype(int)

    if reference_csv is not None and reference_csv.exists():
        ref_df = pd.read_csv(reference_csv, low_memory=False)
        for col in KEY_COLS:
            if col not in ref_df.columns:
                raise KeyError(f"Reference CSV missing required column: {col}")

        drop_cols = [col for col in score_cols + [PREDICT_COL] if col in ref_df.columns]
        if drop_cols:
            ref_df = ref_df.drop(columns=drop_cols)
        out_df = ref_df.merge(grouped_scores, on=KEY_COLS, how="left", validate="many_to_one")
    else:
        label_df = df_all[KEY_COLS + [LABEL_COL]].drop_duplicates(subset=KEY_COLS).reset_index(drop=True)
        out_df = label_df.merge(grouped_scores, on=KEY_COLS, how="left", validate="one_to_one")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"OOF predictions saved: {oof_csv}")
    print(f"Grouped predictions saved: {out_csv}")


def main() -> None:
    args = parse_args()
    ensure_output_layout()

    if args.device != "auto":
        common_module.DEVICE = args.device

    prefix = args.output_prefix or artifact_prefix(args.hf_layer, args.top_k)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_txt_path = results_dir / f"{prefix}.txt"
    result_json_path = results_dir / f"{prefix}.json"
    oof_csv_path = results_dir / f"{prefix}_oof_predictions.csv"

    df_all = load_dataset()
    fold_data = create_tenfold_splits(df_all)

    hf_tok, hf_model, select_layer_idx, layer_tag = load_hf_esm2(args.hf_layer)
    mint_model, mint_alphabet, mint_layers = load_mint_model()
    mint_attn_layer_idx = _norm_single_layer(MINT_ATTN_LAYER, mint_layers)
    partner_mint_layer_idx = _norm_single_layer(PARTNER_MINT_LAYER, mint_layers)

    features, keys, feature_cache_path, feature_dims = build_feature_matrix(
        df_all=df_all,
        hf_tok=hf_tok,
        hf_model=hf_model,
        select_layer_idx=select_layer_idx,
        mint_model=mint_model,
        mint_alphabet=mint_alphabet,
        mint_attn_layer_idx=mint_attn_layer_idx,
        partner_mint_layer_idx=partner_mint_layer_idx,
        top_k=args.top_k,
        prefix=prefix,
        force_rebuild=args.force_rebuild,
    )

    if common_module.DEVICE == "cuda":
        del hf_model
        del mint_model
        _release_gpu_memory()

    y_all = df_all[LABEL_COL].to_numpy().astype(np.int64)
    proba_all = np.zeros((len(df_all), 4), dtype=np.float32)
    pred_all = np.zeros((len(df_all),), dtype=np.int64)
    fold_ids = np.full((len(df_all),), -1, dtype=np.int64)
    fold_metrics: Dict[int, Dict[str, float]] = {}
    collected_metrics: List[Dict[str, float]] = []

    for fold_idx in range(NUM_FOLDS):
        train_idx, test_idx = fold_data[fold_idx]
        print(f"\n=== Fold {fold_idx} / {NUM_FOLDS - 1} (ReCLIP) ===")
        X_train = features[train_idx]
        y_train = y_all[train_idx]
        X_test = features[test_idx]
        y_test = y_all[test_idx]

        proba = train_xgb_and_predict_proba(X_train, y_train, X_test)
        pred = np.argmax(proba, axis=1).astype(np.int64)

        proba_all[test_idx] = proba
        pred_all[test_idx] = pred
        fold_ids[test_idx] = int(fold_idx)

        metrics_fold = compute_13_metrics(y_test.astype(np.int64), pred, proba, num_classes=4)
        fold_metrics[fold_idx] = metrics_fold
        collected_metrics.append(metrics_fold)
        for key in METRIC_ORDER:
            if key in metrics_fold:
                print(f"{key}: {metrics_fold[key]:.6f}")

    summary = aggregate_metrics(collected_metrics)
    write_result_file(result_txt_path, fold_metrics, summary)

    score_cols = [f"{SCORE_PREFIX}{i}" for i in range(4)]
    save_figure_outputs(
        df_all=df_all,
        fold_ids=fold_ids,
        proba_all=proba_all,
        pred_all=pred_all,
        reference_csv=Path(args.reference_csv) if args.reference_csv else None,
        out_csv=Path(args.out_csv),
        oof_csv=oof_csv_path,
    )

    meta = {
        "task": "four_classes_mutation",
        "method": "ReCLIP",
        "classifier": args.classifier,
        "device": common_module.DEVICE,
        "esm_layer": int(args.hf_layer),
        "select_layer_idx": int(select_layer_idx),
        "layer_tag": str(layer_tag),
        "mint_attn_layer_idx": int(mint_attn_layer_idx),
        "partner_mint_layer_idx": int(partner_mint_layer_idx),
        "k_seq": int(args.top_k),
        "target_feature_space": TARGET_FEATURE_SPACE,
        "use_attention_layernorm": bool(USE_ATTENTION_LAYERNORM),
        "target_aggregation": TARGET_AGGREGATION,
        "use_all_heads": True,
        "num_heads": int(feature_dims["num_heads"]),
        "head_dim": int(feature_dims["head_dim"]),
        "target_feature_dim": int(feature_dims["target_feature_dim"]),
        "partner_feature_dim": int(feature_dims["partner_feature_dim"]),
        "feature_dim": int(feature_dims["feature_dim"]),
        "feature_cache_path": str(feature_cache_path),
        "result_txt_path": str(result_txt_path),
        "result_oof_csv_path": str(oof_csv_path),
        "prediction_csv_path": str(Path(args.out_csv)),
        "reference_csv_path": str(Path(args.reference_csv)) if args.reference_csv else None,
        "score_columns": score_cols,
        "predict_column": PREDICT_COL,
        "summary": summary,
    }
    result_json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nResult summary saved: {result_txt_path}")
    print(f"Result metadata saved: {result_json_path}")
    print(f"Feature matrix cache: {feature_cache_path}")


if __name__ == "__main__":
    main()
