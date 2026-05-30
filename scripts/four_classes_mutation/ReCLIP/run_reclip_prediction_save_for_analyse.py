#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, TextIO, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import common as common_module
import run_reclip_prediction_save as base_pipeline
from common import (
    INTERACTOR_ID_COL,
    K_SEQ,
    LABEL_COL,
    MUT_SEQ_COL,
    NUM_FOLDS,
    POSITION_COL,
    TARGET_ID_COL,
    WORK_DIR,
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


ESM_LAYER = base_pipeline.ESM_LAYER
MINT_ATTN_LAYER = base_pipeline.MINT_ATTN_LAYER
PARTNER_MINT_LAYER = base_pipeline.PARTNER_MINT_LAYER
USE_ATTENTION_LAYERNORM = base_pipeline.USE_ATTENTION_LAYERNORM
TARGET_FEATURE_SPACE = base_pipeline.TARGET_FEATURE_SPACE
TARGET_AGGREGATION = base_pipeline.TARGET_AGGREGATION
SCORE_PREFIX = base_pipeline.SCORE_PREFIX
PREDICT_COL = base_pipeline.PREDICT_COL
KEY_COLS = base_pipeline.KEY_COLS
METRIC_ORDER = base_pipeline.METRIC_ORDER

ATTENTION_TABLE_COLUMNS = [
    "sample_id",
    "Target_UPID",
    "Position",
    "Before_AA",
    "After_AA",
    "Interactor_UPID",
    "Y2H_score",
    "Target_Seq",
    "attn_type",
    "rank",
    "pos_1based",
    "aa",
    "L",
    "Layer",
    "Head",
    "delta",
    "abs_delta",
    "trim_start_full",
    "pos_full_1based",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed four-class mutation ReCLIP pipeline and "
            "export a self-attention rank table for downstream analysis."
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
        "--force-rebuild-attention-table",
        action="store_true",
        help="Rebuild the self-attention rank table even if it already exists.",
    )
    parser.add_argument(
        "--attention-rank-csv",
        default=None,
        help=(
            "Output path for the self-attention rank table. Defaults to "
            "ReCLIP/results/<prefix>_self_attention_rank_table.tsv."
        ),
    )
    parser.add_argument(
        "--out-csv",
        default=str(
            WORK_DIR.parents[0]
            / "figure_plot"
            / "data"
            / "IntAct_mutation_reclip_for_figure.csv"
        ),
        help="Figure-ready output CSV. Uses IntAct_mutation_for_figure.csv row format and Cross_Attention_baseline score columns.",
    )
    parser.add_argument(
        "--reference-csv",
        default=str(
            WORK_DIR.parents[0]
            / "figure_plot"
            / "data"
            / "IntAct_mutation_for_figure.csv"
        ),
        help="Reference CSV used to define figure rows and non-score columns.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional custom prefix for result/cache artifact names.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Override compute device for model loading and XGBoost training.",
    )
    return parser.parse_args()


def artifact_prefix() -> str:
    return base_pipeline.artifact_prefix()


def default_attention_rank_csv(prefix: str) -> Path:
    return WORK_DIR / "results" / f"{prefix}_self_attention_rank_table.tsv"


def _open_attention_table(path: Path, mode: str, compressed: bool | None = None) -> TextIO:
    if compressed is None:
        compressed = path.suffix == ".gz"
    if compressed:
        return gzip.open(path, mode, newline="", encoding="utf-8")
    return path.open(mode, newline="", encoding="utf-8")


def attention_table_has_expected_header(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with _open_attention_table(path, "rt") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, [])
    except Exception:
        return False
    return header == ATTENTION_TABLE_COLUMNS


class SelfAttentionRankTableWriter:
    def __init__(self, path: Path):
        self.path = path
        self.tmp_path = path.with_name(f"{path.name}.tmp")
        self.handle: TextIO | None = None
        self.writer: csv.writer | None = None

    def __enter__(self) -> "SelfAttentionRankTableWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = _open_attention_table(self.tmp_path, "wt", compressed=self.path.suffix == ".gz")
        self.writer = csv.writer(self.handle, delimiter="\t")
        self.writer.writerow(ATTENTION_TABLE_COLUMNS)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()
        if exc_type is None:
            os.replace(str(self.tmp_path), str(self.path))
        elif self.tmp_path.exists():
            try:
                self.tmp_path.unlink()
            except OSError:
                pass

    def write_sample(
        self,
        row_idx: int,
        row,
        target_seq_trim: str,
        target_pos_trim: int,
        target_pos_full: int,
        attn_row: np.ndarray,
        layer_idx: int,
    ) -> None:
        if self.writer is None:
            raise RuntimeError("Attention table writer is not open.")

        seq_len = min(len(target_seq_trim), int(attn_row.shape[1]))
        if seq_len <= 0:
            return

        sample_key = _make_sample_key(row)
        sample_id = f"{sample_key}||row_idx={row_idx}"
        trim_start_full = int(target_pos_full) - int(target_pos_trim) + 1
        num_heads = int(attn_row.shape[0])
        rows = []
        target_upid = getattr(row, TARGET_ID_COL)
        before_aa = getattr(row, "Before_AA")
        after_aa = getattr(row, "After_AA")
        interactor_upid = getattr(row, INTERACTOR_ID_COL)
        y2h_score = getattr(row, LABEL_COL)
        target_seq = getattr(row, "Target_Seq", "")
        mut_idx = int(target_pos_trim) - 1

        for head_idx in range(num_heads):
            scores = np.asarray(attn_row[head_idx, :seq_len], dtype=np.float32)
            order = np.argsort(-scores, kind="stable")
            rank_lookup = np.empty(seq_len, dtype=np.int32)
            rank_lookup[order] = np.arange(1, seq_len + 1, dtype=np.int32)
            selected_idx = [int(idx) for idx in order[: min(int(K_SEQ), seq_len)]]
            if 0 <= mut_idx < seq_len and mut_idx not in selected_idx:
                selected_idx.append(mut_idx)

            for pos_idx in selected_idx:
                pos_1based = int(pos_idx) + 1
                delta = pos_1based - int(target_pos_trim)
                rows.append(
                    (
                        sample_id,
                        target_upid,
                        int(target_pos_full),
                        before_aa,
                        after_aa,
                        interactor_upid,
                        y2h_score,
                        target_seq,
                        "self",
                        int(rank_lookup[int(pos_idx)]),
                        pos_1based,
                        target_seq_trim[int(pos_idx)],
                        int(seq_len),
                        int(layer_idx),
                        int(head_idx),
                        int(delta),
                        int(abs(delta)),
                        int(trim_start_full),
                        int(trim_start_full + pos_1based - 1),
                    )
                )

        self.writer.writerows(rows)


def write_self_attention_rank_table(
    df_all: pd.DataFrame,
    hf_tok,
    hf_model,
    select_layer_idx: int,
    attention_table_path: Path,
) -> None:
    print(f"Writing self-attention rank table: {attention_table_path}")
    iterator = tqdm(df_all.itertuples(index=False), total=len(df_all), desc="Write self-attention ranks")
    with SelfAttentionRankTableWriter(attention_table_path) as attention_writer:
        for row_idx, row in enumerate(iterator):
            target_seq_full = getattr(row, MUT_SEQ_COL)
            partner_seq_full = getattr(row, "Interactor_Seq")
            target_pos_orig = int(getattr(row, POSITION_COL))
            target_seq_trim, _, target_pos_trim = truncate_pair_for_models(
                target_seq_full,
                partner_seq_full,
                target_pos_orig,
            )

            attn_row = load_or_compute_attention_row(
                target_seq_trim,
                target_pos_trim,
                select_layer_idx,
                hf_tok,
                hf_model,
                use_attention_layernorm=USE_ATTENTION_LAYERNORM,
            )
            _release_gpu_memory()

            attention_writer.write_sample(
                row_idx=row_idx,
                row=row,
                target_seq_trim=target_seq_trim,
                target_pos_trim=target_pos_trim,
                target_pos_full=target_pos_orig,
                attn_row=attn_row,
                layer_idx=select_layer_idx,
            )


def build_feature_matrix(
    df_all: pd.DataFrame,
    hf_tok,
    hf_model,
    select_layer_idx: int,
    mint_model,
    mint_alphabet,
    mint_attn_layer_idx: int,
    partner_mint_layer_idx: int,
    prefix: str,
    attention_table_path: Path,
    force_rebuild: bool = False,
    force_rebuild_attention_table: bool = False,
) -> Tuple[np.ndarray, List[str], Path, Dict[str, int]]:
    num_heads = int(hf_model.encoder.layer[select_layer_idx].attention.self.num_attention_heads)
    head_dim = int(hf_model.encoder.layer[select_layer_idx].attention.self.attention_head_size)
    partner_dim = int(getattr(mint_model, "embed_dim", hf_model.config.hidden_size))
    target_dim = int(num_heads * K_SEQ * head_dim)
    feature_dim = int(target_dim + partner_dim)

    cache_path = mutation_cache_root() / "feature_matrices" / prefix / "all_samples.npz"
    if force_rebuild and cache_path.exists():
        cache_path.unlink()

    rebuild_attention_table = (
        force_rebuild
        or force_rebuild_attention_table
        or not attention_table_has_expected_header(attention_table_path)
    )
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
            if rebuild_attention_table:
                write_self_attention_rank_table(
                    df_all=df_all,
                    hf_tok=hf_tok,
                    hf_model=hf_model,
                    select_layer_idx=select_layer_idx,
                    attention_table_path=attention_table_path,
                )
            else:
                print(f"Self-attention rank table exists: {attention_table_path}")
            return cached_features, cached_keys, cache_path, {
                "num_heads": num_heads,
                "head_dim": head_dim,
                "target_feature_dim": target_dim,
                "partner_feature_dim": partner_dim,
                "feature_dim": feature_dim,
            }
        print("Feature matrix cache mismatch detected; rebuilding.")
        rebuild_attention_table = True

    features = np.zeros((len(df_all), feature_dim), dtype=np.float32)
    iterator = tqdm(df_all.itertuples(index=False), total=len(df_all), desc="Build ReCLIP features")
    writer_context = (
        SelfAttentionRankTableWriter(attention_table_path)
        if rebuild_attention_table
        else nullcontext(None)
    )

    with writer_context as attention_writer:
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

            if attention_writer is not None:
                attention_writer.write_sample(
                    row_idx=row_idx,
                    row=row,
                    target_seq_trim=target_seq_trim,
                    target_pos_trim=target_pos_trim,
                    target_pos_full=target_pos_orig,
                    attn_row=attn_row,
                    layer_idx=select_layer_idx,
                )

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

            target_feat = base_pipeline.build_attention_target_feature(context_aa, attn_row)
            partner_feat = base_pipeline.aggregate_partner_feature(partner_repr, mint_weights, partner_dim)
            features[row_idx] = np.concatenate([target_feat, partner_feat], axis=0)

    save_feature_matrix_cache(cache_path, features, expected_keys)
    print(f"Feature matrix saved: {cache_path}")
    if rebuild_attention_table:
        print(f"Self-attention rank table saved: {attention_table_path}")
    else:
        print(f"Self-attention rank table exists: {attention_table_path}")
    return features, expected_keys, cache_path, {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "target_feature_dim": target_dim,
        "partner_feature_dim": partner_dim,
        "feature_dim": feature_dim,
    }


def main() -> None:
    args = parse_args()
    ensure_output_layout()

    if args.device != "auto":
        common_module.DEVICE = args.device

    prefix = args.output_prefix or artifact_prefix()
    results_dir = WORK_DIR / "results"
    result_txt_path = results_dir / f"{prefix}.txt"
    result_json_path = results_dir / f"{prefix}.json"
    oof_csv_path = results_dir / f"{prefix}_oof_predictions.csv"
    attention_rank_csv_path = (
        Path(args.attention_rank_csv)
        if args.attention_rank_csv is not None
        else default_attention_rank_csv(prefix)
    )

    df_all = load_dataset()
    fold_data = create_tenfold_splits(df_all)

    hf_tok, hf_model, select_layer_idx, layer_tag = load_hf_esm2(ESM_LAYER)
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
        prefix=prefix,
        attention_table_path=attention_rank_csv_path,
        force_rebuild=args.force_rebuild,
        force_rebuild_attention_table=args.force_rebuild_attention_table,
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
        print(f"\n=== Fold {fold_idx} / {NUM_FOLDS - 1} (ReCLIP Analyse) ===")
        X_train = features[train_idx]
        y_train = y_all[train_idx]
        X_test = features[test_idx]
        y_test = y_all[test_idx]

        proba = base_pipeline.train_xgb_and_predict_proba(X_train, y_train, X_test)
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
    base_pipeline.write_result_file(result_txt_path, fold_metrics, summary)

    score_cols = [f"{SCORE_PREFIX}{i}" for i in range(4)]
    base_pipeline.save_figure_outputs(
        df_all=df_all,
        fold_ids=fold_ids,
        proba_all=proba_all,
        pred_all=pred_all,
        reference_csv=Path(args.reference_csv),
        out_csv=Path(args.out_csv),
        oof_csv=oof_csv_path,
    )

    meta = {
        "task": "four_classes_mutation",
        "method": "ReCLIP_for_analyse",
        "classifier": args.classifier,
        "device": common_module.DEVICE,
        "esm_layer": int(ESM_LAYER),
        "select_layer_idx": int(select_layer_idx),
        "layer_tag": str(layer_tag),
        "mint_attn_layer_idx": int(mint_attn_layer_idx),
        "partner_mint_layer_idx": int(partner_mint_layer_idx),
        "k_seq": int(K_SEQ),
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
        "attention_rank_csv_path": str(attention_rank_csv_path),
        "attention_rank_columns": ATTENTION_TABLE_COLUMNS,
        "result_txt_path": str(result_txt_path),
        "result_oof_csv_path": str(oof_csv_path),
        "figure_csv_path": str(Path(args.out_csv)),
        "reference_csv_path": str(Path(args.reference_csv)),
        "score_columns": score_cols,
        "predict_column": PREDICT_COL,
        "summary": summary,
    }
    result_json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nResult summary saved: {result_txt_path}")
    print(f"Result metadata saved: {result_json_path}")
    print(f"Feature matrix cache: {feature_cache_path}")
    print(f"Self-attention rank table: {attention_rank_csv_path}")


if __name__ == "__main__":
    main()
