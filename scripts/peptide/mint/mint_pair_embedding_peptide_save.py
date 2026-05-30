#!/usr/bin/env python3
"""
MINT baseline for peptide task (binary classification).

Pipeline:
1) Build MINT pair embedding for (Epitope, Long_Sequence/Sequence).
2) Train XGBoost on rows with Set!=Test.
3) Predict on rows with Set starting with "Test".
4) Save figure-ready CSV and metrics TXT.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn import metrics
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:  # optional GPU acceleration for XGBoost
    import cupy as cp  # type: ignore

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover
    cp = None
    CUPY_AVAILABLE = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_FIGURE_DIR = os.path.join(PROJECT_ROOT, "scripts", "peptide", "figure_plot")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mint.mint.data import Alphabet
from mint.mint.helpers.extract import load_config
from mint.mint.model.esm2 import ESM2 as MintESM2


RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_MINT_TOKENS = 1024

MINT_CFG_PATH = os.path.join(PROJECT_ROOT, "mint", "data", "esm2_t33_650M_UR50D.json")
MINT_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "mint", "mint.ckpt")

EPITOPE_COL = "Epitope"
LONG_SEQ_CANDIDATES = ("Long_Sequence", "Sequence")
MHC_COL = "MHC"
LABEL_COL = "Hit"
SET_COL = "Set"

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
        description="Run MINT baseline for peptide task and save per-sample predictions."
    )
    parser.add_argument("--data-set", required=True, help="CSV with Epitope/Hit/Set columns.")
    parser.add_argument(
        "--cache-root",
        default=None,
        help=(
            "Feature cache root dir. "
            "Default: <PROJECT_ROOT>/peptide_result_mint_pair_embed/<dataset_tag>/mint_layer_L<layer>"
        ),
    )
    parser.add_argument("--mint-layer", type=int, default=-1, help="MINT layer index (-1 means last).")
    parser.add_argument("--force-rebuild", action="store_true", help="Recompute all cached pair embeddings.")
    parser.add_argument(
        "--classifier",
        choices=["xgb"],
        default="xgb",
        help="Classifier head. Currently only XGBoost is supported.",
    )
    parser.add_argument(
        "--xgb-device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Execution device for XGBoost. auto tries GPU then CPU fallback.",
    )
    parser.add_argument("--xgb-n-jobs", type=int, default=8, help="XGBoost n_jobs.")
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path. Default: scripts/peptide/figure_plot/MINT_<dataset>_for_figure.csv",
    )
    parser.add_argument(
        "--metric-txt",
        default=None,
        help="Metrics TXT path. Default: scripts/peptide/figure_plot/MINT_<dataset>_metrics.txt",
    )
    return parser.parse_args()


def _release_gpu_memory() -> None:
    if cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def _pick_long_seq_col(df: pd.DataFrame) -> str:
    for col in LONG_SEQ_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(f"Dataset is missing a long-sequence column: {LONG_SEQ_CANDIDATES}")


def _validate_binary_labels(s: pd.Series, name: str, allow_na: bool) -> pd.Series:
    y = pd.to_numeric(s, errors="coerce")
    if not allow_na and y.isna().any():
        raise ValueError(f"{name} contains missing/non-numeric labels.")

    uniq = set(y.dropna().astype(int).unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"{name} labels are not binary 0/1, found: {sorted(uniq)}")
    return y


def load_and_split_dataset(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading dataset: {data_path}")
    df_raw = pd.read_csv(data_path)
    long_col = _pick_long_seq_col(df_raw)

    required = [EPITOPE_COL, LABEL_COL, SET_COL, long_col]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise KeyError(f"Dataset is missing columns: {missing}")

    df = df_raw.copy()
    df[EPITOPE_COL] = df[EPITOPE_COL].astype(str).str.strip()
    df[long_col] = df[long_col].astype(str).str.strip()
    df[SET_COL] = df[SET_COL].astype(str).str.strip()
    if MHC_COL in df.columns:
        df[MHC_COL] = df[MHC_COL].astype(str).str.strip()
    else:
        df[MHC_COL] = ""

    non_empty = (df[EPITOPE_COL] != "") & (df[long_col] != "")
    dropped_empty = int((~non_empty).sum())
    if dropped_empty > 0:
        print(f"[warn] Dropping {dropped_empty} rows with empty epitope/long sequence.")
    df = df.loc[non_empty].copy()

    set_norm = df[SET_COL].str.lower()
    is_test = set_norm.str.startswith("test")
    train_df = df.loc[~is_test].copy()
    test_df = df.loc[is_test].copy()

    if train_df.empty:
        raise ValueError("Training split (Set != 'Test') is empty.")
    if test_df.empty:
        raise ValueError("Test split (Set == 'Test') is empty.")

    train_df[LABEL_COL] = _validate_binary_labels(train_df[LABEL_COL], "train labels", allow_na=False).astype(int)
    test_df[LABEL_COL] = _validate_binary_labels(test_df[LABEL_COL], "test labels", allow_na=True)

    print(f"  rows: {len(df_raw)} -> {len(train_df) + len(test_df)} | dropped: {len(df_raw) - len(train_df) - len(test_df)}")
    print(f"  long_seq_col: {long_col}")
    print(f"  Train rows (Set!=Test): {len(train_df)} | Test rows (Set*=Test): {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), long_col


def load_mint_model(mint_layer: int) -> Tuple[MintESM2, Alphabet, int]:
    if not os.path.exists(MINT_CFG_PATH):
        raise FileNotFoundError(f"MINT config not found: {MINT_CFG_PATH}")
    if not os.path.exists(MINT_CHECKPOINT_PATH):
        raise FileNotFoundError(f"MINT checkpoint not found: {MINT_CHECKPOINT_PATH}")

    cfg = load_config(MINT_CFG_PATH)
    resolved_layer = int(mint_layer)
    if resolved_layer < 0:
        resolved_layer = cfg.encoder_layers - 1
    if resolved_layer < 0 or resolved_layer >= cfg.encoder_layers:
        raise ValueError(f"mint-layer out of range: {mint_layer}, valid [0, {cfg.encoder_layers - 1}]")

    print(f"Loading MINT model: layer={resolved_layer} device={DEVICE}")
    model = MintESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        token_dropout=cfg.token_dropout,
        use_multimer=True,
    )
    try:
        checkpoint = torch.load(MINT_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MINT_CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = OrderedDict((k.replace("model.", ""), v) for k, v in checkpoint["state_dict"].items())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] MINT state dict load: missing={missing}, unexpected={unexpected}")
    model.eval().to(DEVICE)

    alphabet = Alphabet.from_architecture("ESM-1b")
    return model, alphabet, resolved_layer


def _encode_chain(seq: str, alphabet: Alphabet) -> torch.Tensor:
    seq = (seq or "").strip().replace("J", "L")
    tokens = alphabet.encode("<cls>" + seq + "<eos>")
    return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def mint_pair_embedding(
    seq_a: str,
    seq_b: str,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layer: int,
) -> np.ndarray:
    tokens_a = _encode_chain(seq_a, alphabet)
    tokens_b = _encode_chain(seq_b, alphabet)
    total_tokens = int(tokens_a.size(0) + tokens_b.size(0))
    if total_tokens > MAX_MINT_TOKENS:
        raise ValueError(
            f"MINT token budget exceeded: seq_a={tokens_a.size(0)}, seq_b={tokens_b.size(0)}, total={total_tokens}"
        )

    chain_ids_a = torch.zeros_like(tokens_a, dtype=torch.int32)
    chain_ids_b = torch.ones_like(tokens_b, dtype=torch.int32)

    tokens = torch.cat([tokens_a, tokens_b], dim=0).unsqueeze(0).to(DEVICE)
    chain_ids = torch.cat([chain_ids_a, chain_ids_b], dim=0).unsqueeze(0).to(DEVICE)

    out = mint_model(tokens, chain_ids, repr_layers=[mint_layer])
    reps = out["representations"][mint_layer][0]  # [T, D]
    tok = tokens[0]
    valid_mask = (tok != alphabet.cls_idx) & (tok != alphabet.eos_idx) & (tok != alphabet.padding_idx)

    if valid_mask.any():
        emb = reps[valid_mask].mean(dim=0)
    else:
        emb = torch.zeros((reps.size(-1),), device=reps.device)
    return emb.detach().cpu().to(torch.float32).numpy()


def _pair_key(epitope: str, long_seq: str) -> str:
    return f"{epitope}||{long_seq}"


def _resolve_cache_root(args: argparse.Namespace, dataset_tag: str, mint_layer: int) -> str:
    if args.cache_root:
        return os.path.abspath(args.cache_root)
    return os.path.join(
        PROJECT_ROOT,
        "peptide_result_mint_pair_embed",
        dataset_tag,
        f"mint_layer_L{mint_layer}",
    )


def prepare_feature_cache(
    all_df: pd.DataFrame,
    long_col: str,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layer: int,
    cache_root: str,
    force_rebuild: bool,
) -> Tuple[np.ndarray, Dict[str, int], str]:
    feature_dir = os.path.join(cache_root, "Feature_cache")
    os.makedirs(feature_dir, exist_ok=True)
    cache_path = os.path.join(feature_dir, "pair_features.npz")

    pair_df = all_df[[EPITOPE_COL, long_col]].drop_duplicates().reset_index(drop=True)
    needed_keys = [_pair_key(e, s) for e, s in zip(pair_df[EPITOPE_COL], pair_df[long_col])]

    embed_dim = int(mint_model.embed_dim)
    existing_features = np.zeros((0, embed_dim), dtype=np.float32)
    existing_keys: List[str] = []
    if os.path.exists(cache_path) and not force_rebuild:
        data = np.load(cache_path, allow_pickle=True)
        existing_features = data["features"].astype(np.float32, copy=False)
        existing_keys = [str(k) for k in data["keys"].tolist()]
        if existing_features.ndim != 2 or existing_features.shape[1] != embed_dim:
            raise ValueError(
                f"Cached feature shape mismatch at {cache_path}: "
                f"expected (?, {embed_dim}), got {existing_features.shape}"
            )
        print(f"Feature cache found: {cache_path} | existing={len(existing_keys)}")
    elif force_rebuild and os.path.exists(cache_path):
        print(f"Force rebuild enabled. Removing old cache: {cache_path}")
        os.remove(cache_path)

    existing_key_set = set(existing_keys)
    missing_pairs: List[Tuple[str, str, str]] = []
    for e, s, k in zip(pair_df[EPITOPE_COL], pair_df[long_col], needed_keys):
        if k not in existing_key_set:
            missing_pairs.append((str(e), str(s), k))

    new_keys: List[str] = []
    new_features: List[np.ndarray] = []
    if missing_pairs:
        print(f"Feature cache miss: computing {len(missing_pairs)} unique pairs ...")
        for epitope, long_seq, key in tqdm(missing_pairs, total=len(missing_pairs), desc="MINT pair embedding"):
            feat = mint_pair_embedding(epitope, long_seq, mint_model, alphabet, mint_layer)
            new_keys.append(key)
            new_features.append(feat.astype(np.float32, copy=False))
    else:
        print("Feature cache hit: all required pairs already available.")

    if new_features:
        new_feat_arr = np.vstack(new_features).astype(np.float32, copy=False)
        combined_features = (
            np.concatenate([existing_features, new_feat_arr], axis=0).astype(np.float32, copy=False)
            if existing_features.size
            else new_feat_arr
        )
        combined_keys = existing_keys + new_keys
    else:
        combined_features = existing_features
        combined_keys = existing_keys

    np.savez(cache_path, features=combined_features, keys=np.array(combined_keys, dtype=object))
    key_to_idx = {k: i for i, k in enumerate(combined_keys)}
    print(f"Feature cache shape: {combined_features.shape} | path: {cache_path}")
    return combined_features, key_to_idx, cache_path


def features_from_df(df: pd.DataFrame, long_col: str, features: np.ndarray, key_to_idx: Dict[str, int]) -> np.ndarray:
    idxs: List[int] = []
    for epitope, long_seq in zip(df[EPITOPE_COL], df[long_col]):
        key = _pair_key(str(epitope), str(long_seq))
        idx = key_to_idx.get(key)
        if idx is None:
            raise KeyError(f"Feature cache missing key: {key[:80]}...")
        idxs.append(idx)
    return np.ascontiguousarray(features[idxs], dtype=np.float32)


def _train_xgb_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    xgb_device: str,
    xgb_n_jobs: int,
) -> np.ndarray:
    if XGBClassifier is None:
        raise ImportError("xgboost is required.")

    y_train = np.asarray(y_train, dtype=np.int64)
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    ratio = neg / max(1.0, pos)

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
        scale_pos_weight=float(1.3 * ratio),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=int(xgb_n_jobs),
    )

    requested_gpu = xgb_device == "gpu" or (xgb_device == "auto" and DEVICE == "cuda")
    if requested_gpu and not CUPY_AVAILABLE:
        if xgb_device == "gpu":
            raise RuntimeError("Requested GPU XGBoost, but CuPy is unavailable.")
        print("[warn] CuPy unavailable, fallback to CPU XGBoost.")
        requested_gpu = False

    def _fit_predict(use_gpu: bool) -> np.ndarray:
        clf = None
        X_train_dev = X_train
        y_train_dev = y_train
        X_test_dev = X_test
        local = dict(params)
        if use_gpu:
            local.update(tree_method="gpu_hist", predictor="gpu_predictor")
        else:
            local.update(tree_method="hist", predictor="auto")
        try:
            clf = XGBClassifier(**local)
            if use_gpu:
                X_train_dev = cp.asarray(X_train)
                y_train_dev = cp.asarray(y_train)
                X_test_dev = cp.asarray(X_test)
            clf.fit(X_train_dev, y_train_dev, verbose=False)
            proba = clf.predict_proba(X_test_dev)[:, 1]
            if use_gpu and isinstance(proba, cp.ndarray):
                proba = cp.asnumpy(proba)
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


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(metrics.roc_auc_score(y_true, y_prob))


def compute_metrics_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    acc = float(metrics.accuracy_score(y_true, y_pred))
    precision = float(metrics.precision_score(y_true, y_pred, zero_division=0))
    recall = float(metrics.recall_score(y_true, y_pred, zero_division=0))
    f1 = float(metrics.f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(metrics.balanced_accuracy_score(y_true, y_pred))
    roc = _safe_roc_auc(y_true, y_prob)
    ap = float(metrics.average_precision_score(y_true, y_prob))
    youden_j = float("nan")
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        youden_j = float(np.max(tpr - fpr))
    t_stat = float("nan")
    p_val = float("nan")
    try:
        t_stat, p_val = stats.ttest_rel(y_true, y_pred)
        t_stat = float(t_stat)
        p_val = float(p_val)
    except Exception:
        pass

    return {
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Balanced_Accuracy": bal_acc,
        "ROC_AUC": roc,
        "PR_AUC(AP)": ap,
        "Youdens_J": youden_j,
        "t_test_t": t_stat,
        "p_value": p_val,
    }


def save_metrics(metric_txt: str, metric_map: Dict[str, float], notes: Optional[List[str]] = None) -> None:
    os.makedirs(os.path.dirname(metric_txt), exist_ok=True)
    with open(metric_txt, "w", encoding="utf-8") as f:
        if notes:
            for note in notes:
                f.write(f"Note: {note}\n")
        for key in METRIC_ORDER:
            f.write(f"{key}: {float(metric_map.get(key, float('nan'))):.6f}\n")


def main() -> None:
    args = parse_args()
    dataset_tag = _dataset_tag(args.data_set)
    out_csv = args.out_csv or os.path.join(DEFAULT_FIGURE_DIR, f"MINT_{dataset_tag}_for_figure.csv")
    metric_txt = args.metric_txt or os.path.join(DEFAULT_FIGURE_DIR, f"MINT_{dataset_tag}_metrics.txt")
    os.makedirs(DEFAULT_FIGURE_DIR, exist_ok=True)

    print("MINT baseline (peptide) - Save per-sample outputs")
    print("=" * 80)
    print(f"data_set={args.data_set}")
    print(f"classifier={args.classifier} | xgb_device={args.xgb_device}")

    train_df, test_df, long_col = load_and_split_dataset(args.data_set)
    all_df = pd.concat([train_df, test_df], ignore_index=True)

    mint_model, mint_alphabet, mint_layer = load_mint_model(args.mint_layer)
    cache_root = _resolve_cache_root(args, dataset_tag=dataset_tag, mint_layer=mint_layer)
    features, key_to_idx, cache_path = prepare_feature_cache(
        all_df=all_df,
        long_col=long_col,
        mint_model=mint_model,
        alphabet=mint_alphabet,
        mint_layer=mint_layer,
        cache_root=cache_root,
        force_rebuild=bool(args.force_rebuild),
    )

    if DEVICE == "cuda":
        print("Releasing MINT GPU memory before classifier training...")
        del mint_model
        _release_gpu_memory()

    X_train = features_from_df(train_df, long_col, features, key_to_idx)
    X_test = features_from_df(test_df, long_col, features, key_to_idx)
    y_train = train_df[LABEL_COL].astype(int).to_numpy()

    proba = _train_xgb_and_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        xgb_device=args.xgb_device,
        xgb_n_jobs=args.xgb_n_jobs,
    )
    pred = (proba >= 0.5).astype(int)

    out_df = pd.DataFrame(
        {
            EPITOPE_COL: test_df[EPITOPE_COL].astype(str).values,
            MHC_COL: test_df[MHC_COL].astype(str).values,
            "True Y": test_df[LABEL_COL].values,
            "Predicted Y": pred.astype(int),
            "Predicted Probabilities Y": proba.astype(float),
        }
    )
    out_df = out_df.sort_values(by="Predicted Probabilities Y", ascending=False).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    y_true_num = pd.to_numeric(out_df["True Y"], errors="coerce")
    labeled_mask = y_true_num.isin([0, 1])
    notes: List[str] = []
    if not labeled_mask.all():
        notes.append(f"Only {int(labeled_mask.sum())}/{len(out_df)} test rows have 0/1 labels.")

    if labeled_mask.sum() > 0:
        y_true_eval = y_true_num[labeled_mask].astype(int).to_numpy()
        y_pred_eval = out_df.loc[labeled_mask, "Predicted Y"].astype(int).to_numpy()
        y_prob_eval = out_df.loc[labeled_mask, "Predicted Probabilities Y"].astype(float).to_numpy()
        metric_map = compute_metrics_binary(y_true_eval, y_pred_eval, y_prob_eval)
    else:
        metric_map = {k: float("nan") for k in METRIC_ORDER}
        notes.append("No labeled test rows; metrics unavailable.")

    save_metrics(metric_txt, metric_map, notes=notes)

    print(f"Feature cache: {cache_path}")
    print(f"Prediction CSV saved: {out_csv}")
    print(f"Metrics TXT saved: {metric_txt}")
    for k in METRIC_ORDER:
        print(f"{k}: {float(metric_map.get(k, float('nan'))):.6f}")
    if notes:
        for note in notes:
            print(f"Note: {note}")


if __name__ == "__main__":
    main()
