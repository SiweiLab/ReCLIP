#!/usr/bin/env python3
"""
MINT Pair-Embedding Mutation Interaction Prediction.

Pipeline:
 1) Use MINT multimer to compute a global pair embedding (mean over all valid
    tokens) for (mutated_seq, interactor_seq).
 2) Compute another pair embedding for (target_seq, interactor_seq).
 3) Concatenate the two embeddings and train/evaluate a classifier head
    (XGBoost or MLP) with stratified ten-fold CV.

No attention weights or residue-level selection is used in this ablation; only
global pair embeddings. Features are cached to disk to avoid recomputation.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:  # optional GPU acceleration for XGBoost
    import cupy as cp  # type: ignore

    CUPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    cp = None
    CUPY_AVAILABLE = False

# Ensure project root (contains `mint` package) is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mint.mint.data import Alphabet
from mint.mint.helpers.extract import load_config
from mint.mint.model.esm2 import ESM2 as MintESM2

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_MINT_TOKENS = 1024
MAX_MUT_RESIDUES = (MAX_MINT_TOKENS // 2) - 2  # reserve tokens & balance partner window
MAX_TOTAL_RESIDUES = MAX_MINT_TOKENS - 4

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "four_classes_mutation", "Mutation_IMEx_IntAct_clean.tsv")
NUM_FOLDS = 10

MINT_CFG_PATH = os.path.join(PROJECT_ROOT, "mint", "data", "esm2_t33_650M_UR50D.json")
MINT_CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "mint", "mint.ckpt")

TARGET_ID_COL = "Target_UPID"
TARGET_SEQ_COL = "Target_Seq"
MUT_SEQ_COL_RAW = "Mutated_Seq (unless WT)"
MUT_SEQ_COL_ALT = "Mutated_Seq"
MUT_SEQ_COL = "Mutated_Seq_unless_WT"
INTERACTOR_ID_COL = "Interactor_UPID"
INTERACTOR_SEQ_COL = "Interactor_Seq"
POSITION_COL = "Position"
LABEL_COL = "Y2H_score"
MUTATION_COL = "Mutation"
REQ_COLS = [POSITION_COL, TARGET_SEQ_COL, MUT_SEQ_COL, INTERACTOR_SEQ_COL]

CACHE_ROOT_PREFIX = os.path.join(PROJECT_ROOT, "mutation_result_mint_pair_embed")
FEATURE_CACHE_SUBDIR = "Feature_cache"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "Results")
RESULT_BASENAME = "result_mint_pair_embedding_{clf}_tenfold.txt"

# ---------------------------------------------------------------------------
# Argument parsing / reproducibility helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MINT pair-embedding mutation interaction prediction (ablation)."
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
    return parser.parse_args()


def _release_gpu_memory():
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

# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def _prep_for_windows(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.dropna(subset=REQ_COLS).copy()
    df2[POSITION_COL] = df2[POSITION_COL].astype(int)
    return df2


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    print(f"Loading dataset: {DATASET_PATH}")
    df_raw = pd.read_csv(DATASET_PATH, sep="\t")
    if MUT_SEQ_COL_RAW in df_raw.columns:
        df_raw = df_raw.rename(columns={MUT_SEQ_COL_RAW: MUT_SEQ_COL})
    elif MUT_SEQ_COL_ALT in df_raw.columns:
        df_raw = df_raw.rename(columns={MUT_SEQ_COL_ALT: MUT_SEQ_COL})
    df_clean = _prep_for_windows(df_raw)
    print(f"  rows: {len(df_raw)} -> {len(df_clean)} | dropped: {len(df_raw) - len(df_clean)}")
    return df_clean.reset_index(drop=True)


def create_tenfold_splits(df: pd.DataFrame) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    fold_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[LABEL_COL])):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        fold_data[fold_idx] = (df_train, df_test)
        print(f"Fold {fold_idx}: train {len(df_train)} / test {len(df_test)}")
    return fold_data


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def load_mint_model() -> Tuple[MintESM2, Alphabet, int]:
    if not os.path.exists(MINT_CFG_PATH):
        raise FileNotFoundError(f"MINT config not found: {MINT_CFG_PATH}")
    if not os.path.exists(MINT_CHECKPOINT_PATH):
        raise FileNotFoundError(f"MINT checkpoint not found: {MINT_CHECKPOINT_PATH}")

    print("Loading MINT model (multimer ESM2)...")
    cfg = load_config(MINT_CFG_PATH)
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
    state_dict = OrderedDict(
        (key.replace("model.", ""), value) for key, value in checkpoint["state_dict"].items()
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] MINT state dict load: missing={missing}, unexpected={unexpected}")
    model.eval().to(DEVICE)
    alphabet = Alphabet.from_architecture("ESM-1b")
    return model, alphabet, cfg.encoder_layers


# ---------------------------------------------------------------------------
# Sequence truncation helpers
# ---------------------------------------------------------------------------


def _extract_center_window(seq: str, center_idx: int, max_len: int) -> Tuple[str, int]:
    if len(seq) <= max_len:
        return seq, center_idx
    half = max_len // 2
    start = max(0, center_idx - half)
    end = start + max_len
    if end > len(seq):
        end = len(seq)
        start = end - max_len
    new_center = center_idx - start
    return seq[start:end], new_center


def truncate_pair_for_models(
    seq_a: str,
    seq_b: str,
    pos_1based: int,
) -> Tuple[str, str, int]:
    if pos_1based < 1 or pos_1based > len(seq_a):
        raise ValueError(f"Position {pos_1based} out of bounds for length {len(seq_a)}.")

    center_idx = pos_1based - 1
    a_residues = min(len(seq_a), MAX_MUT_RESIDUES)
    seq_a_window, new_center = _extract_center_window(seq_a, center_idx, a_residues)

    available_for_b = max(1, MAX_TOTAL_RESIDUES - len(seq_a_window))
    seq_b_window = seq_b[:available_for_b]

    new_pos = new_center + 1
    return seq_a_window, seq_b_window, new_pos


# ---------------------------------------------------------------------------
# MINT pair embedding utilities
# ---------------------------------------------------------------------------


def _encode_chain(seq: str, alphabet: Alphabet) -> torch.Tensor:
    tokens = alphabet.encode("<cls>" + seq.replace("J", "L") + "<eos>")
    return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def mint_pair_embedding(
    seq_a: str,
    seq_b: str,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layers: int,
) -> np.ndarray:
    tokens_a = _encode_chain(seq_a, alphabet)
    tokens_b = _encode_chain(seq_b, alphabet)

    total_tokens = tokens_a.size(0) + tokens_b.size(0)
    if total_tokens > MAX_MINT_TOKENS:
        raise ValueError(
            f"MINT token budget exceeded: seq_a={tokens_a.size(0)}, seq_b={tokens_b.size(0)}, total={total_tokens}."
        )

    chain_ids_a = torch.zeros_like(tokens_a, dtype=torch.int32)
    chain_ids_b = torch.ones_like(tokens_b, dtype=torch.int32)

    tokens = torch.cat([tokens_a, tokens_b], dim=0).unsqueeze(0).to(DEVICE)
    chain_ids = torch.cat([chain_ids_a, chain_ids_b], dim=0).unsqueeze(0).to(DEVICE)

    out = mint_model(tokens, chain_ids, repr_layers=[mint_layers])
    reps = out["representations"][mint_layers][0]  # [T, D]

    tok = tokens[0]
    valid_mask = (tok != alphabet.cls_idx) & (tok != alphabet.eos_idx) & (tok != alphabet.padding_idx)

    if valid_mask.any():
        pair_embed = reps[valid_mask].mean(dim=0)
    else:
        pair_embed = torch.zeros((reps.size(-1),), device=reps.device)

    return pair_embed.detach().cpu().to(torch.float32).numpy()


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _make_sample_key(row) -> str:
    """Build a stable sample key for mutation-interaction rows."""
    tgt = getattr(row, TARGET_ID_COL, "NA")
    pos = getattr(row, POSITION_COL)
    inter = getattr(row, INTERACTOR_ID_COL, "NA")

    mut_val = None
    if hasattr(row, MUTATION_COL):
        try:
            mut_val = getattr(row, MUTATION_COL)
        except Exception:
            mut_val = None

    if mut_val is None or (isinstance(mut_val, float) and pd.isna(mut_val)):
        before = getattr(row, "Before_AA", None)
        after = getattr(row, "After_AA", None)
        if before is not None and after is not None:
            mut_val = f"{before}{pos}{after}"

    if mut_val is None or (isinstance(mut_val, float) and pd.isna(mut_val)):
        mut_seq = getattr(row, MUT_SEQ_COL, "")
        mut_val = f"SEQ:{mut_seq}"

    return "||".join([str(tgt), str(mut_val), str(pos), str(inter)])


def build_features(
    df: pd.DataFrame,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layers: int,
) -> Tuple[np.ndarray, List[str]]:
    D = mint_model.embed_dim
    pair_cache: Dict[Tuple[str, str], np.ndarray] = {}

    features: List[np.ndarray] = []
    keys: List[str] = []

    iterator = tqdm(df.itertuples(index=False), total=len(df), desc="Build features (MINT pair)")
    for row in iterator:
        mut_seq_full = getattr(row, MUT_SEQ_COL)
        tgt_seq_full = getattr(row, TARGET_SEQ_COL)
        partner_seq_full = getattr(row, INTERACTOR_SEQ_COL)
        pos_orig = int(getattr(row, POSITION_COL))

        mut_seq_trim, partner_seq_trim_mut, _ = truncate_pair_for_models(mut_seq_full, partner_seq_full, pos_orig)
        tgt_seq_trim, partner_seq_trim_tgt, _ = truncate_pair_for_models(tgt_seq_full, partner_seq_full, pos_orig)

        key_mut_pair = (mut_seq_trim, partner_seq_trim_mut)
        if key_mut_pair not in pair_cache:
            pair_cache[key_mut_pair] = mint_pair_embedding(
                mut_seq_trim, partner_seq_trim_mut, mint_model, alphabet, mint_layers
            )
        embed_mut = pair_cache[key_mut_pair]

        key_tgt_pair = (tgt_seq_trim, partner_seq_trim_tgt)
        if key_tgt_pair not in pair_cache:
            pair_cache[key_tgt_pair] = mint_pair_embedding(
                tgt_seq_trim, partner_seq_trim_tgt, mint_model, alphabet, mint_layers
            )
        embed_tgt = pair_cache[key_tgt_pair]

        feat_vec = np.concatenate([embed_mut, embed_tgt], axis=0).astype(np.float32, copy=False)
        features.append(feat_vec)
        keys.append(_make_sample_key(row))

    if features:
        X = np.vstack(features).astype(np.float32, copy=False)
    else:
        X = np.zeros((0, 2 * D), dtype=np.float32)
    return X, keys


def prepare_feature_cache(
    df_unique: pd.DataFrame,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layers: int,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, List[str], str]:
    cache_root = os.path.join(CACHE_ROOT_PREFIX, f"mint_layer_L{mint_layers}")
    feature_dir = os.path.join(cache_root, FEATURE_CACHE_SUBDIR)
    os.makedirs(feature_dir, exist_ok=True)
    cache_path = os.path.join(feature_dir, "all_samples.npz")

    if force_rebuild and os.path.exists(cache_path):
        os.remove(cache_path)

    feature_dim = 2 * mint_model.embed_dim
    existing_features: Optional[np.ndarray] = None
    existing_keys: List[str] = []
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        existing_features = data["features"].astype(np.float32, copy=False)
        existing_keys = data["keys"].tolist()
        if existing_features.ndim != 2 or existing_features.shape[1] != feature_dim:
            raise ValueError(
                f"Cached feature shape mismatch: expected dim {feature_dim}, found {existing_features.shape}"
            )

    existing_key_set = set(existing_keys)
    missing_indices: List[int] = []
    for idx, row in enumerate(df_unique.itertuples(index=False)):
        key = _make_sample_key(row)
        if key not in existing_key_set:
            missing_indices.append(idx)

    if missing_indices:
        missing_df = df_unique.iloc[missing_indices].reset_index(drop=True)
        print(f"Feature cache miss: computing {len(missing_df)} samples ...")
        new_features, new_keys = build_features(
            missing_df, mint_model, alphabet, mint_layers
        )
        if existing_features is None or existing_features.size == 0:
            combined_features = new_features
            combined_keys = new_keys
        else:
            combined_features = np.concatenate([existing_features, new_features], axis=0).astype(np.float32, copy=False)
            combined_keys = existing_keys + new_keys
    else:
        print("Feature cache hit: reusing existing matrix.")
        combined_features = existing_features if existing_features is not None else np.zeros((0, feature_dim), dtype=np.float32)
        combined_keys = existing_keys

    np.savez(cache_path, features=combined_features, keys=np.array(combined_keys, dtype=object))
    print(f"Feature cache shape: {combined_features.shape} | path: {cache_path}")
    return combined_features, combined_keys, cache_path


def features_from_df(
    df: pd.DataFrame,
    features: np.ndarray,
    key_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    idxs: List[int] = []
    for row in df.itertuples(index=False):
        key = _make_sample_key(row)
        if key not in key_to_idx:
            raise KeyError(f"Feature cache missing key: {key}")
        idxs.append(key_to_idx[key])
    X = features[idxs]
    y = df[LABEL_COL].to_numpy().astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_13_metrics(
    y_true: np.ndarray,             # shape [N], int in {0,1,2,3}
    y_pred: np.ndarray,             # shape [N], int in {0,1,2,3}
    y_proba: np.ndarray,            # shape [N, C], C=4, softmax/softprob
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    13 metrics for 4-class setting:
    Accuracy
    Precision_macro / Precision_weighted
    Recall_macro / Recall_weighted
    F1_macro / F1_weighted
    Balanced_Acc
    ROC_AUC_macro_ovr
    PR_AUC_macro_ovr
    Youdens_J_macro_ovr
    t_test_t_mean_ovr
    p_value_fisher_ovr
    """
    from sklearn import metrics
    try:
        from scipy.stats import ttest_ind, combine_pvalues
        has_scipy = True
    except Exception:
        has_scipy = False
        ttest_ind = None
        combine_pvalues = None

    acc = metrics.accuracy_score(y_true, y_pred)
    prec_macro = metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)

    try:
        Y = np.eye(num_classes, dtype=int)[y_true]
        roc_macro = metrics.roc_auc_score(Y, y_proba, average="macro", multi_class="ovr")
    except Exception:
        roc_macro = float("nan")

    try:
        Y = np.eye(num_classes, dtype=int)[y_true]
        pr_macro = metrics.average_precision_score(Y, y_proba, average="macro")
    except Exception:
        pr_macro = float("nan")

    Js = []
    labels = list(range(num_classes))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    for c in labels:
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        Js.append(TPR + TNR - 1.0)
    youden_macro = float(np.mean(Js)) if Js else float("nan")

    if has_scipy:
        t_stats = []
        p_vals = []
        for c in labels:
            pos = y_proba[y_true == c, c]
            neg = y_proba[y_true != c, c]
            if pos.size > 1 and neg.size > 1:
                t_stat, p_val = ttest_ind(pos, neg, equal_var=False)
                t_stats.append(float(t_stat))
                p_vals.append(float(p_val))
        t_mean = float(np.mean(t_stats)) if t_stats else float("nan")
        if p_vals:
            _, p_fisher = combine_pvalues(p_vals, method="fisher")
        else:
            p_fisher = float("nan")
    else:
        t_mean = float("nan")
        p_fisher = float("nan")

    return {
        "Accuracy": acc,
        "Precision_macro": prec_macro,
        "Precision_weighted": prec_weighted,
        "Recall_macro": rec_macro,
        "Recall_weighted": rec_weighted,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "Balanced_Acc": bal_acc,
        "ROC_AUC_macro_ovr": roc_macro,
        "PR_AUC_macro_ovr": pr_macro,
        "Youdens_J_macro_ovr": youden_macro,
        "t_test_t_mean_ovr": t_mean,
        "p_value_fisher_ovr": p_fisher,
    }


def aggregate_metrics(metric_dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    metric_list = list(metric_dicts)
    if not metric_list:
        return {}
    agg: Dict[str, float] = {}
    for key in metric_list[0].keys():
        vals = [float(md[key]) for md in metric_list if isinstance(md.get(key), (int, float, np.floating))]
        if not vals:
            continue
        arr = np.array(vals, dtype=np.float64)
        agg[f"{key}_mean"] = float(np.nanmean(arr))
        agg[f"{key}_std"] = float(np.nanstd(arr))
    return agg


# ---------------------------------------------------------------------------
# Classifier heads
# ---------------------------------------------------------------------------


def run_xgboost_tenfold(
    fold_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]],
    features: np.ndarray,
    key_to_idx: Dict[str, int],
):
    if XGBClassifier is None:
        raise ImportError("xgboost>=1.6.0 is required.")

    fold_metrics = {}
    collected: List[Dict[str, float]] = []

    for fold_idx in range(NUM_FOLDS):
        print(f"\n=== Fold {fold_idx} / {NUM_FOLDS - 1} (XGBoost) ===")
        train_df, test_df = fold_data[fold_idx]
        X_train, y_train = features_from_df(train_df, features, key_to_idx)
        X_test, y_test = features_from_df(test_df, features, key_to_idx)

        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        X_test = np.ascontiguousarray(X_test, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int64)

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

        use_gpu = DEVICE == "cuda"
        if use_gpu:
            tree_method = "gpu_hist"
            predictor = "gpu_predictor"
            device_param = "cuda"
        else:
            tree_method = "hist"
            predictor = "auto"
            device_param = "cpu"

        clf = XGBClassifier(
            **base_params,
            tree_method=tree_method,
            predictor=predictor,
            device=device_param,
        )
        clf.fit(X_train, y_train, verbose=False)

        proba = clf.predict_proba(X_test)  # shape [N,4]
        pred = np.argmax(proba, axis=1).astype(np.int32)
        metrics_fold = compute_13_metrics(y_test.astype(np.int64), pred, proba, num_classes=4)

        fold_metrics[fold_idx] = metrics_fold
        collected.append(metrics_fold)

        for k, v in metrics_fold.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        if use_gpu:
            del X_train, y_train, X_test
            _release_gpu_memory()
        else:
            if DEVICE == "cuda":
                _release_gpu_memory()
        del clf

    summary = aggregate_metrics(collected)
    print("\n=== Aggregated (mean/std) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")
    return fold_metrics, summary


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        hidden_layers = [hidden_dim]
        if num_layers == 3:
            hidden_layers.append(hidden_dim)
        for hidden in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 4))  # 4-class logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mlp_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for xb, yb in data_loader:
        xb = xb.to(device, non_blocking=True)
        optimizer.zero_grad()
        yb = yb.to(device, non_blocking=True).long()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / max(1, len(data_loader.dataset))


def _evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for xb in data_loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)


def run_mlp_tenfold(
    fold_data: Dict[int, Tuple[pd.DataFrame, pd.DataFrame]],
    features: np.ndarray,
    key_to_idx: Dict[str, int],
    args: argparse.Namespace,
):
    fold_metrics = {}
    collected_metrics = []
    input_dim = features.shape[1]

    for fold_idx in range(NUM_FOLDS):
        print(f"\n=== Fold {fold_idx} / {NUM_FOLDS - 1} (MLP) ===")
        train_df, test_df = fold_data[fold_idx]
        X_train, y_train = features_from_df(train_df, features, key_to_idx)
        X_test, y_test = features_from_df(test_df, features, key_to_idx)

        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        eval_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float()), batch_size=args.batch_size, shuffle=False)

        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=args.mlp_hidden,
            num_layers=args.mlp_layers,
            dropout=args.mlp_dropout,
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        for epoch in range(args.epochs):
            loss_epoch = _mlp_epoch(model, train_loader, criterion, optimizer, DEVICE)
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} - loss: {loss_epoch:.4f}")

        proba = _evaluate_model(model, eval_loader, DEVICE)
        pred = np.argmax(proba, axis=1).astype(np.int32)
        metrics_fold = compute_13_metrics(y_test.astype(np.int64), pred, proba, num_classes=4)
        fold_metrics[fold_idx] = metrics_fold
        collected_metrics.append(metrics_fold)

        for k, v in metrics_fold.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        del model
        _release_gpu_memory()

    summary = aggregate_metrics(collected_metrics)
    print("\n=== Aggregated (mean/std) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")
    return fold_metrics, summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    print("MINT Pair Embedding Protein Interaction Prediction (ablation) - Ten Fold Evaluation")
    print("=" * 80)

    df_all = load_dataset()
    if df_all.empty:
        raise RuntimeError("Dataset is empty.")
    print(f"Total samples: {len(df_all)}")
    fold_data = create_tenfold_splits(df_all)

    mint_model, mint_alphabet, mint_layers = load_mint_model()

    features, keys, cache_path = prepare_feature_cache(
        df_all,
        mint_model,
        mint_alphabet,
        mint_layers,
        force_rebuild=args.force_rebuild,
    )
    key_to_idx = {k: i for i, k in enumerate(keys)}

    if DEVICE == "cuda":
        print("\nReleasing MINT GPU memory before classifier training...")
        del mint_model
        torch.cuda.empty_cache()

    if args.classifier == "xgb":
        fold_metrics, summary = run_xgboost_tenfold(fold_data, features, key_to_idx)
        clf_tag = "XGB"
    else:
        fold_metrics, summary = run_mlp_tenfold(fold_data, features, key_to_idx, args)
        clf_tag = "MLP"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, RESULT_BASENAME.format(clf=clf_tag))
    metric_order = [
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
    with open(result_path, "w", encoding="utf-8") as f:
        for fold_idx in range(NUM_FOLDS):
            f.write(f"=== Fold {fold_idx} ===\n")
            metrics_fold = fold_metrics[fold_idx]
            for key in metric_order:
                val = metrics_fold.get(key, float("nan"))
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
    print(f"\nResults written to: {result_path}")
    print(f"Feature cache file: {cache_path}")
    print("Ten-fold evaluation complete.")


if __name__ == "__main__":
    main()
