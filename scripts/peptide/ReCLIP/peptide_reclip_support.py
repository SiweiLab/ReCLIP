#!/usr/bin/env python3
"""
ReCLIP Class-I Epitope Interaction Cross-Prediction.

Feature pipeline:
 1. Epitope sequences are encoded with HuggingFace ESM2; top-K residues around
    the pseudo-mutation site (middle residue of the epitope) provide the epitope
    representation.
 2. Interactor sequences are embedded with ESM2, and their residue embeddings are
    aggregated with weights extracted from the MINT cross-chain attention
    between the pseudo-mutation site and every interactor residue.
 3. Feature vectors are cached once and reused to train on rows with `Set != "Test"`
    while evaluating with stratified K-fold splits drawn only from the `Set == "Test"`
    subset (cross-prediction on the held-out set).

Computed features are cached on disk (`Feature_cache/all_samples.npz`) so that
subsequent runs reuse the stored matrix instead of recomputing expensive model
forward passes.
"""

from __future__ import annotations

import argparse
import os
import random
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
from transformers import AutoTokenizer, EsmModel

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

from mint.mint.data import Alphabet
from mint.mint.helpers.extract import load_config
from mint.mint.model.esm2 import ESM2 as MintESM2

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
SELECT_LAYER = 23
MINT_SELECT_LAYER = -1
MAX_HF_TOKENS = 1022
MAX_MINT_TOKENS = 1024
MAX_MUT_RESIDUES = (MAX_MINT_TOKENS // 2) - 2  # reserve tokens & balance partner window
MAX_TOTAL_RESIDUES = MAX_MINT_TOKENS - 4
K_SEQ = 7

DEFAULT_DATASET_PATH = os.path.join("SWING", "Data", "ClassI_Model", "ClassI_training_210.csv")
DEFAULT_LOOPS = 10

MINT_CFG_PATH = os.path.join("mint", "data", "esm2_t33_650M_UR50D.json")
MINT_CHECKPOINT_PATH = os.path.join("mint", "mint.ckpt")

MUT_SEQ_COL_RAW: Optional[str] = None
MUT_SEQ_COL = "Epitope"
INTERACTOR_SEQ_COL = "Sequence"
POSITION_COL = "Pseudo_Position"
LABEL_COL = "Hit"
SET_COL = "Set"
REQ_COLS = [MUT_SEQ_COL, INTERACTOR_SEQ_COL, LABEL_COL]
SAMPLE_KEY_COLS = [MUT_SEQ_COL, POSITION_COL, INTERACTOR_SEQ_COL]

CACHE_ROOT_PREFIX = "classI_result_reclip"
FEATURE_CACHE_SUBDIR = "Feature_cache"
RESULTS_DIR = "Results"
RESULT_BASENAME = "result_classI_reclip_crosspred_{clf}_{loops}fold_{hf_tag}_{mint_tag}_IEk.txt"

# ---------------------------------------------------------------------------
# Argument parsing / reproducibility helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReCLIP Class-I epitope cross-prediction."
    )
    parser.add_argument(
        "--data-set",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help="CSV containing Epitope/Sequence/Hit/Set columns. Default: %(default)s",
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
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for the MLP classifier.")
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
        "--loops",
        type=int,
        default=DEFAULT_LOOPS,
        help="Number of stratified folds sampled from the Set=='Test' subset.",
    )
    parser.add_argument(
        "--mint-layer",
        type=int,
        default=MINT_SELECT_LAYER,
        help="Which MINT attention layer to use (0-based, -1 for last).",
    )
    parser.add_argument(
        "--hf-layer",
        type=int,
        default=SELECT_LAYER,
        help="Which HuggingFace ESM2 layer to extract representations from (0-based, -1 for last).",
    )
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


def _default_mutation_position(seq: str) -> int:
    seq = (seq or "").strip()
    if not seq:
        return 1
    return max(1, len(seq) // 2 + 1)


def _prep_for_windows(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.dropna(subset=REQ_COLS).copy()
    df2[MUT_SEQ_COL] = df2[MUT_SEQ_COL].astype(str).str.strip()
    df2[INTERACTOR_SEQ_COL] = df2[INTERACTOR_SEQ_COL].astype(str).str.strip()

    non_empty = (df2[MUT_SEQ_COL] != "") & (df2[INTERACTOR_SEQ_COL] != "")
    dropped_empty = (~non_empty).sum()
    if dropped_empty > 0:
        print(f"[warn] Dropping {dropped_empty} rows with empty epitope/interactor sequences.")
    df2 = df2.loc[non_empty].copy()

    if POSITION_COL not in df2.columns:
        df2[POSITION_COL] = df2[MUT_SEQ_COL].apply(_default_mutation_position)
    else:
        df2[POSITION_COL] = df2[POSITION_COL].astype(int)

    seq_lengths = df2[MUT_SEQ_COL].str.len()
    valid_mask = (df2[POSITION_COL] >= 1) & (df2[POSITION_COL] <= seq_lengths)
    dropped_outside = (~valid_mask).sum()
    if dropped_outside > 0:
        print(f"[warn] Dropping {dropped_outside} rows with pseudo positions outside sequence length.")
    df2 = df2.loc[valid_mask].copy()

    df2[LABEL_COL] = df2[LABEL_COL].astype(int)
    return df2.reset_index(drop=True)


def load_dataset(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    print(f"Loading dataset: {data_path}")
    df_raw = pd.read_csv(data_path)
    if MUT_SEQ_COL_RAW and MUT_SEQ_COL_RAW in df_raw.columns:
        df_raw = df_raw.rename(columns={MUT_SEQ_COL_RAW: MUT_SEQ_COL})
    df_clean = _prep_for_windows(df_raw)
    print(f"  rows: {len(df_raw)} -> {len(df_clean)} | dropped: {len(df_raw) - len(df_clean)}")
    return df_clean.reset_index(drop=True)


def split_train_test_by_set(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if SET_COL not in df.columns:
        raise KeyError(f"Dataset is missing the {SET_COL} column required for Train/Test splitting.")
    set_values = df[SET_COL].astype(str).str.strip().str.lower()
    test_mask = set_values == "test"
    train_df = df.loc[~test_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("Training split (Set != 'Test') is empty.")
    if test_df.empty:
        raise ValueError("Test split (Set == 'Test') is empty.")
    print(f"Train rows (Set!=Test): {len(train_df)} | Test rows (Set==Test): {len(test_df)}")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def _norm_single_layer(select_layer: int, num_layers: int) -> int:
    li = select_layer
    if li < 0:
        li = num_layers + li
    if not (0 <= li < num_layers):
        raise ValueError(f"Invalid layer index {select_layer} for model with {num_layers} layers.")
    return li


def load_hf_esm2(select_layer: int) -> Tuple[AutoTokenizer, EsmModel, int, str]:
    print("Loading ESM2 (HuggingFace) model...")
    hf_tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    hf_model = EsmModel.from_pretrained(HF_MODEL_ID, output_attentions=True).eval().to(DEVICE)
    layer_idx = _norm_single_layer(select_layer, hf_model.config.num_hidden_layers)
    layer_tag = f"L{layer_idx}"
    return hf_tok, hf_model, layer_idx, layer_tag


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
    mut_seq: str,
    partner_seq: str,
    mut_pos_1based: int,
) -> Tuple[str, str, int]:
    if mut_pos_1based < 1 or mut_pos_1based > len(mut_seq):
        raise ValueError(f"Mutated position {mut_pos_1based} out of bounds for length {len(mut_seq)}.")

    mut_center_idx = mut_pos_1based - 1
    mut_residues = min(len(mut_seq), MAX_MUT_RESIDUES)
    mut_window, new_center = _extract_center_window(mut_seq, mut_center_idx, mut_residues)

    available_for_partner = max(1, MAX_TOTAL_RESIDUES - len(mut_window))
    partner_window = partner_seq[:available_for_partner]

    new_mut_pos = new_center + 1
    return mut_window, partner_window, new_mut_pos


# ---------------------------------------------------------------------------
# HuggingFace ESM2 feature utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def hf_token_emb_aa(
    seq: str,
    select_layer_idx: int,
    hf_tok: AutoTokenizer,
    hf_model: EsmModel,
) -> np.ndarray:
    if not seq:
        return np.zeros((0, hf_model.config.hidden_size), dtype=np.float32)

    s = seq[: MAX_HF_TOKENS - 2]
    enc = hf_tok(s, return_tensors="pt", add_special_tokens=True)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    out = hf_model(**enc, output_hidden_states=True, output_attentions=False)
    hidden_states = out.hidden_states
    li = _norm_single_layer(select_layer_idx, len(hidden_states) - 1)
    H = hidden_states[li + 1][0]
    T = enc["input_ids"].shape[1]
    H_aa = H[1 : T - 1]
    return H_aa.detach().cpu().to(torch.float32).numpy()


@torch.no_grad()
def hf_self_attention_vector(
    seq: str,
    mut_pos_1based: int,
    select_layer_idx: int,
    hf_tok: AutoTokenizer,
    hf_model: EsmModel,
) -> np.ndarray:
    if not seq:
        return np.zeros((0,), dtype=np.float32)

    s = seq[: MAX_HF_TOKENS - 2]
    enc = hf_tok(s, return_tensors="pt", add_special_tokens=True)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    out = hf_model(**enc, output_attentions=True)
    attentions = out.attentions
    li = _norm_single_layer(select_layer_idx, len(attentions))

    att_layer = attentions[li][0].mean(dim=0)
    tokens = enc["input_ids"][0]
    toks = hf_tok.convert_ids_to_tokens(tokens)
    aa_start = 1
    aa_end = len(toks) - 1
    if not (1 <= mut_pos_1based <= (aa_end - aa_start)):
        raise ValueError(f"Mutated position {mut_pos_1based} out of truncated bounds {aa_end - aa_start}.")

    token_index = aa_start + (mut_pos_1based - 1)
    vec = att_layer[token_index, aa_start:aa_end]
    return vec.detach().cpu().to(torch.float32).numpy()


# ---------------------------------------------------------------------------
# MINT cross-attention utilities
# ---------------------------------------------------------------------------


def _encode_chain(seq: str, alphabet: Alphabet) -> torch.Tensor:
    tokens = alphabet.encode("<cls>" + seq.replace("J", "L") + "<eos>")
    return torch.tensor(tokens, dtype=torch.long)


@torch.no_grad()
def mint_cross_attention_weights(
    mut_seq: str,
    partner_seq: str,
    mut_pos_1based: int,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layer_idx: int,
) -> np.ndarray:
    mut_tokens = _encode_chain(mut_seq, alphabet)
    partner_tokens = _encode_chain(partner_seq, alphabet)

    total_tokens = mut_tokens.size(0) + partner_tokens.size(0)
    if total_tokens > MAX_MINT_TOKENS:
        raise ValueError(
            f"MINT token budget exceeded: mutated={mut_tokens.size(0)}, "
            f"partner={partner_tokens.size(0)}, total={total_tokens}."
        )

    mut_chain_ids = torch.zeros_like(mut_tokens, dtype=torch.int32)
    partner_chain_ids = torch.ones_like(partner_tokens, dtype=torch.int32)

    tokens = torch.cat([mut_tokens, partner_tokens], dim=0).unsqueeze(0).to(DEVICE)
    chain_ids = torch.cat([mut_chain_ids, partner_chain_ids], dim=0).unsqueeze(0).to(DEVICE)

    out = mint_model(tokens, chain_ids, need_head_weights=True)
    if "attentions" not in out:
        raise RuntimeError("MINT model did not return attentions; ensure need_head_weights=True.")
    attentions = out["attentions"]
    if mint_layer_idx < 0 or mint_layer_idx >= attentions.shape[1]:
        raise ValueError(f"MINT layer index {mint_layer_idx} out of bounds for attentions with {attentions.shape[1]} layers.")
    last_attn = attentions[0, mint_layer_idx]

    mut_token_index = mut_pos_1based
    mut_token_count = mut_tokens.size(0)
    partner_token_count = partner_tokens.size(0)
    partner_start = mut_token_count

    partner_indices = torch.arange(
        partner_start + 1,
        partner_start + partner_token_count - 1,
        device=last_attn.device,
        dtype=torch.long,
    )
    if partner_indices.numel() == 0:
        return np.zeros((0,), dtype=np.float32)

    att_head_avg = last_attn[:, mut_token_index, partner_indices].mean(dim=0)
    att_weights = torch.softmax(att_head_avg, dim=0)
    return att_weights.detach().cpu().to(torch.float32).numpy()


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def _make_sample_key(row) -> str:
    return "||".join(str(getattr(row, col)) for col in SAMPLE_KEY_COLS)


def build_features(
    df: pd.DataFrame,
    hf_tok: AutoTokenizer,
    hf_model: EsmModel,
    select_layer_idx: int,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layer_idx: int,
) -> Tuple[np.ndarray, List[str]]:
    D = hf_model.config.hidden_size
    seq_embed_cache: Dict[str, np.ndarray] = {}
    attn_vec_cache: Dict[Tuple[str, int], np.ndarray] = {}
    partner_embed_cache: Dict[str, np.ndarray] = {}
    mint_cache: Dict[Tuple[str, str, int], np.ndarray] = {}

    features: List[np.ndarray] = []
    keys: List[str] = []

    iterator = tqdm(df.itertuples(index=False), total=len(df), desc="Build features")
    for row in iterator:
        mut_seq_full = getattr(row, MUT_SEQ_COL)
        partner_seq_full = getattr(row, INTERACTOR_SEQ_COL)
        mut_pos_orig = int(getattr(row, POSITION_COL))

        mut_seq_trim, partner_seq_trim, mut_pos_trim = truncate_pair_for_models(
            mut_seq_full, partner_seq_full, mut_pos_orig
        )

        if mut_seq_trim not in seq_embed_cache:
            seq_embed_cache[mut_seq_trim] = hf_token_emb_aa(mut_seq_trim, select_layer_idx, hf_tok, hf_model)
        H_mut = seq_embed_cache[mut_seq_trim]

        key_mut = (mut_seq_trim, mut_pos_trim)
        if key_mut not in attn_vec_cache:
            attn_vec_cache[key_mut] = hf_self_attention_vector(
                mut_seq_trim, mut_pos_trim, select_layer_idx, hf_tok, hf_model
            )
        mut_attention_vec = attn_vec_cache[key_mut]

        L_mut = min(mut_attention_vec.shape[0], H_mut.shape[0])
        if L_mut == 0:
            mut_feat = np.zeros((K_SEQ * D,), dtype=np.float32)
        else:
            weights = mut_attention_vec[:L_mut]
            embeddings = H_mut[:L_mut]
            top_k = min(K_SEQ, L_mut)
            top_idx = np.argsort(-weights)[:top_k]
            top_emb = embeddings[top_idx]
            if top_emb.shape[0] < K_SEQ:
                padding = np.zeros((K_SEQ - top_emb.shape[0], D), dtype=np.float32)
                top_emb = np.concatenate([top_emb, padding], axis=0)
            mut_feat = top_emb.reshape(K_SEQ * D).astype(np.float32)

        if partner_seq_trim not in partner_embed_cache:
            partner_embed_cache[partner_seq_trim] = hf_token_emb_aa(
                partner_seq_trim, select_layer_idx, hf_tok, hf_model
            )
        H_partner = partner_embed_cache[partner_seq_trim]

        key_mint = (mut_seq_trim, partner_seq_trim, mut_pos_trim)
        if key_mint not in mint_cache:
            mint_cache[key_mint] = mint_cross_attention_weights(
                mut_seq_trim, partner_seq_trim, mut_pos_trim, mint_model, alphabet, mint_layer_idx
            )
        mint_weights = mint_cache[key_mint]

        L_partner = min(H_partner.shape[0], mint_weights.shape[0])
        if L_partner == 0:
            partner_feat = np.zeros((D,), dtype=np.float32)
        else:
            weights = mint_weights[:L_partner]
            weights = weights / (weights.sum() + 1e-9)
            embeddings = H_partner[:L_partner]
            partner_feat = (embeddings * weights[:, None]).sum(axis=0).astype(np.float32)

        feat_vec = np.concatenate([mut_feat, partner_feat], axis=0)
        features.append(feat_vec)
        keys.append(_make_sample_key(row))

    if features:
        X = np.vstack(features).astype(np.float32, copy=False)
    else:
        X = np.zeros((0, K_SEQ * D + D), dtype=np.float32)
    return X, keys


def prepare_feature_cache(
    df_unique: pd.DataFrame,
    hf_tok: AutoTokenizer,
    hf_model: EsmModel,
    select_layer_idx: int,
    layer_tag: str,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layer_idx: int,
    mint_layer_tag: str,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, List[str], str]:
    attn_cache_root = os.path.join(CACHE_ROOT_PREFIX, f"layer_{layer_tag}_mint_{mint_layer_tag}")
    feature_dir = os.path.join(attn_cache_root, FEATURE_CACHE_SUBDIR)
    os.makedirs(feature_dir, exist_ok=True)
    cache_path = os.path.join(feature_dir, "all_samples.npz")

    if force_rebuild and os.path.exists(cache_path):
        os.remove(cache_path)

    feature_dim = K_SEQ * hf_model.config.hidden_size + hf_model.config.hidden_size
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
            missing_df, hf_tok, hf_model, select_layer_idx, mint_model, alphabet, mint_layer_idx
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
    y = df[LABEL_COL].to_numpy().astype(np.float32)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"Label counts -> positive(1): {pos}, negative(0): {neg}")
    return X, y


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def compute_ten_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    try:
        from scipy.stats import ttest_ind  # type: ignore

        has_scipy = True
    except Exception:
        has_scipy = False

    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    bal = metrics.balanced_accuracy_score(y_true, y_pred)
    roc = metrics.roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else float("nan")
    try:
        ap = metrics.average_precision_score(y_true, y_proba)
    except Exception:
        ap = float("nan")

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    youden_j = tpr + tnr - 1

    if has_scipy:
        pos = y_proba[y_true == 1]
        neg = y_proba[y_true == 0]
        if len(pos) > 1 and len(neg) > 1:
            t_stat, p_val = ttest_ind(pos, neg, equal_var=False)
        else:
            t_stat, p_val = float("nan"), float("nan")
    else:
        t_stat, p_val = float("nan"), float("nan")

    return {
        "ACC": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "Balanced_Accuracy": bal,
        "ROC_AUC": roc,
        "PR_AUC(AP)": ap,
        "Youdens_J": youden_j,
        "t_test_t": t_stat,
        "p_value": p_val,
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


def run_xgboost_crosspred(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: np.ndarray,
    key_to_idx: Dict[str, int],
    loops: int,
):
    if XGBClassifier is None:
        raise ImportError("xgboost>=1.6.0 is required.")

    X_train, y_train = features_from_df(train_df, features, key_to_idx)
    X_test_full, y_test_full = features_from_df(test_df, features, key_to_idx)

    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    r = neg / max(1.0, pos)
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
        scale_pos_weight=float(1.3 * r),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    use_gpu = DEVICE == "cuda"
    if use_gpu and not CUPY_AVAILABLE:
        print("[warn] CUDA is available but CuPy is not installed; falling back to CPU mode.")
        use_gpu = False

    params = dict(base_params)
    if use_gpu:
        params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    else:
        params.update(tree_method="hist", predictor="auto")

    try:
        clf = XGBClassifier(**params)
    except TypeError:
        legacy_params = dict(base_params)
        legacy_params["tree_method"] = "gpu_hist" if use_gpu else "hist"
        clf = XGBClassifier(**legacy_params)

    if use_gpu:
        X_train_dev = cp.asarray(X_train)
        y_train_dev = cp.asarray(y_train)
    else:
        X_train_dev = X_train
        y_train_dev = y_train

    clf.fit(X_train_dev, y_train_dev, verbose=False)

    fold_metrics: Dict[int, Dict[str, float]] = {}
    collected: List[Dict[str, float]] = []
    skf = StratifiedKFold(n_splits=loops, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (_, val_idx) in enumerate(skf.split(X_test_full, y_test_full)):
        print(f"\n=== Cross-Pred Fold {fold_idx + 1} / {loops} (XGBoost) ===")
        X_val = np.ascontiguousarray(X_test_full[val_idx], dtype=np.float32)
        y_val = y_test_full[val_idx]

        if use_gpu:
            X_val_dev = cp.asarray(X_val)
            proba = clf.predict_proba(X_val_dev)[:, 1]
            if isinstance(proba, cp.ndarray):
                proba = cp.asnumpy(proba)
            del X_val_dev
        else:
            proba = clf.predict_proba(X_val)[:, 1]

        pred = (proba >= 0.5).astype(np.int32)
        metrics_fold = compute_ten_metrics(y_val, pred, proba)
        fold_metrics[fold_idx] = metrics_fold
        collected.append(metrics_fold)

        for k, v in metrics_fold.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

        _release_gpu_memory()

    summary = aggregate_metrics(collected)
    print("\n=== Aggregated (mean/std) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    if use_gpu:
        del X_train_dev, y_train_dev
        _release_gpu_memory()
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
        layers.append(nn.Linear(in_dim, 1))
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
        yb = yb.to(device, non_blocking=True).unsqueeze(1)
        optimizer.zero_grad()
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
            prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs, axis=0) if probs else np.zeros((0,), dtype=np.float32)


def run_mlp_crosspred(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: np.ndarray,
    key_to_idx: Dict[str, int],
    args: argparse.Namespace,
):
    X_train, y_train = features_from_df(train_df, features, key_to_idx)
    X_test_full, y_test_full = features_from_df(test_df, features, key_to_idx)

    input_dim = features.shape[1]
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=args.mlp_hidden,
        num_layers=args.mlp_layers,
        dropout=args.mlp_dropout,
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        loss_epoch = _mlp_epoch(model, train_loader, criterion, optimizer, DEVICE)
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{args.epochs} - loss: {loss_epoch:.4f}")

    skf = StratifiedKFold(n_splits=args.loops, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = {}
    collected_metrics: List[Dict[str, float]] = []

    for fold_idx, (_, val_idx) in enumerate(skf.split(X_test_full, y_test_full)):
        print(f"\n=== Cross-Pred Fold {fold_idx + 1} / {args.loops} (MLP) ===")
        X_val = X_test_full[val_idx]
        y_val = y_test_full[val_idx]
        eval_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val).float()),
            batch_size=args.batch_size,
            shuffle=False,
        )
        proba = _evaluate_model(model, eval_loader, DEVICE)
        pred = (proba >= 0.5).astype(np.int32)
        metrics_fold = compute_ten_metrics(y_val, pred, proba)
        fold_metrics[fold_idx] = metrics_fold
        collected_metrics.append(metrics_fold)

        for k, v in metrics_fold.items():
            if isinstance(v, (int, float, np.floating)):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")

    summary = aggregate_metrics(collected_metrics)
    print("\n=== Aggregated (mean/std) ===")
    for k, v in summary.items():
        print(f"{k}: {v:.6f}")

    del model
    _release_gpu_memory()
    return fold_metrics, summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    print("ReCLIP Class-I Epitope Interaction Cross-Prediction")
    print("=" * 80)

    df_all = load_dataset(args.data_set)
    if df_all.empty:
        raise RuntimeError("Dataset is empty.")
    print(f"Total samples: {len(df_all)}")
    train_df, test_df = split_train_test_by_set(df_all)

    hf_tok, hf_model, select_layer_idx, layer_tag = load_hf_esm2(args.hf_layer)
    mint_model, mint_alphabet, mint_layers = load_mint_model()
    mint_select_layer_idx = _norm_single_layer(args.mint_layer, mint_layers)
    mint_layer_tag = f"M{mint_select_layer_idx}"

    features, keys, cache_path = prepare_feature_cache(
        df_all,
        hf_tok,
        hf_model,
        select_layer_idx,
        layer_tag,
        mint_model,
        mint_alphabet,
        mint_select_layer_idx,
        mint_layer_tag,
        force_rebuild=args.force_rebuild,
    )
    key_to_idx = {k: i for i, k in enumerate(keys)}

    if DEVICE == "cuda":
        print("\nReleasing ESM2 / MINT GPU memory before classifier training...")
        del hf_model
        del mint_model
        torch.cuda.empty_cache()

    if args.classifier == "xgb":
        fold_metrics, summary = run_xgboost_crosspred(train_df, test_df, features, key_to_idx, args.loops)
        clf_tag = "XGB"
    else:
        fold_metrics, summary = run_mlp_crosspred(train_df, test_df, features, key_to_idx, args)
        clf_tag = "MLP"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(
        RESULTS_DIR,
        RESULT_BASENAME.format(
            clf=clf_tag,
            loops=args.loops,
            hf_tag=layer_tag,
            mint_tag=mint_layer_tag,
        ),
    )
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
    with open(result_path, "w", encoding="utf-8") as f:
        for fold_idx in range(args.loops):
            f.write(f"=== Fold {fold_idx} ===\n")
            metrics_fold = fold_metrics.get(fold_idx, {})
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
