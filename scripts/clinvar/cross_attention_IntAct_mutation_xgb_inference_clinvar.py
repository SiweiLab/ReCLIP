#!/usr/bin/env python3
"""
Inference script for:
ESM2 + MINT Cross-Attention Mutation Interaction Prediction (XGBoost best.pkl)

Usage (example):
  python inference_mutation_mint_xgb.py \
    --input ukbb_with_interactors.expanded_random5.txt \
    --model Results/best.pkl \
    --output ukbb_with_interactors.expanded_random5.scored.txt \
    --sep "\t"
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from mint.data import Alphabet
from mint.helpers.extract import load_config
from mint.model.esm2 import ESM2 as MintESM2

# ===================== Global config (keep consistent with the training script) =====================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
SELECT_LAYER = 23
MAX_HF_TOKENS = 1022
MAX_MINT_TOKENS = 1024
MAX_MUT_RESIDUES = (MAX_MINT_TOKENS // 2) - 2
MAX_TOTAL_RESIDUES = MAX_MINT_TOKENS - 4
K_SEQ = 5

# MINT config/checkpoint paths should be consistent with the training script
MINT_CFG_PATH = os.path.abspath("data/esm2_t33_650M_UR50D.json")
MINT_CHECKPOINT_PATH = os.path.abspath("mint.ckpt")

TARGET_ID_COL = "AM_uniprot_id"
MUT_SEQ_COL_RAW1 = "Mutated_Seq"
MUT_SEQ_COL_RAW2 = "Mutated_Seq"
MUT_SEQ_COL = "Mutated_Seq"
INTERACTOR_ID_COL = "Interactor_UPID"
INTERACTOR_SEQ_COL = "Interactor_Seq"
POSITION_COL = "Position"
MUTATION_COL = "Mutation"

REQ_COLS = [POSITION_COL, MUT_SEQ_COL, INTERACTOR_SEQ_COL]

CACHE_ROOT_PREFIX = "mutation_result_mint_cross_attn"
FEATURE_CACHE_SUBDIR = "Feature_cache_clinvar"


# ===================== Argparse =====================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inference on new mutation-interaction dataset using best XGBoost model."
    )
    p.add_argument("--input", required=True, help="Path to input TSV/CSV (same schema as training).")
    p.add_argument("--model", required=True, help="Path to best.pkl (trained XGBoost).")
    p.add_argument("--output", required=True, help="Path to save scored table.")
    p.add_argument(
        "--sep",
        default="\t",
        help="Input/output field separator (default: '\\t'). Use ',' for CSV."
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute missing features even if cache exists."
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="If >0, additionally split inference output into chunks of this many rows per file.",
    )
    return p.parse_args()


# ===================== Utils & loaders =====================

def _norm_single_layer(select_layer: int, num_layers: int) -> int:
    li = select_layer
    if li < 0:
        li = num_layers + li
    if not (0 <= li < num_layers):
        raise ValueError(f"Invalid layer index {select_layer} for model with {num_layers} layers.")
    return li


def load_hf_esm2() -> Tuple[AutoTokenizer, EsmModel, int, str]:
    print("Loading ESM2 (HuggingFace) model for inference ...")
    hf_tok = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    hf_model = EsmModel.from_pretrained(HF_MODEL_ID, output_attentions=True).eval().to(DEVICE)
    layer_idx = _norm_single_layer(SELECT_LAYER, hf_model.config.num_hidden_layers)
    layer_tag = f"L{layer_idx}"
    return hf_tok, hf_model, layer_idx, layer_tag


def load_mint_model() -> Tuple[MintESM2, Alphabet, int]:
    if not os.path.exists(MINT_CFG_PATH):
        raise FileNotFoundError(f"MINT config not found: {MINT_CFG_PATH}")
    if not os.path.exists(MINT_CHECKPOINT_PATH):
        raise FileNotFoundError(f"MINT checkpoint not found: {MINT_CHECKPOINT_PATH}")

    print("Loading MINT model (multimer ESM2) for inference ...")
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
    mint_layers: int,
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
    last_attn = attentions[0, mint_layers - 1]

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


def _make_sample_key(row) -> str:
    """
    Key rule exactly consistent with the training script:
    Target_UPID + Mutation / (Before_AA, After_AA) / seq + Position + Interactor_UPID
    """
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
    hf_tok: AutoTokenizer,
    hf_model: EsmModel,
    select_layer_idx: int,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layers: int,
) -> Tuple[np.ndarray, List[str]]:
    D = hf_model.config.hidden_size
    seq_embed_cache: Dict[str, np.ndarray] = {}
    attn_vec_cache: Dict[Tuple[str, int], np.ndarray] = {}
    partner_embed_cache: Dict[str, np.ndarray] = {}
    mint_cache: Dict[Tuple[str, str, int], np.ndarray] = {}

    features: List[np.ndarray] = []
    keys: List[str] = []

    iterator = tqdm(df.itertuples(index=False), total=len(df), desc="Build features (inference)")
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
                mut_seq_trim, partner_seq_trim, mut_pos_trim, mint_model, alphabet, mint_layers
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


def prepare_feature_cache_for_df(
    df: pd.DataFrame,
    hf_tok: AutoTokenizer,
    hf_model: EsmModel,
    select_layer_idx: int,
    layer_tag: str,
    mint_model: MintESM2,
    alphabet: Alphabet,
    mint_layers: int,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, List[str], str]:
    """
    Similar to the training script: maintain a global all_samples.npz,
    and incrementally compute missing samples if it already exists.
    """
    attn_cache_root = os.path.join(CACHE_ROOT_PREFIX, f"layer_{layer_tag}")
    feature_dir = os.path.join(attn_cache_root, FEATURE_CACHE_SUBDIR)
    os.makedirs(feature_dir, exist_ok=True)
    cache_path = os.path.join(feature_dir, "all_samples.npz")

    if force_rebuild and os.path.exists(cache_path):
        os.remove(cache_path)

    feature_dim = K_SEQ * hf_model.config.hidden_size + hf_model.config.hidden_size
    existing_features = None
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
    for idx, row in enumerate(df.itertuples(index=False)):
        key = _make_sample_key(row)
        if key not in existing_key_set:
            missing_indices.append(idx)

    if missing_indices:
        missing_df = df.iloc[missing_indices].reset_index(drop=True)
        print(f"Feature cache miss: computing {len(missing_df)} samples (inference) ...")
        new_features, new_keys = build_features(
            missing_df, hf_tok, hf_model, select_layer_idx, mint_model, alphabet, mint_layers
        )
        if existing_features is None or existing_features.size == 0:
            combined_features = new_features
            combined_keys = new_keys
        else:
            combined_features = np.concatenate([existing_features, new_features], axis=0).astype(
                np.float32, copy=False
            )
            combined_keys = existing_keys + new_keys
    else:
        print("Feature cache hit: no new samples to compute for inference.")
        if existing_features is None:
            combined_features = np.zeros((0, feature_dim), dtype=np.float32)
        else:
            combined_features = existing_features
        combined_keys = existing_keys

    np.savez(cache_path, features=combined_features, keys=np.array(combined_keys, dtype=object))
    print(f"[INFO] Global feature cache shape: {combined_features.shape} | path: {cache_path}")
    return combined_features, combined_keys, cache_path


# ===================== Dataset loader for inference =====================

def load_inference_dataset(path: str, sep: str) -> pd.DataFrame:
    print(f"Loading inference dataset: {path}")
    if sep == "\\t":
        sep = "\t"
    df = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)

    # Handle the Mutated_Seq column compatibility: if it is Mutated_Seq or Mutated_Seq (unless WT),
    # rename it uniformly to Mutated_Seq_unless_WT.
    if MUT_SEQ_COL not in df.columns:
        if MUT_SEQ_COL_RAW1 in df.columns:
            df = df.rename(columns={MUT_SEQ_COL_RAW1: MUT_SEQ_COL})
        elif MUT_SEQ_COL_RAW2 in df.columns:
            df = df.rename(columns={MUT_SEQ_COL_RAW2: MUT_SEQ_COL})
        else:
            raise KeyError(
                f"Mutation sequence column not found: neither '{MUT_SEQ_COL}', "
                f"nor '{MUT_SEQ_COL_RAW1}' or '{MUT_SEQ_COL_RAW2}' exists."
            )

    # Convert Position to int and drop missing values
    df = df.dropna(subset=REQ_COLS).copy()
    df[POSITION_COL] = df[POSITION_COL].astype(int)

    print(f"  rows after dropping NA in REQ_COLS: {len(df)}")
    return df.reset_index(drop=True)


# ===================== Main =====================

def main():
    args = parse_args()
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    df_inf = load_inference_dataset(args.input, args.sep)
    if df_inf.empty:
        raise RuntimeError("Inference dataset is empty.")

    # 1) Load ESM2 / MINT
    hf_tok, hf_model, select_layer_idx, layer_tag = load_hf_esm2()
    mint_model, mint_alphabet, mint_layers = load_mint_model()

    # 2) Build / reuse feature cache
    features_all, keys_all, cache_path = prepare_feature_cache_for_df(
        df_inf,
        hf_tok,
        hf_model,
        select_layer_idx,
        layer_tag,
        mint_model,
        mint_alphabet,
        mint_layers,
        force_rebuild=args.force_rebuild,
    )
    key_to_idx = {k: i for i, k in enumerate(keys_all)}

    # 2.5 Immediately release large models to reduce memory pressure in the later XGBoost stage
    try:
        del hf_model
        del mint_model
    except NameError:
        pass
    # If the tokenizer is no longer needed during inference, it can also be deleted here
    # del hf_tok

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    # 3) Extract features for this batch from the feature cache
    idxs: List[int] = []
    for row in df_inf.itertuples(index=False):
        key = _make_sample_key(row)
        if key not in key_to_idx:
            raise KeyError(f"[ERROR] Feature cache missing key: {key}")
        idxs.append(key_to_idx[key])

    X_inf = features_all[idxs]
    X_inf = np.ascontiguousarray(X_inf, dtype=np.float32)

    # 3.5 Drop feature cache / index after use to further save memory
    try:
        del features_all
        del keys_all
        del key_to_idx
    except NameError:
        pass
    gc.collect()

    # 4) Load best.pkl
    print(f"Loading trained XGBoost model: {args.model}")
    with open(args.model, "rb") as f:
        clf = pickle.load(f)

    # 5) Predict. If OOM still happens, this can also be changed to batched predict_proba.
    print("Running inference ...")
    proba = clf.predict_proba(X_inf)       # [N, C]
    pred = np.argmax(proba, axis=1)        # Integer classes {0,1,2,3}

    df_inf["pred_class"] = pred
    num_classes = proba.shape[1]
    for c in range(num_classes):
        df_inf[f"proba_{c}"] = proba[:, c]

    # ================= Chunked output + full output =================
    # 1) Normalize sep to avoid passing the two-character string "\t"
    sep = args.sep
    if sep == "\\t":
        sep = "\t"

    chunk_size = args.chunk_size if getattr(args, "chunk_size", 0) is not None else 0
    n_rows = len(df_inf)

    # ---- Optional chunked output: output_base_part0.ext, output_base_part1.ext, ... ----
    if chunk_size > 0 and n_rows > 0:
        base, ext = os.path.splitext(args.output)
        print(f"[INFO] Total rows: {n_rows}. Additionally writing chunk files with {chunk_size} rows per chunk...")

        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            part_idx = start // chunk_size
            out_path = f"{base}_part{part_idx}{ext or ''}"

            chunk_df = df_inf.iloc[start:end].copy()
            chunk_df.to_csv(out_path, sep=sep, index=False)

            print(f"  Wrote rows [{start}:{end}) -> {out_path}")

        print("[INFO] All chunk files have been written.")

    # ---- Write the complete summary file, consistent with previous behavior ----
    df_inf.to_csv(args.output, sep=sep, index=False)
    print(f"\n[INFO] Summary results written to: {args.output}")


if __name__ == "__main__":
    main()