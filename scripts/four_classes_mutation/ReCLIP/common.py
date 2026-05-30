#!/usr/bin/env python3
from __future__ import annotations

import os
import random
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, EsmModel

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:  # optional GPU memory pool cleanup
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None


FILE_PATH = Path(__file__).resolve()
WORK_DIR = FILE_PATH.parent
PROJECT_ROOT = FILE_PATH.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mint.mint.data import Alphabet
from mint.mint.helpers.extract import load_config
from mint.mint.model.esm2 import ESM2 as MintESM2


RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
K_SEQ = 5
MAX_HF_TOKENS = 1022
MAX_MINT_TOKENS = 1024
MAX_MUT_RESIDUES = (MAX_MINT_TOKENS // 2) - 2
MAX_TOTAL_RESIDUES = MAX_MINT_TOKENS - 4

DATASET_PATH = PROJECT_ROOT / "data" / "four_classes_mutation" / "Mutation_IMEx_IntAct_clean.tsv"
NUM_FOLDS = 10
MINT_CFG_PATH = PROJECT_ROOT / "mint" / "data" / "esm2_t33_650M_UR50D.json"
MINT_CHECKPOINT_PATH = PROJECT_ROOT / "mint" / "mint.ckpt"

TARGET_ID_COL = "Target_UPID"
MUT_SEQ_COL_RAW = "Mutated_Seq (unless WT)"
MUT_SEQ_COL = "Mutated_Seq_unless_WT"
INTERACTOR_ID_COL = "Interactor_UPID"
INTERACTOR_SEQ_COL = "Interactor_Seq"
POSITION_COL = "Position"
LABEL_COL = "Y2H_score"
MUTATION_COL = "Mutation"
REQ_COLS = [POSITION_COL, MUT_SEQ_COL, INTERACTOR_SEQ_COL]


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def ensure_output_layout() -> None:
    for name in ["results", "logs", "cache"]:
        (WORK_DIR / name).mkdir(parents=True, exist_ok=True)


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


def hash_key(*parts: object) -> str:
    payload = "||".join(str(part) for part in parts).encode("utf-8")
    return __import__("hashlib").blake2b(payload, digest_size=16).hexdigest()


def _safe_load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".npy", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as f:
            np.save(f, np.asarray(arr, dtype=np.float32), allow_pickle=False)
        os.replace(str(tmp_path), str(path))
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


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


def mutation_cache_root() -> Path:
    root = WORK_DIR / "cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _prep_for_windows(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.dropna(subset=REQ_COLS).copy()
    df2[POSITION_COL] = df2[POSITION_COL].astype(int)
    return df2


def load_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    path = DATASET_PATH if dataset_path is None else Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    print(f"Loading dataset: {path}")
    df_raw = pd.read_csv(path, sep="\t")
    if MUT_SEQ_COL_RAW in df_raw.columns:
        df_raw = df_raw.rename(columns={MUT_SEQ_COL_RAW: MUT_SEQ_COL})
    elif "Mutated_Seq" in df_raw.columns:
        df_raw = df_raw.rename(columns={"Mutated_Seq": MUT_SEQ_COL})
    df_clean = _prep_for_windows(df_raw)
    print(f"  rows: {len(df_raw)} -> {len(df_clean)} | dropped: {len(df_raw) - len(df_clean)}")
    return df_clean.reset_index(drop=True)


def create_tenfold_splits(df: pd.DataFrame) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    fold_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, df[LABEL_COL])):
        fold_data[fold_idx] = (train_idx, test_idx)
    return fold_data


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
    layer_idx = _norm_single_layer(int(select_layer), hf_model.config.num_hidden_layers)
    layer_tag = f"L{layer_idx}"
    return hf_tok, hf_model, layer_idx, layer_tag


def load_mint_model() -> Tuple[MintESM2, Alphabet, int]:
    if not MINT_CFG_PATH.exists():
        raise FileNotFoundError(f"MINT config not found: {MINT_CFG_PATH}")
    if not MINT_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"MINT checkpoint not found: {MINT_CHECKPOINT_PATH}")

    print("Loading MINT model (multimer ESM2)...")
    cfg = load_config(str(MINT_CFG_PATH))
    model = MintESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        token_dropout=cfg.token_dropout,
        use_multimer=True,
    )
    try:
        checkpoint = torch.load(str(MINT_CHECKPOINT_PATH), map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(MINT_CHECKPOINT_PATH), map_location=DEVICE)
    state_dict = OrderedDict((key.replace("model.", ""), value) for key, value in checkpoint["state_dict"].items())
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] MINT state dict load: missing={missing}, unexpected={unexpected}")
    model.eval().to(DEVICE)
    alphabet = Alphabet.from_architecture("ESM-1b")
    return model, alphabet, int(cfg.encoder_layers)


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


def truncate_pair_for_models(mut_seq: str, partner_seq: str, mut_pos_1based: int) -> Tuple[str, str, int]:
    if mut_pos_1based < 1 or mut_pos_1based > len(mut_seq):
        raise ValueError(f"Mutated position {mut_pos_1based} out of bounds for length {len(mut_seq)}.")
    mut_center_idx = mut_pos_1based - 1
    mut_residues = min(len(mut_seq), MAX_MUT_RESIDUES)
    mut_window, new_center = _extract_center_window(mut_seq, mut_center_idx, mut_residues)
    available_for_partner = max(1, MAX_TOTAL_RESIDUES - len(mut_window))
    partner_window = partner_seq[:available_for_partner]
    return mut_window, partner_window, new_center + 1


def _encode_chain(seq: str, alphabet: Alphabet) -> torch.Tensor:
    tokens = alphabet.encode("<cls>" + seq.replace("J", "L") + "<eos>")
    return torch.tensor(tokens, dtype=torch.long)


def _make_sample_key(row) -> str:
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


def layer_head_value_context_and_attention(
    hf_model,
    hidden_in: torch.Tensor,
    extended_attention_mask: torch.Tensor,
    layer_index: int,
    use_attention_layernorm: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    layer_module = hf_model.encoder.layer[layer_index]
    attn_module = layer_module.attention
    self_attn = attn_module.self

    attn_input = attn_module.LayerNorm(hidden_in) if use_attention_layernorm else hidden_in
    mixed_query_layer = self_attn.query(attn_input)
    key_layer = self_attn.transpose_for_scores(self_attn.key(attn_input))
    value_layer = self_attn.transpose_for_scores(self_attn.value(attn_input))
    query_layer = self_attn.transpose_for_scores(mixed_query_layer)
    query_layer = query_layer * self_attn.attention_head_size**-0.5

    if self_attn.position_embedding_type == "rotary":
        query_layer, key_layer = self_attn.rotary_embeddings(query_layer, key_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores + extended_attention_mask
    attention_probs = torch.softmax(attention_scores, dim=-1)
    context = torch.matmul(attention_probs, value_layer)
    return value_layer, context, attention_probs


@torch.no_grad()
def compute_head_residue_features_for_sequence(
    seq: str,
    select_layer_idx: int,
    hf_tok,
    hf_model,
    feature_space: str = "context",
    use_attention_layernorm: bool = False,
) -> np.ndarray:
    if not seq:
        num_heads = int(hf_model.encoder.layer[select_layer_idx].attention.self.num_attention_heads)
        head_dim = int(hf_model.encoder.layer[select_layer_idx].attention.self.attention_head_size)
        return np.zeros((num_heads, 0, head_dim), dtype=np.float32)

    encoded = hf_tok(seq[: MAX_HF_TOKENS - 2], return_tensors="pt", add_special_tokens=True)
    device = next(hf_model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = hf_model(**encoded, output_hidden_states=True, return_dict=True)
    hidden_in = outputs.hidden_states[select_layer_idx]
    ext_mask = hf_model.get_extended_attention_mask(encoded["attention_mask"], encoded["input_ids"].shape, device)
    value_layer, context, _ = layer_head_value_context_and_attention(
        hf_model,
        hidden_in,
        ext_mask,
        select_layer_idx,
        use_attention_layernorm=use_attention_layernorm,
    )

    seq_len = int(encoded["input_ids"].shape[1] - 2)
    aa_slice = slice(1, 1 + seq_len)
    if feature_space == "context":
        residue_features = context[0, :, aa_slice, :]
    elif feature_space == "restricted_v":
        residue_features = value_layer[0, :, aa_slice, :]
    else:
        raise ValueError(f"Unsupported feature_space: {feature_space}")
    return residue_features.detach().cpu().to(torch.float32).numpy()


@torch.no_grad()
def compute_head_attention_row(
    seq: str,
    mut_pos_1based: int,
    select_layer_idx: int,
    hf_tok,
    hf_model,
    use_attention_layernorm: bool = False,
) -> np.ndarray:
    if not seq:
        num_heads = int(hf_model.encoder.layer[select_layer_idx].attention.self.num_attention_heads)
        return np.zeros((num_heads, 0), dtype=np.float32)

    encoded = hf_tok(seq[: MAX_HF_TOKENS - 2], return_tensors="pt", add_special_tokens=True)
    device = next(hf_model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = hf_model(**encoded, output_hidden_states=True, return_dict=True)
    hidden_in = outputs.hidden_states[select_layer_idx]
    ext_mask = hf_model.get_extended_attention_mask(encoded["attention_mask"], encoded["input_ids"].shape, device)
    _, _, attention_probs = layer_head_value_context_and_attention(
        hf_model,
        hidden_in,
        ext_mask,
        select_layer_idx,
        use_attention_layernorm=use_attention_layernorm,
    )

    seq_len = int(encoded["input_ids"].shape[1] - 2)
    if not (1 <= mut_pos_1based <= seq_len):
        raise ValueError(f"Mutation position {mut_pos_1based} is outside trimmed sequence length {seq_len}.")
    query_index = mut_pos_1based
    attn_row = attention_probs[0, :, query_index, 1 : 1 + seq_len]
    return attn_row.detach().cpu().to(torch.float32).numpy()


def load_or_compute_target_residue_features(
    seq: str,
    select_layer_idx: int,
    hf_tok,
    hf_model,
    feature_space: str,
    use_attention_layernorm: bool,
) -> np.ndarray:
    cache_dir = mutation_cache_root() / "target_residue_features"
    cache_key = hash_key("target", seq, select_layer_idx, feature_space, int(use_attention_layernorm))
    cache_path = cache_dir / f"{cache_key}.npy"
    cached = _safe_load_npy(cache_path)
    if cached is not None:
        return cached
    arr = compute_head_residue_features_for_sequence(
        seq,
        select_layer_idx,
        hf_tok,
        hf_model,
        feature_space=feature_space,
        use_attention_layernorm=use_attention_layernorm,
    )
    _atomic_save_npy(cache_path, arr)
    return arr


def load_or_compute_attention_row(
    seq: str,
    mut_pos_1based: int,
    select_layer_idx: int,
    hf_tok,
    hf_model,
    use_attention_layernorm: bool,
) -> np.ndarray:
    cache_dir = mutation_cache_root() / "attention_rows"
    cache_key = hash_key("attnrow", seq, mut_pos_1based, select_layer_idx, int(use_attention_layernorm))
    cache_path = cache_dir / f"{cache_key}.npy"
    cached = _safe_load_npy(cache_path)
    if cached is not None:
        return cached
    arr = compute_head_attention_row(
        seq,
        mut_pos_1based,
        select_layer_idx,
        hf_tok,
        hf_model,
        use_attention_layernorm=use_attention_layernorm,
    )
    _atomic_save_npy(cache_path, arr)
    return arr


def mint_partner_representations_and_weights(
    mut_seq: str,
    partner_seq: str,
    mut_pos_1based: int,
    mint_model,
    alphabet,
    repr_layer_idx: int,
    attn_layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    mut_tokens = _encode_chain(mut_seq, alphabet)
    partner_tokens = _encode_chain(partner_seq, alphabet)

    total_tokens = mut_tokens.size(0) + partner_tokens.size(0)
    if total_tokens > MAX_MINT_TOKENS:
        raise ValueError(
            f"MINT token budget exceeded: mutated={mut_tokens.size(0)}, partner={partner_tokens.size(0)}, total={total_tokens}."
        )

    mut_chain_ids = torch.zeros_like(mut_tokens, dtype=torch.int32)
    partner_chain_ids = torch.ones_like(partner_tokens, dtype=torch.int32)
    tokens = torch.cat([mut_tokens, partner_tokens], dim=0).unsqueeze(0).to(DEVICE)
    chain_ids = torch.cat([mut_chain_ids, partner_chain_ids], dim=0).unsqueeze(0).to(DEVICE)

    repr_layers = [int(repr_layer_idx) + 1]
    with torch.no_grad():
        out = mint_model(tokens, chain_ids, repr_layers=repr_layers, need_head_weights=True)
    attentions = out["attentions"]
    attn_layer = attentions[0, int(attn_layer_idx)]

    mut_token_index = int(mut_pos_1based)
    mut_token_count = mut_tokens.size(0)
    partner_token_count = partner_tokens.size(0)
    partner_start = mut_token_count
    partner_indices = torch.arange(
        partner_start + 1,
        partner_start + partner_token_count - 1,
        device=attn_layer.device,
        dtype=torch.long,
    )
    if partner_indices.numel() == 0:
        hidden_dim = int(getattr(mint_model, "embed_dim", 1280))
        return np.zeros((0, hidden_dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # MINT returns post-softmax attention probabilities. After slicing out only
    # partner tokens, preserve their relative probabilities with L1 normalization
    # instead of applying a second softmax.
    att_head_avg = attn_layer[:, mut_token_index, partner_indices].mean(dim=0)
    weight_total = torch.sum(att_head_avg)
    if torch.isfinite(weight_total) and float(weight_total.detach().cpu()) > 0.0:
        weights_tensor = att_head_avg / weight_total
    else:
        weights_tensor = torch.full_like(att_head_avg, 1.0 / float(att_head_avg.numel()))
    weights = weights_tensor.detach().cpu().to(torch.float32).numpy()

    repr_key = int(repr_layer_idx) + 1
    if repr_key not in out["representations"]:
        raise KeyError(f"MINT representation layer {repr_key} missing from forward output.")
    hidden = out["representations"][repr_key][0]
    partner_hidden = hidden[partner_indices, :].detach().cpu().to(torch.float32).numpy()
    return partner_hidden, weights


def load_or_compute_mint_partner_representations_and_weights(
    target_seq_trim: str,
    partner_seq_trim: str,
    target_pos_trim: int,
    mint_model,
    mint_alphabet,
    partner_mint_layer_idx: int,
    mint_attn_layer_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_dir = mutation_cache_root() / "mint_partner_repr"
    cache_key = hash_key(
        "mintrepr_l1_partner_attn_v2",
        target_seq_trim,
        partner_seq_trim,
        target_pos_trim,
        partner_mint_layer_idx,
        mint_attn_layer_idx,
    )
    repr_path = cache_dir / f"{cache_key}_repr.npy"
    weight_path = cache_dir / f"{cache_key}_weights.npy"
    cached_repr = _safe_load_npy(repr_path)
    cached_weights = _safe_load_npy(weight_path)
    if cached_repr is not None and cached_weights is not None:
        return cached_repr, cached_weights
    partner_repr, mint_weights = mint_partner_representations_and_weights(
        mut_seq=target_seq_trim,
        partner_seq=partner_seq_trim,
        mut_pos_1based=target_pos_trim,
        mint_model=mint_model,
        alphabet=mint_alphabet,
        repr_layer_idx=partner_mint_layer_idx,
        attn_layer_idx=mint_attn_layer_idx,
    )
    _atomic_save_npy(repr_path, partner_repr)
    _atomic_save_npy(weight_path, mint_weights)
    return partner_repr, mint_weights


def load_feature_matrix_cache(path: Path) -> Tuple[np.ndarray, List[str]] | None:
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


def save_feature_matrix_cache(path: Path, features: np.ndarray, keys: List[str]) -> None:
    _atomic_save_npz(path, features=np.asarray(features, dtype=np.float32), keys=np.asarray(keys, dtype=object))


def compute_13_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int = 4,
) -> Dict[str, float]:
    acc = metrics.accuracy_score(y_true, y_pred)
    prec_macro = metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)

    try:
        y_bin = np.eye(num_classes, dtype=int)[y_true]
        roc_macro = metrics.roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")
    except Exception:
        roc_macro = float("nan")
    try:
        y_bin = np.eye(num_classes, dtype=int)[y_true]
        pr_macro = metrics.average_precision_score(y_bin, y_proba, average="macro")
    except Exception:
        pr_macro = float("nan")

    js = []
    labels = list(range(num_classes))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    for c in labels:
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - (tp + fn + fp)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        js.append(tpr + tnr - 1.0)
    youden_macro = float(np.mean(js)) if js else float("nan")

    try:
        from scipy.stats import combine_pvalues, ttest_ind

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
        p_fisher = float(combine_pvalues(p_vals, method="fisher")[1]) if p_vals else float("nan")
    except Exception:
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
        arr = np.asarray(vals, dtype=np.float64)
        agg[f"{key}_mean"] = float(np.nanmean(arr))
        agg[f"{key}_std"] = float(np.nanstd(arr))
    return agg
