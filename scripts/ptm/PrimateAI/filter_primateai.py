#!/usr/bin/env python3
"""Filter PrimateAI-3D score table for PTM-related refseqs and save compact CSV."""

from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DEFAULT_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/illumina-ai/PrimateAI-3D-scores/resolve/main/PrimateAI-3D.hg38.txt.gz"
)

USE_COLS = [
    "refseq",
    "change_position_1based",
    "ref_aa",
    "alt_aa",
    "score_PAI3D",
    "percentile_PAI3D",
    "prediction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter PrimateAI-3D table and keep PTM-relevant rows/columns."
    )
    parser.add_argument(
        "--input-path",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "PrimateAI", "PrimateAI-3D.hg38.txt.gz"),
        help="Path to PrimateAI-3D hg38 txt/txt.gz.",
    )
    parser.add_argument(
        "--mutation-refseq-csv",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "PrimateAI", "Mutation_ptm_with_refseq.csv"),
        help="Optional PTM mutation CSV with refseq column. Used to pre-filter rows.",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "PrimateAI", "PrimateAI-3D_hg38_filtered.csv"),
        help="Output filtered CSV path.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Chunk size for streaming large PrimateAI file.",
    )
    parser.add_argument(
        "--download-url",
        default=DEFAULT_DOWNLOAD_URL,
        help="Fallback download URL if --input-path is missing.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Optional HF token for gated dataset download.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Try downloading input file when missing.",
    )
    parser.add_argument(
        "--no-download-if-missing",
        dest="download_if_missing",
        action="store_false",
        help="Do not attempt automatic download when input file is missing.",
    )
    parser.set_defaults(download_if_missing=True)
    return parser.parse_args()


def _download_file(url: str, out_path: str, hf_token: str) -> None:
    headers = {"User-Agent": "ptm-primateai-script/1.0"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token.strip()}"
    req = urllib.request.Request(url, headers=headers)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as out_f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _resolve_input_path(input_path: str) -> str:
    if os.path.exists(input_path):
        return input_path
    # fallback: if user points to .txt but only .txt.gz exists (or vice versa)
    if input_path.endswith(".txt") and os.path.exists(input_path + ".gz"):
        return input_path + ".gz"
    if input_path.endswith(".txt.gz") and os.path.exists(input_path[:-3]):
        return input_path[:-3]
    return input_path


def main() -> None:
    args = parse_args()
    input_path = _resolve_input_path(args.input_path)

    if not os.path.exists(input_path):
        if not args.download_if_missing:
            raise FileNotFoundError(
                f"PrimateAI input file not found: {input_path}. "
                "Provide --input-path or enable --download-if-missing."
            )
        print(f"Input not found, downloading from: {args.download_url}")
        try:
            _download_file(args.download_url, input_path, args.hf_token)
            print(f"Downloaded: {input_path}")
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise RuntimeError(
                    "Download blocked (401/403). PrimateAI dataset is likely gated.\n"
                    "Resolve by either:\n"
                    "1) placing PrimateAI-3D.hg38.txt(.gz) at --input-path, or\n"
                    "2) providing an authorized HF token via --hf-token / HF_TOKEN."
                ) from e
            raise

    refseq_full = set()
    refseq_trim = set()
    if args.mutation_refseq_csv and os.path.exists(args.mutation_refseq_csv):
        mut_df = pd.read_csv(args.mutation_refseq_csv, usecols=["refseq"], low_memory=False)
        mut_df["refseq"] = mut_df["refseq"].fillna("").astype(str).str.strip()
        mut_df = mut_df[mut_df["refseq"] != ""]
        refseq_full = set(mut_df["refseq"].unique().tolist())
        refseq_trim = set(mut_df["refseq"].str.split(".").str[0].unique().tolist())
        print(f"Loaded PTM refseq filters: full={len(refseq_full)}, trim={len(refseq_trim)}")
    else:
        print("No mutation refseq CSV found; filtering by refseq will be skipped.")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    total_rows = 0
    kept_rows = 0
    first_write = True
    pred_to_y2h = {"pathogenic": 1, "benign": 0}

    reader = pd.read_csv(
        input_path,
        sep="\t",
        usecols=USE_COLS,
        compression="infer",
        chunksize=max(1, args.chunksize),
        low_memory=False,
    )
    for chunk_idx, chunk in enumerate(reader):
        total_rows += len(chunk)
        chunk["refseq"] = chunk["refseq"].fillna("").astype(str).str.strip()
        chunk["refseq_trim"] = chunk["refseq"].str.split(".").str[0]

        if refseq_full or refseq_trim:
            mask = chunk["refseq"].isin(refseq_full) | chunk["refseq_trim"].isin(refseq_trim)
            chunk = chunk[mask].copy()
        if chunk.empty:
            if (chunk_idx + 1) % 20 == 0:
                print(f"Processed chunks: {chunk_idx + 1}, kept_rows={kept_rows}")
            continue

        chunk["Y2H_score"] = chunk["prediction"].map(pred_to_y2h)
        kept_rows += len(chunk)

        chunk.to_csv(args.output_path, mode="w" if first_write else "a", header=first_write, index=False)
        first_write = False

        if (chunk_idx + 1) % 20 == 0:
            print(f"Processed chunks: {chunk_idx + 1}, kept_rows={kept_rows}")

    if first_write:
        empty = pd.DataFrame(columns=USE_COLS + ["refseq_trim", "Y2H_score"])
        empty.to_csv(args.output_path, index=False)

    print(f"Total rows scanned: {total_rows}")
    print(f"Rows kept after filtering: {kept_rows}")
    print(f"Saved filtered file: {args.output_path}")


if __name__ == "__main__":
    main()
