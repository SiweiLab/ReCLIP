#!/usr/bin/env python3
"""Build PTM mutation table with refseq column for PrimateAI matching."""

from __future__ import annotations

import argparse
import glob
import io
import os
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add refseq to PTM mutation rows via UniProt->Gene and MANE gene->NM mapping."
    )
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "ten_folds"),
        help="Input file/dir/glob for PTM rows. Default: data/ptm/ten_folds",
    )
    parser.add_argument(
        "--idmap-path",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "PrimateAI", "idmapping_ptm_uniprot_gene.tsv"),
        help="Cache TSV for UniProt->Gene mapping (columns: Target_UPID, Gene).",
    )
    parser.add_argument(
        "--mane-path",
        default=os.path.join(PROJECT_ROOT, "data", "four_classes_mutation", "PrimateAI", "MANE.GRCh38.v1.4.summary.txt"),
        help="MANE summary txt with symbol + RefSeq_nuc.",
    )
    parser.add_argument(
        "--output-path",
        default=os.path.join(PROJECT_ROOT, "data", "ptm", "PrimateAI", "Mutation_ptm_with_refseq.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--no-online-idmap",
        action="store_true",
        help="Disable querying UniProt REST for missing UniProt IDs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=80,
        help="Batch size for UniProt REST queries (smaller is safer for URL length).",
    )
    return parser.parse_args()


def _collect_input_files(path_or_glob: str) -> List[str]:
    if os.path.isdir(path_or_glob):
        # Prefer PTM fold test files when present.
        test_files = sorted(glob.glob(os.path.join(path_or_glob, "fold_*_test.csv")))
        if test_files:
            return test_files
        files = sorted(glob.glob(os.path.join(path_or_glob, "*.csv")))
        if not files:
            files = sorted(glob.glob(os.path.join(path_or_glob, "*.tsv")))
        if not files:
            raise FileNotFoundError(f"No CSV/TSV files found in directory: {path_or_glob}")
        return files

    hits = sorted(glob.glob(path_or_glob))
    if hits:
        return hits
    if os.path.exists(path_or_glob):
        return [path_or_glob]
    raise FileNotFoundError(f"Input path not found: {path_or_glob}")


def _read_one_table(path: str) -> pd.DataFrame:
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep, low_memory=False)


def _derive_before_after(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    seq_col = "Seq" if "Seq" in df.columns else "Target_Seq"
    mut_col = "Mutated_Seq (unless WT)" if "Mutated_Seq (unless WT)" in df.columns else "Mutated_Seq"
    seqs = df[seq_col].fillna("").astype(str).tolist()
    mut_seqs = df[mut_col].fillna("").astype(str).tolist()
    pos_list = pd.to_numeric(df["Position"], errors="coerce").fillna(-1).astype(int).tolist()

    before: list[str] = []
    after: list[str] = []
    for s, ms, pos in zip(seqs, mut_seqs, pos_list):
        idx = pos - 1
        b = s[idx] if 0 <= idx < len(s) else ""
        a = ms[idx] if 0 <= idx < len(ms) else b
        before.append(b)
        after.append(a)
    return before, after


def _normalize_ptm_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "Target_UPID" not in d.columns and "Uniprot" in d.columns:
        d["Target_UPID"] = d["Uniprot"]
    if "Interactor_UPID" not in d.columns and "Int_uniprot" in d.columns:
        d["Interactor_UPID"] = d["Int_uniprot"]

    for col in ("Target_UPID", "Interactor_UPID", "Position", "Y2H_score"):
        if col not in d.columns:
            raise ValueError(f"Missing required column '{col}' in PTM input table.")

    before, after = _derive_before_after(d)
    d["Before_AA"] = before
    d["After_AA"] = after
    d["Y2H_score"] = pd.to_numeric(d["Y2H_score"], errors="coerce").fillna(0).astype(int)
    d["Y2H_score_two_classes"] = (d["Y2H_score"] != 0).astype(int)
    if "residue_id" not in d.columns:
        d["residue_id"] = d["Target_UPID"].astype(str) + "-" + d["Position"].astype(str)
    if "ppi_id" not in d.columns:
        d["ppi_id"] = ""
    return d


def _chunks(items: Iterable[str], n: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for x in items:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def _fetch_uniprot_gene_map(accessions: list[str], batch_size: int = 80) -> Dict[str, str]:
    """Query UniProt REST and return accession -> gene_primary."""
    out: Dict[str, str] = {}
    for batch in _chunks(accessions, batch_size):
        query = " OR ".join(f"accession:{x}" for x in batch)
        url = (
            "https://rest.uniprot.org/uniprotkb/search"
            f"?query={urllib.parse.quote(query)}"
            "&fields=accession,gene_primary"
            "&format=tsv"
            f"&size={max(len(batch), 1)}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ptm-primateai-script/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        if not text.strip():
            continue
        table = pd.read_csv(io.StringIO(text), sep="\t")
        if table.empty:
            continue

        # Typical columns are 'Entry' and 'Gene Names (primary)'.
        entry_col = None
        gene_col = None
        for c in table.columns:
            lc = c.lower()
            if lc == "entry":
                entry_col = c
            if "gene names" in lc and "primary" in lc:
                gene_col = c
        if entry_col is None:
            continue
        if gene_col is None:
            # fallback: any gene column
            for c in table.columns:
                if "gene" in c.lower():
                    gene_col = c
                    break
        if gene_col is None:
            continue

        for _, row in table.iterrows():
            upid = str(row.get(entry_col, "")).strip()
            gene = str(row.get(gene_col, "")).strip()
            if upid and gene and gene.lower() != "nan":
                out[upid] = gene
        print(f"UniProt REST batch done: queried={len(batch)}, mapped_now={len(out)}")
    return out


def main() -> None:
    args = parse_args()
    input_files = _collect_input_files(args.input)
    print(f"Input files: {len(input_files)}")
    for p in input_files:
        print(f"  - {p}")

    ptm_df = pd.concat([_normalize_ptm_df(_read_one_table(p)) for p in input_files], ignore_index=True)
    ptm_df["Target_UPID"] = ptm_df["Target_UPID"].astype(str).str.strip()
    target_upids = sorted(ptm_df["Target_UPID"].dropna().astype(str).unique().tolist())
    print(f"PTM rows: {len(ptm_df)} | unique Target_UPID: {len(target_upids)}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.idmap_path), exist_ok=True)

    if os.path.exists(args.idmap_path):
        idmap_df = pd.read_csv(args.idmap_path, sep="\t", dtype=str).fillna("")
        if "Target_UPID" not in idmap_df.columns or "Gene" not in idmap_df.columns:
            raise ValueError(f"{args.idmap_path} must contain Target_UPID and Gene columns.")
        idmap_df = idmap_df[["Target_UPID", "Gene"]].drop_duplicates(subset=["Target_UPID"])
    else:
        idmap_df = pd.DataFrame(columns=["Target_UPID", "Gene"])

    known_map = dict(zip(idmap_df["Target_UPID"], idmap_df["Gene"]))
    missing = [u for u in target_upids if not known_map.get(u)]
    print(f"UniProt->Gene preloaded: {len(known_map)} | missing for PTM targets: {len(missing)}")

    if missing and not args.no_online_idmap:
        fetched = _fetch_uniprot_gene_map(missing, batch_size=max(1, args.batch_size))
        print(f"Fetched from UniProt REST: {len(fetched)}")
        if fetched:
            for k, v in fetched.items():
                known_map[k] = v
            merged_idmap = pd.DataFrame(
                {"Target_UPID": list(known_map.keys()), "Gene": list(known_map.values())}
            ).drop_duplicates(subset=["Target_UPID"])
            merged_idmap = merged_idmap.sort_values("Target_UPID").reset_index(drop=True)
            merged_idmap.to_csv(args.idmap_path, sep="\t", index=False)
            print(f"Updated idmap cache: {args.idmap_path}")

    idmap_final = pd.DataFrame(
        {"Target_UPID": list(known_map.keys()), "Gene": list(known_map.values())}
    ).drop_duplicates(subset=["Target_UPID"])

    mane_df = pd.read_csv(args.mane_path, sep="\t", dtype=str).fillna("")
    mane_df = mane_df.rename(columns={"symbol": "Gene", "RefSeq_nuc": "RefSeq_NM"})
    if "Gene" not in mane_df.columns or "RefSeq_NM" not in mane_df.columns:
        raise ValueError("MANE summary must contain columns: symbol and RefSeq_nuc.")
    mane_df = mane_df[mane_df["RefSeq_NM"].str.startswith("NM_")]
    mane_df = mane_df[["Gene", "RefSeq_NM"]].drop_duplicates(subset=["Gene"])

    out_df = ptm_df.merge(idmap_final, on="Target_UPID", how="left")
    out_df = out_df.merge(mane_df, on="Gene", how="left")
    out_df["refseq"] = out_df["RefSeq_NM"]

    total_rows = len(out_df)
    gene_hits = (out_df["Gene"].fillna("").astype(str) != "").sum()
    refseq_hits = out_df["refseq"].fillna("").astype(str).str.startswith("NM_").sum()
    print(f"Rows with mapped Gene: {gene_hits}/{total_rows}")
    print(f"Rows with mapped refseq: {refseq_hits}/{total_rows}")

    out_df.to_csv(args.output_path, index=False)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
