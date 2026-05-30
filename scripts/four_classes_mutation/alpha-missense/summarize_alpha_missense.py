#!/usr/bin/env python3
"""Summarize AlphaMissense scores into a figure table.

Reads rows from an input CSV/TSV (e.g., Mutation_IMEx_IntAct_clean.tsv),
looks up AlphaMissense PDB by Target_UPID, and extracts the B-factor at
Position (CA atom). If the residue at Position does not match Before_AA,
we skip by default (use --allow-mismatch to still fill scores).
Missing/invalid entries default to score=0 and Y2H_predict=0.
"""

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def parse_pdb_ca_map(pdb_path: str) -> Dict[int, Tuple[str, float]]:
    """Return residue position -> (one-letter AA, B-factor) from CA atoms.

    Chooses chain A if present, otherwise the chain with most CA residues.
    """
    chain_maps: Dict[str, Dict[int, Tuple[str, float]]] = defaultdict(dict)
    chain_counts: Counter = Counter()

    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            resname = line[17:20].strip()
            chain = line[21].strip() or "?"
            resseq = line[22:26].strip()
            if not resseq:
                continue
            try:
                pos = int(resseq)
            except ValueError:
                continue
            try:
                bfactor = float(line[60:66])
            except ValueError:
                continue

            aa1 = AA3_TO_1.get(resname, "X")
            if pos not in chain_maps[chain]:
                chain_maps[chain][pos] = (aa1, bfactor)
                chain_counts[chain] += 1

    if not chain_maps:
        return {}

    if "A" in chain_maps:
        return chain_maps["A"]

    # choose chain with most residues
    best_chain = max(chain_counts.items(), key=lambda x: x[1])[0]
    return chain_maps[best_chain]


def load_pdb_cache(pdb_dir: str, upid: str) -> Optional[Dict[int, Tuple[str, float]]]:
    pdb_path = os.path.join(pdb_dir, f"AF-{upid}-F1-AM_v4.pdb")
    if not os.path.exists(pdb_path):
        return None
    return parse_pdb_ca_map(pdb_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize AlphaMissense scores.")
    parser.add_argument(
        "--input-csv",
        default=str(PROJECT_ROOT / "data" / "four_classes_mutation" / "Mutation_IMEx_IntAct_clean.tsv"),
        help="Input CSV/TSV (Mutation_IMEx_IntAct_clean.tsv by default)",
    )
    parser.add_argument(
        "--pdb-dir",
        default=str(PROJECT_ROOT / "data" / "four_classes_mutation" / "alpha_missense_data"),
        help="Directory with AlphaMissense PDB files",
    )
    parser.add_argument(
        "--out-csv",
        default=str(PROJECT_ROOT / "scripts" / "four_classes_mutation" / "figure_plot" / "AlphaMissense_for_figure.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--threshold", type=float, default=0.56, help="AlphaMissense score threshold")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Allow filling scores even if Before_AA mismatches PDB residue",
    )

    args = parser.parse_args()

    required_cols = [
        "Target_UPID",
        "Position",
        "Before_AA",
        "After_AA",
        "Y2H_score",
        "Interactor_UPID",
    ]

    pdb_cache: Dict[str, Optional[Dict[int, Tuple[str, float]]]] = {}
    stats = Counter()

    # detect delimiter
    with open(args.input_csv, "r", encoding="utf-8") as sniff_f:
        first_line = sniff_f.readline()
    delimiter = "\t" if "\t" in first_line else ","

    with open(args.input_csv, "r", encoding="utf-8") as in_f, open(
        args.out_csv, "w", encoding="utf-8", newline=""
    ) as out_f:
        reader = csv.DictReader(in_f, delimiter=delimiter)
        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing required column: {col}")

        out_fields = [
            "Target_UPID",
            "Position",
            "Before_AA",
            "After_AA",
            "Y2H_score",
            "Y2H_score_two_classes",
            "Interactor_UPID",
            "score_alphamissense",
            "Y2H_predict",
        ]
        writer = csv.DictWriter(out_f, fieldnames=out_fields)
        writer.writeheader()

        for row in reader:
            stats["total"] += 1
            upid = row["Target_UPID"].strip()
            pos_str = row["Position"].strip()
            before_aa = row["Before_AA"].strip()
            y2h_score = row.get("Y2H_score", "").strip()
            y2h_two = row.get("Y2H_score_two_classes", "").strip()
            if not y2h_two:
                # default: 0 if Y2H_score == 0 else 1
                try:
                    y2h_two = "0" if float(y2h_score) == 0 else "1"
                except Exception:
                    y2h_two = "0"

            score = "0"
            pred = "0"

            try:
                pos = int(pos_str)
            except ValueError:
                stats["bad_position"] += 1
                out_row = {
                    "Target_UPID": row.get("Target_UPID", ""),
                    "Position": row.get("Position", ""),
                    "Before_AA": row.get("Before_AA", ""),
                    "After_AA": row.get("After_AA", ""),
                    "Y2H_score": y2h_score,
                    "Y2H_score_two_classes": y2h_two,
                    "Interactor_UPID": row.get("Interactor_UPID", ""),
                    "score_alphamissense": score,
                    "Y2H_predict": pred,
                }
                writer.writerow(out_row)
                continue

            if upid not in pdb_cache:
                pdb_cache[upid] = load_pdb_cache(args.pdb_dir, upid)

            pdb_map = pdb_cache[upid]
            if not pdb_map:
                stats["missing_pdb"] += 1
            else:
                if pos not in pdb_map:
                    stats["missing_pos"] += 1
                else:
                    aa1, bfactor = pdb_map[pos]
                    if before_aa and aa1 != before_aa and not args.allow_mismatch:
                        stats["aa_mismatch"] += 1
                    else:
                        score = f"{bfactor:.6f}"
                        pred = "1" if bfactor > args.threshold else "0"
                        stats["scored"] += 1

            out_row = {
                "Target_UPID": row.get("Target_UPID", ""),
                "Position": row.get("Position", ""),
                "Before_AA": row.get("Before_AA", ""),
                "After_AA": row.get("After_AA", ""),
                "Y2H_score": y2h_score,
                "Y2H_score_two_classes": y2h_two,
                "Interactor_UPID": row.get("Interactor_UPID", ""),
                "score_alphamissense": score,
                "Y2H_predict": pred,
            }
            writer.writerow(out_row)

    print(f"Wrote: {args.out_csv}")
    print(
        "Stats: "
        + ", ".join(
            f"{k}={v}" for k, v in stats.items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
