#!/usr/bin/env python3
"""
Add a 'refseq' column to Mutation_IMEx_IntAct_clean.tsv via:
1) Target_UPID -> Gene (idmapping_2025_11_25.tsv)
2) Gene (symbol) -> RefSeq_NM (MANE.GRCh38.v1.4.summary.txt)

Output: Mutation_IMEx_IntAct_clean_with_refseq.csv
"""

import pandas as pd

# https://www.uniprot.org/id-mapping （ Uniprot AC/ID -> Gene Name)

# ======= file paths =======
MUTATION_PATH = "Mutation_IMEx_IntAct_clean.tsv"
IDMAP_PATH    = "idmapping_2025_11_25.tsv"
MANE_PATH     = "MANE.GRCh38.v1.4.summary.txt"
OUTPUT_PATH   = "Mutation_IMEx_IntAct_clean_with_refseq.csv"


def main():
    # 1) load mutation table (TSV!)
    mut_df = pd.read_csv(MUTATION_PATH, sep="\t")

    # 2) load Target_UPID -> Gene mapping
    #    expected columns: Target_UPID, Gene (tab-separated)
    idmap_df = pd.read_csv(IDMAP_PATH, sep="\t")

    # keep only needed cols and drop duplicate Target_UPID
    idmap_df = idmap_df[["Target_UPID", "Gene"]].drop_duplicates(subset=["Target_UPID"])

    # 3) load MANE summary
    #    expected columns include: symbol, RefSeq_nuc, ...
    mane_df = pd.read_csv(MANE_PATH, sep="\t")

    # rename to align with idmap_df
    mane_df = mane_df.rename(columns={
        "symbol": "Gene",
        "RefSeq_nuc": "RefSeq_NM"
    })

    # only keep NM_ transcripts (filter out NR_, XM_, etc.)
    mane_df = mane_df[mane_df["RefSeq_NM"].astype(str).str.startswith("NM_")]

    # one row per Gene
    mane_df = mane_df[["Gene", "RefSeq_NM"]].drop_duplicates(subset=["Gene"])

    # 4) merge Target_UPID -> Gene into mutation table
    mut_with_gene = mut_df.merge(
        idmap_df,
        on="Target_UPID",
        how="left"
    )

    # 5) merge Gene -> RefSeq_NM (MANE) into mutation table
    mut_with_nm = mut_with_gene.merge(
        mane_df,
        on="Gene",
        how="left"
    )

    # 6) add 'refseq' column
    mut_with_nm["refseq"] = mut_with_nm["RefSeq_NM"]

    # 7) stats: hit / miss counts
    total_rows = len(mut_with_nm)
    hit_rows = mut_with_nm["refseq"].notna().sum()
    miss_rows = total_rows - hit_rows

    print(f"Total rows: {total_rows}")
    print(f"Matched refseq (non-NA): {hit_rows}")
    print(f"Unmatched refseq (NA):   {miss_rows}")

    # 8) save result
    mut_with_nm.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()