#!/usr/bin/env python3
"""
Filter PrimateAI-3D txt file and keep selected columns,
then add a binary Y2H_score column:
- pathogenic -> 1
- benign     -> 0
"""

import pandas as pd

# ====== Configuration ======
# Supports .txt and .txt.gz through compression="infer".
INPUT_PATH  = "PrimateAI-3D.hg38.txt"
OUTPUT_PATH = "PrimateAI-3D_hg38_filtered.csv"

# Load only the columns needed for downstream evaluation.
USE_COLS = [
    "refseq",
    "change_position_1based",
    "ref_aa",
    "alt_aa",
    "score_PAI3D",
    "percentile_PAI3D",
    "prediction",
]

def main():
    # 1) Read the tab-delimited input file.
    df = pd.read_csv(
        INPUT_PATH,
        sep="\t",
        usecols=USE_COLS,
        compression="infer"
    )

    # 2) Derive a binary Y2H_score column from the prediction label.
    # pathogenic -> 1, benign -> 0, other labels -> NaN
    pred_to_y2h = {
        "pathogenic": 1,
        "benign": 0,
    }
    df["Y2H_score"] = df["prediction"].map(pred_to_y2h)

    # 3) Save as CSV.
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved filtered file with Y2H_score to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
