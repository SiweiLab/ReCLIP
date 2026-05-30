import pandas as pd

meta_path = "data/Mutation_perturbation_clean.csv"
emb_path  = "swing_bert_baseline_mutint_scv/embedding.csv"
out_path  = "data/Mutation_perturbation_clean_with_bert.csv"

meta = pd.read_csv(meta_path)
emb  = pd.read_csv(emb_path, index_col=0)

assert len(meta) == len(emb), (len(meta), len(emb))

merged = pd.concat(
    [meta.reset_index(drop=True),
     emb.reset_index(drop=True)],
    axis=1,
)

merged.to_csv(out_path, index=False)
print("Saved:", out_path, "shape:", merged.shape)
