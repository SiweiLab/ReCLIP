# Step Three: Pretraining Dataset Construction

`build_pretrain_data.py` converts encoded text files into Hugging Face dataset
artifacts for RoBERTa pretraining.

```bash
python3 build_pretrain_data.py \
  --train_dir encoded_txt \
  --tokenizer tokenizer/tokenizer_vocab16_max7619.json \
  --out_dir pretrain \
  --max_size 4500 \
  --split_prop 0.15
```

The script reads training inputs from `--train_dir`, creates a train/evaluation
split when requested, applies tokenizer-based truncation, and writes Arrow
datasets under `--out_dir`.
