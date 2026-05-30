# Step Two: Tokenizer Training

SWING RoBERTa uses a byte-pair encoding tokenizer trained on the encoded k-mer
text files from step one. Tokenizer training is handled by
`train_tokenizer.py`.

```bash
python3 train_tokenizer.py \
  --data_dir encoded_txt \
  --out_dir tokenizer \
  --vocab_size 16 \
  --max_size 1 \
  --tokenizer by-digit
```

`--max_size` controls sequence-length truncation. A value of `1` uses the
longest encoded training sequence as the maximum length; `2` uses the second
longest sequence, and so on. The by-digit tokenizer uses a small vocabulary
containing special tokens and digits.
