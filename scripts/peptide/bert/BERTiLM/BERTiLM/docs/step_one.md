# Step One: Data Preparation And Vocabulary Files

This step prepares HLA/MHC or MUTINT-style interaction tables and converts
paired protein sequences into SWING k-mer encodings.

## HLA/MHC Data Preparation

HLA ligand data can be collected with `scrape_hla.py` and cleaned with
`build_hla_files.py`.

```bash
python3 scrape_hla.py --out_dir path/to/raw_hla_files
```

```bash
python3 build_hla_files.py \
  --iedb_dir path/to/iedb_data \
  --netmhc_dir path/to/netmhc_data \
  --chain_file path/to/chain_sequences.tsv \
  --out_dir path/to/clean_hla_files \
  --truncate
```

The cleaner writes merged IEDB/NetMHC tables and chain-only NetMHC tables.
IEDB input files can be obtained from the IEDB export portal and normalized with
`iedb_cleaner.py` when needed.

```bash
python3 iedb_cleaner.py path/to/iedb_file.csv --out_dir path/to/clean_iedb
```

## Vocabulary File Construction

`build_vocabulary_files.py` converts cleaned interaction tables into window
encodings, k-mer text files, and k-mer CSV files. The main parameters are:

- `--k`: number of SWING scores per k-mer.
- `--sub_size`: stride between consecutive k-mers.
- `--l`: half-window size around the queried residue; omit or set to the full
  partner sequence for full-context encodings.
- `--freq`: k-mer frequency filter.
- `--type`: input schema, either `HLA` or `MUTINT`.

```bash
python3 build_vocabulary_files.py \
  --data_dir path/to/csv_data \
  --out_dir path/to/vocabulary_output \
  --k 7 \
  --sub_size 7 \
  --l 1 \
  --freq -1 \
  --type MUTINT
```

The output directories are `window_encodings/`, `kmers_txt/`, and `kmers_csv/`.
