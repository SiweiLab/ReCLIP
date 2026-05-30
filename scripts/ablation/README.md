# ReCLIP Ablation Run Scripts

This directory keeps lightweight command wrappers for rerunning the ReCLIP
ablation settings used during development. The manuscript plotting notebook,
plotting helper code, preview figures, and plotting tables are temporarily
withheld from the public repository.

## Scripts

- `run_reclip_topk_ablation.sh`: reruns top-k settings for mutation and PTM.
- `run_reclip_layer_ablation.sh`: reruns ESM2 layer settings for mutation and PTM.

Both scripts call the task-level ReCLIP entrypoints under:

- `scripts/four_classes_mutation/ReCLIP/run_reclip_prediction_save.py`
- `scripts/ptm/ReCLIP/esm2_ptm_reclip_prediction_save.py`

## Usage

Run from the repository root:

```bash
bash scripts/ablation/run_reclip_topk_ablation.sh
bash scripts/ablation/run_reclip_layer_ablation.sh
```

To run only one task, pass `mutation` or `ptm`:

```bash
bash scripts/ablation/run_reclip_topk_ablation.sh mutation
bash scripts/ablation/run_reclip_layer_ablation.sh ptm
```

Optional environment variables:

- `PYTHON`: Python executable, defaults to `python`.
- `DEVICE`: model device passed to ReCLIP scripts, defaults to `auto`.
- `XGB_DEVICE`: PTM XGBoost device, defaults to `auto`.
- `FORCE_REBUILD=1`: rebuild feature caches.
- `RUN_ROOT`: output root, defaults to `scripts/ablation/runs/<ablation>`.

Generated logs, caches, predictions, and metrics under `scripts/ablation/runs/`
are ignored by Git.
