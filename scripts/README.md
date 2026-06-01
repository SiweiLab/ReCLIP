# Scripts Layout

The task-specific training and evaluation code lives under `scripts/`.

- `scripts/four_classes_mutation/`: four-class mutation pipelines and retained baselines.
- `scripts/ptm/`: PTM pipelines and retained baselines.
- `scripts/peptide/`: peptide-MHC pipelines and retained baselines.
- `scripts/clinvar/`: ClinVar interaction perturbation inference.
- `scripts/ablation/`: lightweight scripts for rerunning ReCLIP ablation settings.

Generated logs, caches, prediction tables, and model outputs are runtime
artifacts and are ignored by Git.
