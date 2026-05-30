# Scripts Layout

The task-specific training and evaluation code lives under `scripts/`.

- `scripts/four_classes_mutation/`: four-class mutation pipelines and retained baselines.
- `scripts/ptm/`: PTM pipelines and retained baselines.
- `scripts/peptide/`: peptide-MHC pipelines and retained baselines.
- `scripts/ablation/`: lightweight scripts for rerunning ReCLIP ablation settings.

Task-level manuscript plotting notebooks and `notebook_data/` bundles for the
mutation, PTM, and peptide-MHC benchmarks are temporarily withheld while
manuscript revisions are in progress.

The ablation plotting notebook, plotting tables, and preview figures are also
withheld during revision; only the ablation run wrappers are kept here.

The residue-context Figure 5 analysis bundle is maintained under
`analysis/figure_plot/`.
