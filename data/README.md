# Data Archives

The source release stores the runnable task datasets as compressed tar archives
to keep the Git checkout compact.

Included archives:

- `ClassI_Model.tar.gz`
- `four_classes_mutation.tar.gz`
- `MixedClass_Model.tar.gz`
- `ptm.tar.gz`

Extract them from the repository root before running training or feature
generation scripts:

```bash
tar -xzf data/ClassI_Model.tar.gz
tar -xzf data/four_classes_mutation.tar.gz
tar -xzf data/MixedClass_Model.tar.gz
tar -xzf data/ptm.tar.gz
```

These archives intentionally exclude AlphaMissense, AlphaFold, PrimateAI, and
local backup outputs. Model checkpoints such as `mint/mint.ckpt` are still
downloaded separately as described in the root README.
