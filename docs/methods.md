# Benchmark Methods

The formal method registry is `methods/registry.json`. Each registered method
has its own runner and retained native implementation. The shared CLI does not
replace a method's model, optimizer, losses, or training loop.

| Registry name | Paradigm | Native training entrypoint |
| --- | --- | --- |
| `rosie` | context-aggregated regression | `methods/rosie/implementation/train_he2sp.py` |
| `hex` | context-aggregated regression with FDS | `methods/hex/implementation/hex/run_train_dist_sp_fds_paper_norm01.py` |
| `cut` | unpaired translation | `methods/cut/implementation/train.py` |
| `pix2pix` | paired conditional GAN | `methods/pytorch_cyclegan_and_pix2pix/implementation/train.py` |
| `cyclegan` | unpaired cycle-consistent GAN | `methods/pytorch_cyclegan_and_pix2pix/implementation/train.py` |
| `histoplexer` | paired consecutive-section generation | `methods/histoplexer/implementation/bin/train.py` or `bin.train_ddp` |
| `gigatime_original` | multi-label binary segmentation | `methods/gigatime/implementation/original/scripts/db_train.py` |
| `gigatime_reg` | dense regression adaptation | `methods/miphei_vit/implementation/hepro/run.py` |
| `miphei_vit` | dense regression | `methods/miphei_vit/implementation/hepro/run.py` |
| `dpt_fm` | frozen encoder + DPT dense regression | `methods/miphei_vit/implementation/hepro/run.py` |

## HistoPlexer objective

HistoPlexer is designed for prediction between non-aligned consecutive tissue
sections. Its multiscale Gaussian-pyramid loss tolerates local misalignment,
while its patch-wise contrastive objective preserves corresponding tissue
content. These objectives are part of the HistoPlexer trainer and are exposed
in `configs/experiments/histoplexer.json` as `use_gp`, `w_GP`, and `w_ASP`.
It should therefore be interpreted as a consecutive-section method, not as a
generic aligned Pix2Pix baseline.

## GigaTIME naming

`gigatime_original` retains the multi-label binary segmentation formulation and
uses predicted probabilities as virtual stains. `gigatime_reg` is the dense
regression adaptation evaluated in the existing benchmark. Results and
checkpoints must use these distinct names.

## Foundation models

All 15 pathology encoders use the same comparison protocol: frozen encoder,
DPT decoder, last four intermediate features (four stages for Swin), 256×256
benchmark input, dataset-channel-stat normalization, and marker-standard-
deviation-weighted MSE. See `methods/pfm/specs.json` and
`configs/foundation_models/`.

MUSK is not one of the 15 encoders. References to MUSK inside HEX are preserved
because it is part of that method's internal feature extraction.

## Legacy demo

The YAML files under `configs/methods/` select tiny adapters used by
`run_demo.sh`. They are retained for comparison with the first review package
and are not the formal method-specific implementations above.
