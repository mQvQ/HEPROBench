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
| `histoplexer` | paired consecutive-section generation | `methods/histoplexer/implementation/bin/train_imc01.py` or `bin.train_ddp_imc01` |
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

## Recorded formal defaults

The formal JSONs retain the key settings used by the benchmark training runs.
Batch size is per process/GPU; HistoPlexer's formal two-process launch therefore
has a global batch size of 16.

| Method | Key formal training defaults |
| --- | --- |
| ROSIE | batch 8; 128px sampled patches; 16 samples/image; 100k iterations; LR 1e-4; seed 42 |
| HEX | batch 16; 384px input; 100k iterations; stage-1 83,333; LR 1e-5; FDS enabled |
| CUT | batch 8; 100k iterations; 50k constant + 50k decay; LR 2e-4; LSGAN + PatchNCE |
| Pix2Pix | batch 16; 100k iterations; LR 2e-4; UNet-256; vanilla GAN; L1 weight 100 |
| CycleGAN | batch 16; 100k iterations; LR 2e-4; ResNet-9; LSGAN; cycle weights 10/10 |
| HistoPlexer | batch 8/process; 2 processes; 100k steps; seed 96; GP weight 5; ASP weight 1; R1 weight 1 |
| GigaTIME-Original | batch 16; 100 epochs; LR 1e-3; BCE+Dice multi-label binary objective |
| GigaTIME-Reg | batch 16; 100k steps; LR 2e-4; dense regression objective |
| MIPHEI-ViT | batch 16; 100k steps; LR 2e-4; H-Optimus-0 + ViTMatte/LoRA |
| DPT-FM | batch 16; 100k steps; LR 3e-5; frozen encoder + DPT decoder |

Dataset-specific channel counts and paths must be changed for a new cohort;
the optimization and architecture defaults above do not need to be rewritten.

## Legacy demo

The YAML files under `configs/methods/` select tiny adapters used by
`run_demo.sh`. They are retained for comparison with the first review package
and are not the formal method-specific implementations above.
