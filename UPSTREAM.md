# Upstream Source and License Provenance

The top-level MIT license covers HEPROBench's own framework code. Retained
third-party or adapted source remains under its upstream license; the license
files shipped inside each implementation directory take precedence.

| Component | Upstream snapshot | License status in this repository |
| --- | --- | --- |
| ROSIE | `enable-medicine-public/rosie`, `0477e5f245a1416d8467377f04ce74df6e5aa192` | CC BY-NC 4.0 file retained |
| HEX | `lilab-stanford/HEX`, `83a8264883637ec894e08cb1e80b218db2b297ea` | no license file was present in the local upstream snapshot; redistribution permission must be confirmed |
| CUT | `taesungp/contrastive-unpaired-translation`, `b3ac297708dfb6f7589d04662277e53c0d579c27` | BSD-style file retained |
| Pix2Pix/CycleGAN | `junyanz/pytorch-CycleGAN-and-pix2pix`, `c3268edd50ec37a81600c9b981841f48929671b8` | BSD-style file retained |
| GigaTIME Original | `prov-gigatime/GigaTIME`, `9240b5bac9114fc7dbc849cad741563e4d1b1a43` | upstream custom license retained; its source-distribution restriction must be resolved before pushing this snapshot publicly |
| HistoPlexer | `sonialagunac/HistoPlexer` | no VCS revision or license was present in the local source snapshot; both must be confirmed before public redistribution |
| HEPRO/MIPHEI-derived pipeline | local HEPRO implementation derived from Sanofi MIPHEI-ViT | academic/non-commercial license file retained |

Large pretrained weights are not copied into Git. Foundation-model provider,
gating, local path, revision, and SHA-256 status are recorded in
`methods/pfm/specs.json`. Entries with null revision or checksum are explicit
provenance TODOs, not claims that an unpinned artifact is reproducible.
