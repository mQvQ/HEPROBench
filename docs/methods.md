# Methods Included In This Review Demo

This package includes lightweight runnable method adapters. They are designed
to exercise the benchmark interface, not to reproduce final paper performance.

Included method configs:

```text
configs/methods/miphei_vit.yaml
configs/methods/dpt_fm_hoptimus0.yaml
configs/methods/dpt_fm_conch.yaml
configs/methods/cut.yaml
configs/methods/hex.yaml
configs/methods/gigatime.yaml
configs/methods/pytorch_cyclegan_and_pix2pix.yaml
configs/methods/rosie.yaml
```

`miphei_vit` is kept as an independent method because it corresponds to the
H-Optimus-0 + ViTMatte-style architecture used in the paper code.

`dpt_fm` selects the pathology foundation model through:

```yaml
method:
  name: dpt_fm
  encoder_name: hoptimus0
```

Available pathology foundation model names are defined in:

```text
methods/pathology-foundation-models/registry.json
```

The supported names are:

```text
hoptimus0, h0-mini, ctranspath, conch, conchv1_5, uni, univ2,
gpfm, phikonv2, pathgen, chief, keep, virchow2, omiclip,
provgigapath
```

The CUT, HEX, GigaTIME, pytorch-CycleGAN-and-pix2pix, and ROSIE adapters share
the same HEPROBench inference/submission interface. Full external method
repositories and full-size checkpoints are intentionally not bundled in this
lightweight review package.

