# SpecEdge

Code for **"Overlap Metrics Do Not Predict Measurement Reliability: Qualifying and
Monitoring Segmentation for SEM Metrology."**

Lithography metrology reads critical dimension (CD) and edge roughness off contours
that threshold operators extract from SEM images. At advanced nodes those operators
stop producing usable contours, and learned segmentation is the candidate replacement.
Adopting it puts a trained model inside a measurement instrument and raises a question
the deterministic route never posed: how do you establish that a segmentation model
*measures* correctly? The field answers with mask overlap on an in-distribution split.
This repository contains the experiments showing that answer does not hold, and the two
mechanisms proposed in its place.

## Verify the paper's numbers without running anything

The aggregated per-sample results are included, so every number printed in the
manuscript can be checked against the artifact that produced it:

```bash
python scripts/verify_claims.py
```

It holds 72 values as the paper prints them, each paired with an artifact field,
rounds the artifact value to the printed precision, and exits nonzero on any
disagreement. Expected output: `72/72 claims verified`.

## What the code establishes

| Claim | Where |
|---|---|
| Overlap reads one functional of the boundary displacement field, so the conditional spread of a boundary-derived measurand is unbounded at fixed overlap: 23x within one narrow IoU bin, 38x over 588 cross-validated images, frontend identity explaining 0.1% | `revision_v4_analysis.py` (E7), `e21_decoupling_cv.py` |
| That derivation tested under control: 586 masks perturbed by displacement fields of equal L1 norm and different shape hold IoU within 0.005 while CD error moves 26x, and the attenuation tracks L/A at r = 0.989 | `e23_mechanism_synthetic.py` |
| Cross-validated retraining separates the four frontends by less than each varies across folds, and the leader changes with the fold | `e18_make_cv_folds.py`, `e18_run_cv.sh` |
| Out-of-window collapse reproduces but is not confined to one architecture | same, evaluated on the Extreme split |
| Against layout intent, the frontend with the higher Extreme overlap issues 11 wrong design calls where the lower issues 6 | `e13_layout_dfm_verdict.py`, `e12_dfm_verdict_fidelity.py` |
| Frontend choice shifts the estimated process-window boundary along a foreground-ratio proxy axis | `e14_pw_boundary.py` |
| A reference-free guard detects out-of-window failures at AUROC 0.910 | `e4d_routing.py`, `e4c_loo_guard.py` |
| The guard against published uncertainty signals (max-softmax, entropy, MC-dropout, deep ensembles) | `e20_uncertainty_baselines.py` |
| The guard against the policy it must beat: deploying the fallback frontend alone | `e22_policy_baselines.py`, `e15_guard_protects_dfm.py` |
| sup-F bootstrap for the estimated breakpoint, cluster bootstrap, Holm correction | `e16_statistical_corrections.py` |
| Conditional-risk model, cross-fitting, calibration drift out of window | `e19_risk_model.py` |

## Layout

```
specedge/
  metrology.py          mask -> metrology record (CD, LWR, LER, PSD, topology)
  metrics.py            overlap metrics
  metrics_psd.py        edge-PSD metrics
  baselines/registry.py the four segmentation frontends
  data/litho_dataset.py image/mask loader
scripts/
  baselines/            training and evaluation entry points, one config per frontend
  prepare_*.py          dataset construction
  eval_metrology.py     metrology record for a directory of masks
  revision_v4_analysis.py   noise floor, breakpoint, guard, IoU bins
  e1*.py e2*.py         the experiments listed above
  verify_claims.py      checks every printed number against its artifact
  make_fig*.py replot_paper_figures.py   paper figures
output/revision_v4/     aggregated results (no images), enough to re-derive every number
```

## Setup

```bash
conda create -n specedge python=3.10 && conda activate specedge
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The frontends need `segmentation-models-pytorch` (U-Net, DeepLabV3+, and the
HRNet-W32 encoder via `timm`) and `transformers` (SegFormer).

## Training protocol

All four frontends are trained identically, so any difference is attributable to the
architecture: 512x512 inputs, BCE+Dice, AdamW at lr 1e-4 and weight decay 1e-4, batch
size 8, cosine annealing over 100 epochs with no warm-up, mixed precision, seed 42,
flip / 90-degree rotation / +-0.1 brightness-contrast augmentation. Every encoder
starts pretrained (ImageNet for the CNNs, `nvidia/mit-b2` for SegFormer). The reported
checkpoint is best validation IoU; those land at epochs 63, 84, 75 and 61, all inside
the budget. Configs are in `scripts/baselines/configs/`.

## Reproducing

The SEM images are not redistributable (see below), so the data-dependent steps need
your own acquisitions in the same layout: `images/{train,val,test}/*.png` with matching
`masks/`.

```bash
# 1. train and evaluate the four frontends
bash scripts/baselines/run_all.sh dataset/litho output/baselines

# 2. metrology records per frontend
python scripts/eval_metrology.py --gt-dir dataset/litho/masks/test \
    --pred-dir output/baselines/segformer/preds/masks \
    --output output/metrology/segformer_test_metrics.csv

# 3. main analysis: noise floor, breakpoint, guard, IoU bins
python scripts/revision_v4_analysis.py

# 4. cross-validation (about 3 GPU-hours for 4 frontends x 5 folds)
python scripts/e18_make_cv_folds.py --k 5 --seed 42 && bash scripts/e18_run_cv.sh

# 5. design-side, statistical and policy experiments (no GPU)
python scripts/e11_classical_baseline.py
python scripts/e12_dfm_verdict_fidelity.py
python scripts/e13_layout_dfm_verdict.py
python scripts/e14_pw_boundary.py
python scripts/e15_guard_protects_dfm.py
python scripts/e16_statistical_corrections.py --n-boot 2000
python scripts/e19_risk_model.py
python scripts/e21_decoupling_cv.py
python scripts/e22_policy_baselines.py
python scripts/e23_mechanism_synthetic.py    # needs reference masks

# 6. GPU-dependent probes
python scripts/e5a_inference_probe.py
python scripts/e20_uncertainty_baselines.py --mc-passes 20

# 7. figures, then check every number
python scripts/make_fig1_overview.py
python scripts/replot_paper_figures.py
python scripts/verify_claims.py
```

## Data availability

The SEM acquisitions and the reference masks derived from them are production material
and cannot be redistributed. The reference masks were produced by LithoSeg
([arXiv:2511.12005](https://arxiv.org/abs/2511.12005)), separate work by the author and
colleagues; that protocol is an input to the present study rather than part of it. What is here is the full analysis pipeline, the metrology
extractor, the qualification and guard protocols, every script that turns masks into
the reported numbers, and the aggregated per-sample results those scripts produced.
The aggregates contain measurement values and zero-padded sample indices only, no
imagery.

Applying the protocol to other data needs a directory of binary masks and a matching
reference; nothing in `specedge/metrology.py` or the guard is specific to this dataset.

## Not included

The blinded re-annotation control (Appendix A of the paper) is reproducible from
`scripts/prepare_reannotation_kit.py`, but the unblinding key for our own pass is
deliberately withheld so the control can be repeated blind.
