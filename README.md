# When Does Multimodal Fusion Help? SLO–OCT Glaucoma Detection

Code and aggregate results for the manuscript:

> Cho, H.R.; Lee, C.S.; Kim, W.S. *When Does Multimodal Fusion Help? Cross-Domain Transfer, Joint Fusion, and Late Fusion of SLO and OCT for Glaucoma Detection under an Analysis-Locked Evaluation.* Submitted to *Diagnostics* (MDPI), 2026.

The repository contains the scripts used for every experiment in the paper, the locked evaluation protocol with SHA-256 hashes, and aggregate result files. **No images, trained weights or case-level predictions are included** (see [Data and licenses](#data-and-licenses)).

## Study overview

| Experiment | Question | Data | Main scripts |
| --- | --- | --- | --- |
| E03 | Develop multimodal models on GAMMA | GAMMA (99 cases, 5-fold × 3 seeds) | `e03a_volume_mil.py`, `e03c_train.py` |
| E04 | Do GAMMA models transfer to FairVision without retraining? | FairVision test (3000) | `e04_full_external.py` |
| E05-A | Input checks (OCT axis, intensity, preprocessing) | FairVision training/validation | `e05/e05_preflight.py`, `e05/e05_step1_oct_axis_audit.py`, `e05/e05_step2a_input_audit.py` |
| E05-B | In-domain unimodal baselines | FairVision training/validation | `e05/e05b_baselines.py` |
| E05-C | Joint fusion (concatenation, cross-attention) | FairVision training/validation | `e05/e05c_fusion.py` |
| E05-D | GAMMA vs ImageNet initialization (control) | FairVision training/validation | `e05/e05d_gamma_init.py` |
| E05 test | Analysis-locked, single evaluation of four candidates | FairVision test (3000) | `e05/e05_formal_test.py` |
| Post hoc | Subgroups, racial-disparity decomposition, multiplicity | Locked test predictions | `e05/e05_fairness.py`, `e05/e05_race_gap.py`, `analysis/*.py` |

## Repository layout

```
.
├── README.md, LICENSE, CITATION.cff, environment.yml
├── e03a_volume_mil.py, e03c_train.py        # E03 (GAMMA development)
├── e04_full_external.py                     # E04 (cross-domain transfer)
├── e05/                                     # E05 runners and post hoc analyses
├── analysis/                                # small analyses run from the command line in the study
├── results/                                 # aggregate results only (no case-level data)
│   ├── e04/                                 # E04 protocol and summary metrics
│   └── e05/                                 # E05 protocol, test results, supplementary and subgroup summaries
├── docs/checksums.sha256                    # SHA-256 of released scripts and locked checkpoints
├── tools/prepare_release.sh, tools/check_release.py
└── data/                                    # NOT included; see below
```

Scripts keep the relative paths used in the study, so run every command from the repository root.

## Environment

Experiments were run on Rocky Linux 9 with a single NVIDIA RTX 3090 (24 GB).

```bash
conda env create -f environment.yml
conda activate opticfusion
```

Core versions: Python 3.11.17, PyTorch 2.14.1 (CUDA 12.6), torchvision 0.29.1, NumPy 2.4.6, pandas 3.1.0, scikit-learn 1.9.1. The exact exported environment is in `environment.exported.yml`.

## Data and licenses

The datasets must be obtained from their providers. They are not redistributed here.

| Dataset | Source | License | Expected location |
| --- | --- | --- | --- |
| Harvard-FairVision (glaucoma) | https://ophai.hms.harvard.edu/datasets/harvard-ef30k | CC BY-NC-ND 4.0 (non-commercial research only; not for clinical use) | `data/external/fairvision/Glaucoma/Dataset/dataset.zip` and `data/external/fairvision/Glaucoma/ReadMe/data_summary_glaucoma.csv` |
| GAMMA | https://gamma.grand-challenge.org/ | As stated by the challenge | `data/gamma/` (see E03 manifests) |

Trained model weights and case-level predictions are not distributed because they are derived from data licensed under CC BY-NC-ND 4.0. Aggregate metrics in `results/` contain no case identifiers.

## Reproducing the experiments

Run from the repository root with the environment activated. GPU training is not bit-wise deterministic across hardware, so retrained models can differ slightly from the reported values.

**E03–E04 (GAMMA development and transfer).**

```bash
python -u e03c_train.py            # see script header for options
python -u e04_full_external.py --stage lock
python -u e04_full_external.py --stage prepare
python -u e04_full_external.py --stage infer --batch 4
python -u e04_full_external.py --stage evaluate
```

**E05-A (input checks).**

```bash
python -u e05/e05_preflight.py
python -u e05/e05_step1_oct_axis_audit.py
python -u e05/e05_step2a_input_audit.py --oct-axis 0
```

**E05-B/C/D (training on FairVision; Test is never read).**

```bash
python -u e05/e05b_baselines.py --stage prepare
for s in 42 43 44; do
  python -u e05/e05b_baselines.py --stage train --model oct --seed $s
  python -u e05/e05b_baselines.py --stage train --model slo --seed $s
  python -u e05/e05c_fusion.py --stage train --mode concat --init imagenet --seed $s
  python -u e05/e05c_fusion.py --stage train --mode cross  --init imagenet --seed $s
  python -u e05/e05c_fusion.py --stage train --mode concat --init unimodal --seed $s
  python -u e05/e05c_fusion.py --stage train --mode cross  --init unimodal --seed $s
  for a in oct concat cross; do python -u e05/e05d_gamma_init.py --stage train --arch $a --seed $s; done
done
python -u e05/e05c_fusion.py --stage compare
python -u e05/e05d_gamma_init.py --stage compare
python -u analysis/e05c_latefusion_diagnostic.py
python -u analysis/e05c_crossfit_weight.py
```

**E05 analysis-locked test evaluation.** The lock records candidates, calibration, thresholds, comparisons and SHA-256 hashes of the twelve checkpoints and of the evaluation script. The script refuses to run if any hash changes and allows a single evaluation.

```bash
python -u e05/e05_formal_test.py --stage lock
python -u e05/e05_formal_test.py --stage prepare
python -u e05/e05_formal_test.py --stage infer
python -u e05/e05_formal_test.py --stage evaluate
```

**Post hoc analyses.**

```bash
python -u analysis/e05_supplementary_stats.py    # Bonferroni/Holm, partial AUROC, matched-specificity sensitivity
python -u e05/e05_fairness.py
python -u e05/e05_race_gap.py
python -u analysis/e05_race_fp_md.py
python -u analysis/e05d_layer4_relative_change.py
```

## Verifying the analysis lock

`results/e05/formal_test/protocol.json` stores the SHA-256 of the evaluation script and of each checkpoint. To confirm that the released script is the one that was locked:

```bash
python tools/check_release.py --verify-lock
```

## Key results (from the paper)

| Comparison on FairVision test (n = 3000) | ΔAUROC vs OCT-only (95% CI) |
| --- | --- |
| Cross-attention fusion (primary) | −0.0045 (−0.0110 to 0.0021) |
| Concatenation fusion | +0.0009 (−0.0056 to 0.0074) |
| Late fusion, SLO weight 0.15 | +0.0059 (0.0033 to 0.0087); Holm-adjusted p = 0.0012 |

The FairVision test split was first analyzed in E04; the E05 test evaluation is an analysis-locked evaluation on a previously accessed split, not an independent external confirmation.

## License

Code: MIT License (see `LICENSE`). Datasets remain under their original licenses.

## Citation

See `CITATION.cff`. Please cite the paper once published, and the FairVision and GAMMA dataset papers.

## Contact

Woong Sik Kim (wskim@konyang.ac.kr) and Chang Soo Lee (cslee@kribb.re.kr).
