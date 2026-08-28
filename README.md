# Prostate Cancer Diffusion

A repository exploring diffusion models for prostate cancer image synthesis and analysis.

## Overview

This project focuses on applying diffusion models to generate and analyze medical images related to prostate cancer detection and diagnosis.

## Table of Contents

- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
-- [License](#license)

## Getting Started

### Requirements
Python 3.8+
GPU strongly recommended (training uses mixed precision / TF32)
torch, numpy, nibabel, scikit-image, tqdm

bash
pip install torch numpy nibabel scikit-image tqdm

## Usage
bash

# 1. Weakly mask-conditioned 2.5D training
python train_weak_cond.py --images /path/imagesTr --labels /path/labelsTr --out ./ckpts

# 2. Sample synthetic volumes from a trained 2.5D checkpoint
python sample_mask2img_25D.py --labels /path/labelsTr --ckpt ./ckpts/best.pt --out ./synthetic

Note:
export_2Dslices.py have hardcoded dataset paths (ROOT, DS, SAVE) — update these before running rather than relying on CLI args.
train_weak_cond.py + sample_mask2img_25D.py are the 2.5D weakly-conditioned pipeline.
sample_mask2img_25D.py infers model architecture from the checkpoint's state dict, so it should load checkpoints from any of the three UNet25D* variants without needing to know which was used at train time.

## Project Structure
.
├── README.md
├── best.pt       # best checkpoint
├── dockerfile       
├── train_weak_cond.py       # 2.5D weakly mask-conditioned DDPM (+ EMA, eval, early stopping)
└── sample_mask2img_25D.py
└── filter_best_samples.py # composite filter
└── histogram_matching_msd.py #histogram matching script


# Sampling from a trained 2.5D checkpoint
License

MIT
