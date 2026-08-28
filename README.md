# Prostate Cancer Diffusion

A research repository for **2.5D weakly mask-conditioned diffusion-based synthesis of prostate MRI**, including training, synthetic image generation, sample filtering, and histogram matching for downstream analysis.

## Overview

This project implements a **2.5D weakly mask-conditioned Denoising Diffusion Probabilistic Model (DDPM)** for generating synthetic prostate MRI images from prostate segmentation masks.

The repository includes the complete inference and training pipeline, a trained model checkpoint, and utilities for post-processing generated images. The generated synthetic images are intended for research and downstream medical image segmentation experiments.

## Table of Contents

* [Getting Started](#getting-started)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Notes](#notes)
* [License](#license)

## Getting Started

### Requirements

The project is designed to run with:

* Python 3.8+
* CUDA-enabled GPU (strongly recommended for training)
* PyTorch
* NumPy
* NiBabel
* scikit-image
* tqdm

Training uses **mixed precision and TF32** where supported by the hardware.

### Installation

The recommended environment is the provided Docker image.

```bash
docker build -t prostate-diffusion .
docker run --gpus all -it prostate-diffusion
```

For a local Python environment, install the required packages with:

```bash
pip install torch numpy nibabel scikit-image tqdm
```

> For reproducible experiments, the provided `Dockerfile` should be preferred over a manually configured Python environment.

## Usage

### 1. Train the 2.5D weakly mask-conditioned DDPM

```bash
python train_weak_cond.py \
    --images /path/imagesTr \
    --labels /path/labelsTr \
    --out ./ckpts
```

The training script includes:

* Exponential Moving Average (EMA)
* Model evaluation
* Early stopping
* Mixed-precision training

### 2. Generate synthetic prostate MRI

Use a trained checkpoint to generate synthetic images from prostate segmentation masks:

```bash
python sample_mask2img_25D.py \
    --labels /path/labelsTr \
    --ckpt ./best.pt \
    --out ./synthetic
```

The sampling script infers the model architecture from the checkpoint state dictionary and supports the available `UNet25D*` model variants without requiring the architecture to be specified separately at inference time.

### 3. Filter generated samples

Synthetic images can be evaluated using the composite filtering procedure:

```bash
python filter_best_samples.py
```

Update the input/output paths in the script as required by your dataset organisation.

### 4. Histogram matching for MSD

Histogram matching can be applied to align synthetic image intensities with the target MSD prostate MRI distribution:

```bash
python histogram_matching_msd.py
```

Update the dataset paths in the script before execution.

## Project Structure

```text
.
├── README.md
├── Dockerfile
├── best.pt
│
├── train_weak_cond.py
├── sample_mask2img_25D.py
├── filter_best_samples.py
└── histogram_matching_msd.py
```

### File Descriptions

| File                        | Description                                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------------------- |
| `README.md`                 | Project documentation, setup instructions, usage, and reproducibility information.                    |
| `Dockerfile`                | Defines the containerised environment and dependencies required to run the project.                   |
| `best.pt`                   | Trained DDPM checkpoint used for synthetic prostate MRI generation.                                   |
| `train_weak_cond.py`        | Trains the **2.5D weakly mask-conditioned DDPM**, including EMA, evaluation, and early stopping.      |
| `sample_mask2img_25D.py`    | Generates synthetic prostate MRI images from segmentation masks using a trained 2.5D DDPM checkpoint. |
| `filter_best_samples.py`    | Applies the composite image-quality filtering procedure to generated samples.                         |
| `histogram_matching_msd.py` | Performs histogram matching to align generated image intensities with the MSD prostate MRI cohort.    |

## Notes

* The repository does **not** contain the original medical imaging datasets. The required datasets must be obtained separately and prepared according to the expected directory structure.
* The paths used by the post-processing scripts may need to be updated before execution.
* `train_weak_cond.py` and `sample_mask2img_25D.py` form the main **2.5D weakly mask-conditioned diffusion pipeline**.
* The supplied `best.pt` checkpoint is intended to be used with the code in this repository.
* GPU acceleration is strongly recommended for model training and sampling.

## License

This project is released under the **MIT License**.
