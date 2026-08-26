# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
# System default is Python 3.10 on Ubuntu 22.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    python3-distutils \
    git \
    wget \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*



# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel


# Install PyTorch with CUDA 12.6 support
# Note: Using PyTorch 2.5.1 as 2.7.1 doesn't exist
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Clone and install nnUNet
WORKDIR /opt
RUN git clone https://github.com/MIC-DKFZ/nnUNet.git
WORKDIR /opt/nnUNet
RUN pip install -e .

# Install optional dependencies
RUN pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git

# Install additional useful packages for medical imaging
RUN pip install \
    numpy \
    scipy \
    scikit-image \
    SimpleITK \
    nibabel \
    matplotlib \
    pandas \
    tqdm \
    tensorboard

# Set up nnUNet environment variables
# These should be configured based on your specific paths
# ENV nnUNet_raw_data_base="/data/nnUNet_raw_data_base"
# ENV nnUNet_preprocessed="/data/nnUNet_preprocessed"
# ENV RESULTS_FOLDER="/data/nnUNet_trained_models"

ENV nnUNet_raw="/data/nnUNet_raw"
ENV nnUNet_preprocessed="/data/nnUNet_preprocessed"
ENV nnUNet_results="/data/nnUNet_results"

# Create directories for nnUNet data
# RUN mkdir -p ${nnUNet_raw_data_base}/nnUNet_raw_data && \
    # mkdir -p ${nnUNet_preprocessed} && \
   # mkdir -p ${RESULTS_FOLDER}

RUN mkdir -p ${nnUNet_raw} && \
    mkdir -p ${nnUNet_preprocessed} && \
    mkdir -p ${nnUNet_results}

# Set working directory
WORKDIR /workspace

# Set the default command
CMD ["/bin/bash"]

# Labels
LABEL maintainer="Your Name"
LABEL description="Docker image for nnUNet with Python 3.12 and PyTorch 2.5.1 CUDA 12.6"
LABEL version="1.0"

# Build instructions:
# docker build -t nnunet .
#
# Run instructions:
# docker run --gpus all -it -v /path/to/your/data:/data -v /path/to/your/workspace:/workspace nnunet
# /space/slow/cug/basePart$ 
# Note: Replace /path/to/your/data and /path/to/your/workspace with your actual paths