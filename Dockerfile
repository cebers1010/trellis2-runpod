# Base image with CUDA and Devel tools for compiling extensions
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Create Conda environment (Python 3.10)
RUN conda create -n trellis2 python=3.10 -y
SHELL ["conda", "run", "-n", "trellis2", "/bin/bash", "-c"]

# Install PyTorch 2.6.0 + CUDA 12.4
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install Basic Dependencies
RUN pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard \
    && pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8 \
    && pip install pillow-simd kornia timm

# Install Flash Attention
RUN pip install flash-attn==2.7.3

# Install NVDiffrast
RUN mkdir -p /tmp/extensions \
    && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
    && pip install /tmp/extensions/nvdiffrast --no-build-isolation

# Install NVDiffrec
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec \
    && pip install /tmp/extensions/nvdiffrec --no-build-isolation

# Install CuMesh
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive \
    && pip install /tmp/extensions/CuMesh --no-build-isolation

# Install FlexGEMM
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive \
    && pip install /tmp/extensions/FlexGEMM --no-build-isolation

# Set working directory
WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Install O-Voxel (from local directory copied in)
RUN pip install ./o-voxel --no-build-isolation

# Set entrypoint to use the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "trellis2"]

# Default command (keep container alive or run app)
CMD ["python", "app.py"]
