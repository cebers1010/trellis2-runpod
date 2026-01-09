# Base image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
# We keep these as they are still needed for some python packages (like opencv, etc) or general utility
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniforge3-Linux-x86_64.sh -b -p /root/miniconda3 \
    && rm -f Miniforge3-Linux-x86_64.sh 

# Set working directory
WORKDIR /app

# Copy setup script and necessary local dependencies
COPY setup.sh .
COPY o-voxel o-voxel

# Set platform for setup.sh to avoid GPU check failure during build
ENV PLATFORM=cuda

# Run setup script to create environment and install dependencies
# We use --new-env to create 'trellis2' and install pytorch, and other flags for extensions
RUN chmod +x setup.sh && \
    ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Set shell to use the new environment for subsequent steps
SHELL ["conda", "run", "-n", "trellis2", "/bin/bash", "-c"]

# Copy and run model download script to cache weights
COPY download_model.py .
RUN python download_model.py

# Copy the application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "trellis2"]
CMD ["./start.sh"]
