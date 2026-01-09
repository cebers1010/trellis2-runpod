# ──────────────────────────────────────────────────────────────────────────────
# Use official RunPod PyTorch image with CUDA 12.4 — much faster pull on RunPod
# Prefer runtime variant for inference/serverless unless you really need nvcc
# ──────────────────────────────────────────────────────────────────────────────
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-runtime-ubuntu22.04
# Alternative (if you need full devel tools like nvcc): 
# FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Early log to confirm build start
RUN echo "Build process started..."

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install minimal system dependencies still needed by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────────────────────────────────────
# Create conda env (optional — you can skip and use system python 3.11)
# Keeping it since your original setup uses conda + python 3.10
# ──────────────────────────────────────────────────────────────────────────────
RUN conda create -n trellis2 python=3.10 -y

# Switch shell to use the new conda env
SHELL ["conda", "run", "-n", "trellis2", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy and install requirements first → best layer caching
COPY requirements.txt .

# Install PyTorch from the pytorch index (overrides the one in base image if needed)
# Note: base image has torch 2.4.0 — we upgrade to 2.6.0 here
RUN echo "Installing PyTorch 2.6.0 and core dependencies..." && \
    pip install --no-cache-dir -v \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 && \
    echo "Installing remaining requirements..." && \
    pip install --no-cache-dir -v -r requirements.txt

# Copy and run model download script (caches weights in image)
COPY download_model.py .
RUN python download_model.py

# Copy the rest of the application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Entrypoint — runs everything inside the conda env
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "trellis2"]
CMD ["./start.sh"]