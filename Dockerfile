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

# Create Conda environment (Python 3.10)
RUN conda create -n trellis2 python=3.10 -y
SHELL ["conda", "run", "-n", "trellis2", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy requirements and install
# We install these FIRST to cache layers suitable mostly only for updates to code
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Entrypoint
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "trellis2"]
CMD ["./start.sh"]
