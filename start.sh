#!/bin/bash

# Ensure we are in the conda environment
# (This is handled by ENTRYPOINT in Dockerfile, but good for manual runs)

echo "Starting TRELLIS.2 Application..."

# You can change this to 'python app.py' to run the web demo directly
# or 'sleep infinity' if you want to exec into the pod manually.

if [ "$1" == "interactive" ]; then
    echo "Sleeping for interactive session..."
    sleep infinity
else
    # Check for GPU
    python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)"
    if [ $? -ne 0 ]; then
        echo "ERROR: No GPU detected! Make sure to run with '--gpus all'"
        echo "Typical command: docker run --gpus all -p 7860:7860 trellis2"
        exit 1
    fi

    echo "GPU Detected. Running Web Demo..."
    python app.py --ip 0.0.0.0 --port 7860
fi
