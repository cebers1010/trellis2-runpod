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
    echo "Running Web Demo..."
    python app.py --ip 0.0.0.0 --port 7860
fi
