# ============================================================
# Sparse MoE — Docker Container
# ============================================================
# Build:
#   docker build -t sparse-moe .
#
# Run (GPU):
#   docker run --gpus all -v $(pwd)/dataset:/app/dataset sparse-moe
#
# Run (CPU fallback):
#   docker run -v $(pwd)/dataset:/app/dataset sparse-moe
#
# Generate text:
#   docker run --gpus all -v $(pwd)/outputs:/app/outputs sparse-moe \
#     python generate.py checkpoint=outputs/moe_best.pt prompt="Once upon a time"
# ============================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default: train the MoE model
# Override with: docker run sparse-moe python generate.py ...
CMD ["python", "train.py"]
