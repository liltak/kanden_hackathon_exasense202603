FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget build-essential cmake ninja-build \
    libboost-all-dev libcgal-dev libeigen3-dev \
    libflann-dev libfreeimage-dev libgflags-dev \
    libglew-dev libgoogle-glog-dev libmetis-dev \
    libsqlite3-dev libceres-dev \
    python3.12 python3.12-venv python3.12-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install COLMAP
RUN git clone --depth 1 https://github.com/colmap/colmap.git /tmp/colmap \
    && cd /tmp/colmap && mkdir build && cd build \
    && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES="75;86;89;90" \
    && ninja -j$(nproc) && ninja install \
    && rm -rf /tmp/colmap

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Install Python dependencies
RUN uv sync --extra dev

# GPU-specific dependencies
RUN uv pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install \
    open3d transformers accelerate bitsandbytes \
    vllm unsloth

EXPOSE 7860 8000

CMD ["uv", "run", "python", "-m", "src.ui.app"]
