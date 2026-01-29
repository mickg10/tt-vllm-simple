# syntax=docker/dockerfile:1
FROM ubuntu:22.04 AS base

# Build args for version control
ARG TT_METAL_REF=main
ARG VLLM_REF=dev
ARG DEBIAN_FRONTEND=noninteractive

# System dependencies (from tt-metal's install_dependencies.sh)
RUN apt-get update && apt-get install -y \
    git build-essential cmake ninja-build pkg-config \
    wget curl software-properties-common gnupg \
    python3 python3-dev python3-pip python3-venv \
    libhwloc-dev libnuma-dev libatomic1 libtbb-dev \
    libcapstone-dev xxd libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install clang-20 (required compiler)
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y clang-20 lld-20 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone and build tt-metal (shallow clone for reliability)
WORKDIR /
RUN git config --global http.postBuffer 524288000 && \
    git clone --depth 1 -b ${TT_METAL_REF} \
    https://github.com/tenstorrent/tt-metal.git && \
    cd tt-metal && \
    git submodule update --init --recursive --depth 1

WORKDIR /tt-metal

# Install SFPI compiler (Tenstorrent-specific)
RUN ./install_dependencies.sh --docker --no-distributed

# Build tt-metal C++ components (disable distributed for simpler build)
RUN ./build_metal.sh --without-distributed

# Create Python virtual environment and install tt-metal
ENV PYTHON_ENV_DIR=/opt/venv
RUN ./create_venv.sh --env-dir /opt/venv

# Clone and install vLLM
WORKDIR /
RUN git clone -b ${VLLM_REF} https://github.com/tenstorrent/vllm.git

WORKDIR /vllm
ENV VLLM_TARGET_DEVICE=tt
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

# Install vLLM for TT (uses CPU PyTorch, no CUDA)
RUN . /opt/venv/bin/activate && \
    uv pip install -e . \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match

# Runtime configuration
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal

WORKDIR /vllm
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8088

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--max-model-len", "32768", "--block-size", "64", "--max-num-seqs", "32", "--port", "8088"]
