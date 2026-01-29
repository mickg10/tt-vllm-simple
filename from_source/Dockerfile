# syntax=docker/dockerfile:1
# Source build variant - compiles tt-metal and vLLM from source
FROM ubuntu:22.04 AS base

# Build args
ARG TT_METAL_REF=main
ARG VLLM_REF=dev
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=16

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build pkg-config \
    wget curl software-properties-common gnupg ca-certificates \
    python3 python3-dev python3-pip python3-venv \
    libhwloc-dev libnuma-dev libatomic1 libtbb-dev \
    libcapstone-dev xxd libssl-dev \
    # For distributed support (Galaxy/TG)
    libopenmpi-dev openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Install clang-20 (required compiler)
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-20 main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends clang-20 lld-20 \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone tt-metal (shallow clone for speed)
WORKDIR /
RUN git config --global http.postBuffer 524288000 \
    && git clone --depth 1 -b ${TT_METAL_REF} https://github.com/tenstorrent/tt-metal.git \
    && cd tt-metal \
    && git submodule update --init --recursive --depth 1

WORKDIR /tt-metal

# Install dependencies (WITH distributed support for Galaxy/TG)
RUN ./install_dependencies.sh --docker

# Build tt-metal C++ components with parallel compilation
ENV MAKEFLAGS="-j${BUILD_JOBS}"
RUN CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} ./build_metal.sh

# Create Python virtual environment and install tt-metal
ENV PYTHON_ENV_DIR=/opt/venv
RUN ./create_venv.sh --env-dir /opt/venv

# Clone vLLM
WORKDIR /
RUN git clone --depth 1 -b ${VLLM_REF} https://github.com/tenstorrent/vllm.git

# Install vLLM for TT target
WORKDIR /vllm
ENV VLLM_TARGET_DEVICE=tt
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

RUN . /opt/venv/bin/activate \
    && uv pip install -e . \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       --index-strategy unsafe-best-match

# Runtime configuration
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal

# Copy entrypoint from parent context
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Health check (longer start period for source builds)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:8088/health || exit 1

WORKDIR /vllm
EXPOSE 8088

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--max-model-len", "32768", "--block-size", "64", "--max-num-seqs", "32", "--port", "8088"]
