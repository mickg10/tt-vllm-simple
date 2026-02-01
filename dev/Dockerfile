# syntax=docker/dockerfile:1
# Dev variant - uses bind-mounted sources for rapid iteration
# tt-metal and vllm are bind-mounted, not cloned during build
FROM ubuntu:22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

# System dependencies (same as from_source)
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

# Create directories for bind mounts
RUN mkdir -p /tt-metal /vllm

# Create marker to indicate this is a dev image (not pre-built tt-metal)
RUN touch /dev-image-marker

# Create Python virtual environment
ENV PYTHON_ENV_DIR=/opt/venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV=/opt/venv

# Install basic Python dependencies that won't change often
RUN pip install --upgrade pip wheel setuptools

# Runtime environment
ENV TT_METAL_HOME=/tt-metal
ENV PYTHONPATH=/tt-metal
ENV VLLM_TARGET_DEVICE=tt

# Copy entrypoints (context is parent directory)
COPY dev/dev-entrypoint.sh /usr/local/bin/dev-entrypoint.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/dev-entrypoint.sh /usr/local/bin/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -sf http://localhost:8088/health || exit 1

WORKDIR /vllm
EXPOSE 8088

ENTRYPOINT ["/usr/local/bin/dev-entrypoint.sh"]
CMD ["--max-model-len", "32768", "--block-size", "64", "--max-num-seqs", "32", "--port", "8088"]
