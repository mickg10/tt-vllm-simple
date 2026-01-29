# syntax=docker/dockerfile:1
# Prebuilt variant - uses pre-compiled tt-metal from ghcr.io
FROM ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.8.0-e95ffa5-48eba14

USER root

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy entrypoint from parent context
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8088/health || exit 1

EXPOSE 8088

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--max-model-len", "32768", "--block-size", "64", "--max-num-seqs", "32", "--port", "8088"]
