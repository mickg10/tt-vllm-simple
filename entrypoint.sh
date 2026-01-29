#!/bin/bash
set -eo pipefail

# Activate virtual environment if it exists
if [ -n "${PYTHON_ENV_DIR}" ] && [ -f "${PYTHON_ENV_DIR}/bin/activate" ]; then
    source "${PYTHON_ENV_DIR}/bin/activate"
elif [ -f "/opt/venv/bin/activate" ]; then
    source /opt/venv/bin/activate
fi

# Export required environment
export VLLM_TARGET_DEVICE=tt

# Device reset if requested (must run as root)
if [ "${TT_METAL_RESET_DEVICES:-0}" = "1" ]; then
    echo "Resetting Tenstorrent devices..."
    if command -v tt-smi &>/dev/null; then
        tt-smi -r 2>/dev/null || echo "WARNING: tt-smi reset failed, continuing anyway"
        sleep 5
    else
        echo "WARNING: tt-smi not found, skipping device reset"
    fi
fi

# Change to TT_METAL_HOME if set
if [ -n "${TT_METAL_HOME}" ]; then
    cd "${TT_METAL_HOME}"
fi

# Override model from environment if set
if [ -n "${HF_MODEL}" ] && ! echo "$@" | grep -q -- "--model"; then
    set -- "--model" "${HF_MODEL}" "$@"
fi

echo "Starting vLLM server with args: $@"

# Use a Python wrapper that registers TT models before starting vLLM.
# Model registration must happen in the same process as the engine.
# --disable-frontend-multiprocessing keeps everything in one process.
exec python -c "
import os, sys, runpy
from vllm import ModelRegistry

# Register TT model architectures (must be done before engine init)
ModelRegistry.register_model('TTLlamaForCausalLM', 'models.tt_transformers.tt.generator_vllm:LlamaForCausalLM')
ModelRegistry.register_model('TTMllamaForConditionalGeneration', 'models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration')
ModelRegistry.register_model('TTQwen2ForCausalLM', 'models.tt_transformers.tt.generator_vllm:QwenForCausalLM')
ModelRegistry.register_model('TTQwen3ForCausalLM', 'models.tt_transformers.tt.generator_vllm:QwenForCausalLM')
ModelRegistry.register_model('TTMistralForCausalLM', 'models.tt_transformers.tt.generator_vllm:MistralForCausalLM')
ModelRegistry.register_model('TTGemma3ForConditionalGeneration', 'models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration')
ModelRegistry.register_model('TTArceeForCausalLM', 'models.tt_transformers.tt.generator_vllm:TTArceeForCausalLM')

runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
" --host 0.0.0.0 --disable-frontend-multiprocessing "$@"
