#!/bin/bash
set -eo pipefail

# Activate virtual environment if it exists
if [ -n "${PYTHON_ENV_DIR}" ] && [ -f "${PYTHON_ENV_DIR}/bin/activate" ]; then
    source "${PYTHON_ENV_DIR}/bin/activate"
elif [ -f "/opt/venv/bin/activate" ]; then
    source /opt/venv/bin/activate
fi

# ttnn reads TTNN_CONFIG_OVERRIDES as JSON at import time. If the variable is
# present but empty, json.loads("") will raise, breaking vLLM's TT platform
# detection (which imports ttnn).
if [ "${TTNN_CONFIG_OVERRIDES+x}" = "x" ] && [ -z "${TTNN_CONFIG_OVERRIDES}" ]; then
    unset TTNN_CONFIG_OVERRIDES
fi

# Export required environment
export VLLM_TARGET_DEVICE=tt

# TT Device check and auto-reset
# Check if devices are responsive; if not, reset them (up to 4 attempts)
check_and_reset_devices() {
    if ! command -v tt-smi &>/dev/null; then
        echo "WARNING: tt-smi not found, skipping device check"
        return 0
    fi

    echo "Checking TT device status..."

    # Try a quick tt-smi query with timeout - if it hangs, devices need reset
    if timeout 15 tt-smi -ls >/dev/null 2>&1; then
        echo "TT devices responding normally."
        return 0
    fi

    echo "TT devices unresponsive, attempting reset..."

    local max_attempts=4
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        echo "Reset attempt $attempt of $max_attempts..."

        if tt-smi -r 2>/dev/null; then
            echo "Waiting for devices to stabilize..."
            sleep 10

            # Verify devices are now responsive
            if timeout 15 tt-smi -ls >/dev/null 2>&1; then
                echo "TT devices recovered after reset."
                return 0
            fi
            echo "Devices still unresponsive after reset."
        else
            echo "WARNING: tt-smi reset command failed."
        fi

        attempt=$((attempt + 1))
        [ $attempt -le $max_attempts ] && sleep 5
    done

    echo "ERROR: Failed to reset TT devices after $max_attempts attempts."
    echo "Please check hardware connections and try 'tt-smi -r' manually."
    exit 1
}

# Auto-reset if enabled (default: enabled)
if [ "${TT_AUTO_RESET:-1}" != "0" ]; then
    check_and_reset_devices
fi

# Manual reset if explicitly requested
if [ "${TT_METAL_RESET_DEVICES:-0}" = "1" ]; then
    echo "Forcing TT device reset (TT_METAL_RESET_DEVICES=1)..."
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

# For GLM-4.7 models, disable thinking by default at the chat-template layer.
# This keeps user-facing outputs concise and avoids long hidden-reasoning runs,
# while still allowing explicit opt-in via request chat_template_kwargs.
if [ "${GLM_DEFAULT_DISABLE_THINKING:-0}" = "1" ] \
    && echo "${HF_MODEL:-}" | grep -qE "^zai-org/GLM-4\.7" \
    && ! echo " $* " | grep -q -- " --chat-template "; then
    export GLM_CHAT_TEMPLATE_FILE="${GLM_CHAT_TEMPLATE_FILE:-/tmp/glm47_default_no_think.jinja}"

    if python - <<'PY' >/dev/null 2>&1
import os
from transformers import AutoTokenizer

model_id = os.environ.get("HF_MODEL", "")
output_path = os.environ.get("GLM_CHAT_TEMPLATE_FILE", "/tmp/glm47_default_no_think.jinja")

pattern = "enable_thinking is defined and not enable_thinking"
replacement = "enable_thinking is not defined or not enable_thinking"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
chat_template = getattr(tokenizer, "chat_template", None)
if not chat_template:
    raise RuntimeError("Tokenizer did not provide a chat_template.")

patched_template = chat_template.replace(pattern, replacement)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(patched_template)
PY
    then
        echo "Using GLM chat template with thinking disabled by default: ${GLM_CHAT_TEMPLATE_FILE}"
        set -- "--chat-template" "${GLM_CHAT_TEMPLATE_FILE}" "$@"
    else
        echo "WARNING: Could not generate GLM no-thinking chat template; continuing with model default."
    fi
fi

echo "Starting vLLM server with args: $@"

# Use a Python wrapper that registers TT models before starting vLLM.
# Model registration must happen in the same process as the engine.
# --disable-frontend-multiprocessing keeps everything in one process.
exec python -c "
import os, sys, runpy

# TT-NN uses loguru throughout; default loguru level is DEBUG and can be very
# noisy during weight/cache loading. Configure it early to avoid I/O overhead.
try:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=os.environ.get('LOGURU_LEVEL', 'INFO'))
except Exception:
    pass

from vllm import ModelRegistry

# Register TT model architectures (must be done before engine init)
ModelRegistry.register_model('TTLlamaForCausalLM', 'models.tt_transformers.tt.generator_vllm:LlamaForCausalLM')
ModelRegistry.register_model('TTMllamaForConditionalGeneration', 'models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration')
ModelRegistry.register_model('TTQwen2ForCausalLM', 'models.tt_transformers.tt.generator_vllm:QwenForCausalLM')
ModelRegistry.register_model('TTQwen3ForCausalLM', 'models.tt_transformers.tt.generator_vllm:QwenForCausalLM')
ModelRegistry.register_model('TTMistralForCausalLM', 'models.tt_transformers.tt.generator_vllm:MistralForCausalLM')
ModelRegistry.register_model('TTGemma3ForConditionalGeneration', 'models.tt_transformers.tt.generator_vllm:Gemma3ForConditionalGeneration')
ModelRegistry.register_model('TTDeepseekV3ForCausalLM', 'models.demos.deepseek_v3.tt.generator_vllm:DeepseekV3ForCausalLM')
ModelRegistry.register_model('TTGlm4MoeLiteForCausalLM', 'models.demos.glm4_moe_lite.tt.generator_vllm:Glm4MoeLiteForCausalLM')
ModelRegistry.register_model('TTArceeForCausalLM', 'models.tt_transformers.tt.generator_vllm:TTArceeForCausalLM')

runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
" --host 0.0.0.0 "$@"
