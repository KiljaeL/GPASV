import os
import warnings

# Read the HuggingFace token from the environment. The Chatbot Arena dataset
# requires authentication; export HF_TOKEN before running:
#   export HF_TOKEN=hf_xxx
#
# We only warn at import time (so `--help` still works without a token); the
# actual data loaders raise at the point where the token is needed.
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    warnings.warn(
        "HF_TOKEN is not set; loading Chatbot Arena data or downloading models "
        "will fail. Export your token with: export HF_TOKEN=hf_xxx",
        stacklevel=2,
    )

CACHE_DIR = os.environ.get("HF_CACHE_DIR", "model")
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 32768

MODEL_CONFIGS = {
    "qwen3-30b": {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "torch_dtype": "auto",
        "chat_tokenize": False,
        "generate_kwargs": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
        },
        "use_vllm_inprocess": True,
        "gpu_memory_utilization": 0.9,
    },
    "qwen3-30b-fp8": {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "torch_dtype": "auto",
        "chat_tokenize": False,
        "generate_kwargs": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0,
        },
        "use_vllm_inprocess": True,
        "gpu_memory_utilization": 0.9,
    },
    "qwen3.5-35b": {
        "name": "Qwen/Qwen3.5-35B-A3B",
        "torch_dtype": "auto",
        "chat_tokenize": False,
        "generate_kwargs": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "chat_template_kwargs": {"enable_thinking": False},
        "use_vllm_inprocess": True,
        "gpu_memory_utilization": 0.9,
    },
    "qwen3.5-35b-fp8": {
        "name": "Qwen/Qwen3.5-35B-A3B-FP8",
        "torch_dtype": "auto",
        "chat_tokenize": False,
        "generate_kwargs": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "chat_template_kwargs": {"enable_thinking": False},
        "use_vllm_inprocess": True,
        "gpu_memory_utilization": 0.9,
    },
    "qwen3.5-9b": {
        "name": "Qwen/Qwen3.5-9B",
        "torch_dtype": "auto",
        "chat_tokenize": False,
        "generate_kwargs": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "chat_template_kwargs": {"enable_thinking": False},
        "use_vllm_inprocess": True,
        "gpu_memory_utilization": 0.9,
    },
    "qwen3.5-4b": {
        "name": "Qwen/Qwen3.5-4B",
        "torch_dtype": "auto",
        "chat_tokenize": False,
        "generate_kwargs": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "chat_template_kwargs": {"enable_thinking": False},
        "use_vllm_inprocess": True,
        "gpu_memory_utilization": 0.9,
    },
}
