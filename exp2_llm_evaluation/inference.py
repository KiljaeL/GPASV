from pathlib import Path

import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reduce tqdm refresh rate so vLLM progress bars don't flood logs (vLLM has no use_tqdm interval option).
# Use a subclass so "tqdm | None" type hints in vLLM still work (replacing with a function broke deep_gemm_warmup).
class _TqdmSlow(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("mininterval", 2.0)
        kwargs.setdefault("miniters", 50)
        super().__init__(*args, **kwargs)
tqdm.tqdm = _TqdmSlow

# Absolute cache dir so vLLM worker subprocesses resolve it correctly (avoid empty lock path).
_CACHE_DIR = str(Path(__file__).resolve().parent / "model")


def load_model_and_tokenizer(model_name, config_or_dtype):
    """
    config_or_dtype: full config dict (for use_vllm_inprocess / torch_dtype) or just torch_dtype.
    If config has use_vllm_inprocess=True, load vLLM in-process and return (LLM, tokenizer).
    Otherwise load with transformers and return (model, tokenizer).
    """
    if isinstance(config_or_dtype, dict) and config_or_dtype.get("use_vllm_inprocess"):
        from vllm import LLM

        from config import MAX_MODEL_LEN

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=_CACHE_DIR)
        gpu_mem = config_or_dtype.get("gpu_memory_utilization", 0.9)
        llm_kwargs = dict(
            model=model_name,
            download_dir=_CACHE_DIR,
            gpu_memory_utilization=gpu_mem,
        )
        if MAX_MODEL_LEN is not None:
            llm_kwargs["max_model_len"] = MAX_MODEL_LEN
        llm = LLM(**llm_kwargs)
        return (llm, tokenizer)

    torch_dtype = config_or_dtype["torch_dtype"] if isinstance(config_or_dtype, dict) else config_or_dtype
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        cache_dir=_CACHE_DIR,
    )
    return model, tokenizer


def generate(model, tokenizer, prompt, config, max_new_tokens=2048):
    if isinstance(config, dict) and config.get("use_vllm_inprocess"):
        from vllm import SamplingParams

        messages = [{"role": "user", "content": prompt}]
        chat_kwargs = config.get("chat_template_kwargs", {})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        g = config.get("generate_kwargs", {})
        temperature = g.get("temperature", 0.7)
        if not g.get("do_sample", True):
            temperature = 0.0
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=g.get("top_p", 1.0),
            top_k=g.get("top_k", 0),
            min_p=g.get("min_p", 0.0),
        )
        outputs = model.generate([text], sampling_params)
        return outputs[0].outputs[0].text

    messages = [{"role": "user", "content": prompt}]
    chat_kwargs = config.get("chat_template_kwargs", {})

    if config["chat_tokenize"]:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **chat_kwargs,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": pad_id, **config["generate_kwargs"]}
    generated_ids = model.generate(**inputs, **gen_kwargs)
    output_ids = generated_ids[0][input_len:].tolist()
    out = tokenizer.decode(output_ids, skip_special_tokens=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def generate_batch(model, tokenizer, prompts, config, max_new_tokens=2048):
    """
    Generate for multiple prompts in one call. Use this with vLLM to leverage
    FP8's memory savings (larger effective batch → higher throughput).
    Returns: list of output strings, same order as prompts.
    """
    if not prompts:
        return []
    if isinstance(config, dict) and config.get("use_vllm_inprocess"):
        from vllm import SamplingParams

        chat_kwargs = config.get("chat_template_kwargs", {})
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs,
            )
            texts.append(text)
        g = config.get("generate_kwargs", {})
        temperature = g.get("temperature", 0.7)
        if not g.get("do_sample", True):
            temperature = 0.0
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=g.get("top_p", 1.0),
            top_k=g.get("top_k", 0),
            min_p=g.get("min_p", 0.0),
        )
        outputs = model.generate(texts, sampling_params)
        return [o.outputs[0].text for o in outputs]
    return [generate(model, tokenizer, p, config, max_new_tokens) for p in prompts]
