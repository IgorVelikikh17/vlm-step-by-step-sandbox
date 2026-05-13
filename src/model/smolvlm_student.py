from __future__ import annotations


def load_smolvlm(
    model_name: str,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    eval_mode: bool = True,
):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    resolved_device = _resolve_device(device, torch)
    torch_dtype = _resolve_dtype(dtype, torch)

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        force_download=force_download,
        local_files_only=local_files_only,
    )
    model.to(resolved_device)
    if eval_mode:
        model.eval()
    else:
        model.train()
    return model, processor


def generate_smolvlm_answer(model, processor, image, prompt: str, max_new_tokens: int = 64) -> str:
    device = _model_device(model)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # SmolVLM HF examples use chat template + processor(text=..., images=...).
    # If Transformers changes the exact image chat API, this is the small place to adapt.
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat_prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(generated_tokens, skip_special_tokens=True).strip()


def _resolve_device(device: str, torch) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    if device not in ["cpu", "cuda", "mps"]:
        raise ValueError(f"Unsupported device: {device}")
    return device


def _resolve_dtype(dtype: str, torch):
    if dtype == "auto":
        return "auto"
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def _model_device(model):
    return next(model.parameters()).device
