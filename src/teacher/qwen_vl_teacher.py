from __future__ import annotations

from parsing import parse_answer_letter, parse_reasoning_text


def load_qwen_vl_teacher(
    model_name: str,
    device: str = "auto",
    dtype: str = "auto",
):
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError as error:
        raise ImportError(
            "Qwen2.5-VL support is not available in this environment. "
            "Try: pip install -U transformers accelerate qwen-vl-utils pillow. "
            "If your Transformers version still does not expose Qwen2_5_VLForConditionalGeneration, "
            "try: pip install git+https://github.com/huggingface/transformers accelerate"
        ) from error

    resolved_device = _resolve_device(device, torch)
    torch_dtype = _resolve_dtype(dtype, torch)

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.to(resolved_device)
    model.eval()
    return model, processor


def format_qwen_teacher_prompt(example: dict) -> str:
    choices = "\n".join(
        f"{chr(65 + choice_index)}. {choice}"
        for choice_index, choice in enumerate(example["choices"])
    )
    return (
        "You are solving a science multiple-choice question using the image.\n\n"
        f"Question: {example['question']}\n\n"
        "Choices:\n"
        f"{choices}\n\n"
        "Give a short reasoning. Then give the final answer.\n\n"
        "Your response must have exactly two fields:\n"
        "Reasoning: <one or two short sentences>\n"
        "Answer: <single letter A, B, C, D, or E>\n\n"
        "Do not add anything after the Answer line."
    )


def format_qwen_teacher_retry_prompt(example: dict) -> str:
    choices = "\n".join(
        f"{chr(65 + choice_index)}. {choice}"
        for choice_index, choice in enumerate(example["choices"])
    )
    return (
        "Look at the image and answer the same multiple-choice question.\n\n"
        f"Question: {example['question']}\n\n"
        "Choices:\n"
        f"{choices}\n\n"
        "Return only this format:\n"
        "Reasoning: <very short reason>\n"
        "Answer: <single letter>"
    )


def generate_qwen_teacher_output(
    model,
    processor,
    image,
    prompt: str,
    max_new_tokens: int = 256,
) -> dict:
    device = _model_device(model)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[chat_prompt], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(device)

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    raw_output = processor.decode(generated_tokens, skip_special_tokens=True).strip()
    teacher_answer = parse_answer_letter(raw_output)
    return {
        "teacher_reasoning": parse_reasoning_text(raw_output),
        "teacher_answer": teacher_answer,
        "teacher_raw_output": raw_output,
    }


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
