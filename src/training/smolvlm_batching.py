from __future__ import annotations


def build_smolvlm_training_batch(rows: list[dict], processor, device: str):
    full_texts = []
    prompt_texts = []
    images = []

    for row in rows:
        user_message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": row["prompt"]},
            ],
        }
        assistant_message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": row["target"]},
            ],
        }

        prompt_texts.append(processor.apply_chat_template([user_message], add_generation_prompt=True))
        full_texts.append(processor.apply_chat_template([user_message, assistant_message], add_generation_prompt=False))
        images.append(row["image"])

    inputs = processor(text=full_texts, images=images, padding=True, return_tensors="pt")
    prompt_inputs = processor(text=prompt_texts, images=images, padding=True, return_tensors="pt")

    labels = inputs["input_ids"].clone()
    for row_index in range(len(rows)):
        prompt_length = int(prompt_inputs["attention_mask"][row_index].sum().item())
        labels[row_index, :prompt_length] = -100

    if "attention_mask" in inputs:
        labels[inputs["attention_mask"] == 0] = -100

    inputs["labels"] = labels
    return inputs.to(device)
