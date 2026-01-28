"""
Text generation using Hugging Face transformers library on Apple Silicon.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are a helpful programming assistant."},
        {"role": "user", "content": "Write a Python function that checks if a number is prime."},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(model.device)

    print("Generating response...\n")

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    print(response)


if __name__ == "__main__":
    main()
