"""
Text generation using transformer components.

Instead of calling model.generate(), this script:
- Separates the model into backbone (Qwen2Model) and lm_head
- Implements manual autoregressive generation loop
- Implements temperature scaling and top-p sampling from scratch
- Shows the internal structure of the transformer
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: sample from smallest set of tokens
    whose cumulative probability exceeds top_p.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float("-inf"))

    return logits


def forward_through_components(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass showing the two main architectural components:

    1. model.model (Qwen2Model backbone):
       - embed_tokens: input_ids -> hidden_states
       - rotary_emb: computes (cos, sin) for RoPE
       - layers[0..N]: transformer decoder layers
         - input_layernorm (RMSNorm)
         - self_attn (Q/K/V projection, RoPE rotation, attention, output projection)
         - post_attention_layernorm (RMSNorm)
         - mlp (gate_proj, up_proj, down_proj with SiLU)
       - norm: final RMSNorm

    2. model.lm_head: Linear(hidden_size -> vocab_size)
    """
    # Component 1: Transformer backbone
    # Processes: embedding -> N layers (attention + MLP) -> final norm
    backbone_output = model.model(
        input_ids=input_ids,
        position_ids=position_ids,
    )
    hidden_states = backbone_output.last_hidden_state
    # Shape: [batch, seq_len, hidden_size]

    # Component 2: Language model head
    # Projects hidden states to vocabulary logits
    logits = model.lm_head(hidden_states)
    # Shape: [batch, seq_len, vocab_size]

    return logits


def generate_token_low_level(
    model,
    input_ids: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Generate one token using explicit layer-by-layer forward pass.
    """
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    with torch.no_grad():
        logits = forward_through_components(model, input_ids, position_ids)

    # Get logits for the last position only
    last_logits = logits[:, -1, :]

    # Temperature scaling
    if temperature > 0:
        last_logits = last_logits / temperature

    # Top-p filtering
    last_logits = top_p_sampling(last_logits, top_p=top_p)

    # Sample from distribution
    probs = F.softmax(last_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    Autoregressive generation using layer-level forward pass.

    Note: This version doesn't use KV-cache for simplicity,
    recomputing all positions each step (slower but clearer).
    """
    generated_ids = input_ids.clone()

    print("Generating tokens: ", end="", flush=True)

    for i in range(max_new_tokens):
        next_token = generate_token_low_level(
            model,
            generated_ids,
            temperature,
            top_p,
        )

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

        if (i + 1) % 50 == 0:
            print(f"{i + 1}", end="", flush=True)
        else:
            print(".", end="", flush=True)

    print(f" done ({i + 1} tokens)")

    return generated_ids


def print_model_structure(model):
    """Print the model's layer structure in detail."""
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE: Layer-by-layer breakdown")
    print("=" * 60)

    config = model.config
    print(f"\nModel: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Vocab size: {config.vocab_size}")

    print("\n--- Data flow ---")
    print("input_ids [batch, seq_len]")
    print("    |")
    print("    v")

    # Embedding
    print(f"embed_tokens: Embedding({config.vocab_size}, {config.hidden_size})")
    print(f"    -> hidden_states [batch, seq_len, {config.hidden_size}]")
    print("    |")
    print("    v")

    # Transformer layers
    print(f"layers: {config.num_hidden_layers} x Qwen2DecoderLayer")
    layer = model.model.layers[0]
    print("  Each layer contains:")
    for name, module in layer.named_children():
        print(f"    - {name}: {type(module).__name__}")
    print("  Inside self_attn:")
    for name, module in layer.self_attn.named_children():
        print(f"    - {name}: {type(module).__name__}")
    print("    |")
    print("    v")

    # Final norm
    print(f"norm: {type(model.model.norm).__name__}")
    print("    |")
    print("    v")

    # LM head
    print(f"lm_head: Linear({config.hidden_size}, {config.vocab_size})")
    print(f"    -> logits [batch, seq_len, {config.vocab_size}]")

    print("\n" + "=" * 60)


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )

    # Show model structure
    print_model_structure(model)

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

    print("\n=== Generation ===")
    print(f"Input tokens: {inputs['input_ids'].shape[-1]}")

    # Layer-level generation (no KV-cache for clarity)
    output_ids = generate(
        model,
        tokenizer,
        inputs["input_ids"],
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
    )

    # Decode only the generated part
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    print("\n=== Response ===")
    print(response)


if __name__ == "__main__":
    main()
