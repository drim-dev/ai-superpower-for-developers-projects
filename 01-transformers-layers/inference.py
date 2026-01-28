"""
Text generation using low-level transformer primitives.

This script manually implements the forward pass through:
- Token embeddings
- Each transformer layer (attention + MLP)
- Final normalization
- Language model head

It shows exactly how data flows through a decoder-only transformer.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """Nucleus (top-p) sampling."""
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


def rotate_half(x):
    """Rotates half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply Rotary Position Embedding to query and key tensors."""
    # cos, sin: [batch, seq_len, head_dim]
    # q, k: [batch, num_heads, seq_len, head_dim]
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def manual_attention(
    hidden_states: torch.Tensor,
    attention: torch.nn.Module,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Manual implementation of multi-head attention with RoPE.

    Steps:
    1. Project to Q, K, V
    2. Reshape for multi-head attention
    3. Apply RoPE to Q and K
    4. Compute scaled dot-product attention
    5. Concatenate heads and project output
    """
    batch_size, seq_len, _ = hidden_states.shape

    # Step 1: Linear projections
    query_states = attention.q_proj(hidden_states)
    key_states = attention.k_proj(hidden_states)
    value_states = attention.v_proj(hidden_states)

    # Step 2: Reshape to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # Step 3: Apply RoPE (Rotary Position Embedding)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle grouped-query attention (GQA): repeat K,V for each query group
    if num_kv_heads != num_heads:
        n_rep = num_heads // num_kv_heads
        key_states = key_states.repeat_interleave(n_rep, dim=1)
        value_states = value_states.repeat_interleave(n_rep, dim=1)

    # Step 4: Scaled dot-product attention with causal mask
    # attn_weights = (Q @ K^T) / sqrt(head_dim)
    attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / (head_dim ** 0.5)

    # Apply causal mask (lower triangular)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
    attn_weights = attn_weights + causal_mask

    # Softmax and apply to values
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    # Step 5: Concatenate heads and project
    # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, num_heads * head_dim)
    attn_output = attention.o_proj(attn_output)

    return attn_output


def manual_mlp(hidden_states: torch.Tensor, mlp: torch.nn.Module) -> torch.Tensor:
    """
    Manual implementation of the MLP block.

    Qwen2 uses SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    gate = mlp.gate_proj(hidden_states)
    up = mlp.up_proj(hidden_states)
    hidden_states = F.silu(gate) * up
    hidden_states = mlp.down_proj(hidden_states)
    return hidden_states


def forward_through_layers(
    model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Manual forward pass through all transformer components.

    Data flow:
    input_ids
        -> embed_tokens -> hidden_states
        -> for each layer:
            -> input_layernorm
            -> self_attention (Q/K/V proj -> RoPE -> attention -> output proj)
            -> residual connection
            -> post_attention_layernorm
            -> mlp: down_proj(silu(gate_proj) * up_proj)
            -> residual connection
        -> final norm
        -> lm_head
        -> logits
    """
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    # Access model components
    embed_tokens = model.model.embed_tokens
    rotary_emb = model.model.rotary_emb
    layers = model.model.layers
    norm = model.model.norm
    lm_head = model.lm_head

    # Step 1: Token embedding
    hidden_states = embed_tokens(input_ids)
    # Shape: [batch, seq_len, hidden_size]

    # Step 2: Compute RoPE embeddings (shared across all layers)
    cos, sin = rotary_emb(hidden_states, position_ids)
    # cos, sin shape: [batch, seq_len, head_dim]

    # Step 3: Pass through each transformer layer
    for layer in layers:
        residual = hidden_states

        # Pre-attention normalization (RMSNorm)
        hidden_states = layer.input_layernorm(hidden_states)

        # Self-attention with RoPE
        hidden_states = manual_attention(
            hidden_states,
            layer.self_attn,
            cos, sin,
            num_heads, num_kv_heads, head_dim,
        )

        # Residual connection
        hidden_states = residual + hidden_states

        residual = hidden_states

        # Pre-MLP normalization (RMSNorm)
        hidden_states = layer.post_attention_layernorm(hidden_states)

        # MLP (SwiGLU)
        hidden_states = manual_mlp(hidden_states, layer.mlp)

        # Residual connection
        hidden_states = residual + hidden_states

    # Step 4: Final normalization
    hidden_states = norm(hidden_states)

    # Step 5: Project to vocabulary
    logits = lm_head(hidden_states)
    # Shape: [batch, seq_len, vocab_size]

    return logits


def generate_token(
    model,
    input_ids: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Generate one token using manual forward pass."""
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    with torch.no_grad():
        logits = forward_through_layers(model, input_ids, position_ids)

    last_logits = logits[:, -1, :]

    if temperature > 0:
        last_logits = last_logits / temperature

    last_logits = top_p_sampling(last_logits, top_p=top_p)

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
    """Autoregressive generation using manual forward pass."""
    generated_ids = input_ids.clone()

    print("Generating tokens: ", end="", flush=True)

    for i in range(max_new_tokens):
        next_token = generate_token(model, generated_ids, temperature, top_p)
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
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    config = model.config
    print(f"\nModel: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Num KV heads: {config.num_key_value_heads}")
    print(f"Head dim: {config.hidden_size // config.num_attention_heads}")
    print(f"Vocab size: {config.vocab_size}")

    print("\n--- Components used in forward pass ---")
    print(f"embed_tokens: {type(model.model.embed_tokens).__name__}")
    print(f"rotary_emb: {type(model.model.rotary_emb).__name__}")
    print(f"layers: {config.num_hidden_layers} x {type(model.model.layers[0]).__name__}")

    layer = model.model.layers[0]
    print(f"  - input_layernorm: {type(layer.input_layernorm).__name__}")
    print(f"  - self_attn.q_proj: Linear({config.hidden_size}, {config.hidden_size})")
    print(f"  - self_attn.k_proj: Linear({config.hidden_size}, {config.num_key_value_heads * config.hidden_size // config.num_attention_heads})")
    print(f"  - self_attn.v_proj: Linear({config.hidden_size}, {config.num_key_value_heads * config.hidden_size // config.num_attention_heads})")
    print(f"  - self_attn.o_proj: Linear({config.hidden_size}, {config.hidden_size})")
    print(f"  - post_attention_layernorm: {type(layer.post_attention_layernorm).__name__}")
    print(f"  - mlp.gate_proj: Linear({config.hidden_size}, {config.intermediate_size})")
    print(f"  - mlp.up_proj: Linear({config.hidden_size}, {config.intermediate_size})")
    print(f"  - mlp.down_proj: Linear({config.intermediate_size}, {config.hidden_size})")

    print(f"norm: {type(model.model.norm).__name__}")
    print(f"lm_head: Linear({config.hidden_size}, {config.vocab_size})")

    print("\n" + "=" * 60)


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

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

    output_ids = generate(
        model,
        tokenizer,
        inputs["input_ids"],
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
    )

    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    print("\n=== Response ===")
    print(response)


if __name__ == "__main__":
    main()
