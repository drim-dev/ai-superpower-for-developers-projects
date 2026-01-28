# Демонстрация инференса на уровне примитивов Transformer

Генерация текста с ручной реализацией attention, RoPE и MLP.

## Описание

Проект демонстрирует работу с языковой моделью **Qwen2.5-0.5B-Instruct** на самом низком уровне абстракции, вручную реализуя:

- **Rotary Position Embedding (RoPE)** — поворот Q и K для кодирования позиций
- **Multi-Head Attention** — Q/K/V проекции, scaled dot-product attention, causal mask
- **Grouped-Query Attention (GQA)** — оптимизация с меньшим числом K/V голов
- **SwiGLU MLP** — gate_proj, up_proj, SiLU activation, down_proj
- **Residual connections** и **RMSNorm**

## Что реализовано вручную

```python
# RoPE - поворот векторов на угол, зависящий от позиции
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

q_embed = (q * cos) + (rotate_half(q) * sin)

# Scaled Dot-Product Attention
attn_weights = (Q @ K^T) / sqrt(head_dim)
attn_weights = attn_weights + causal_mask  # -inf для будущих позиций
attn_output = softmax(attn_weights) @ V

# SwiGLU MLP
output = down_proj(silu(gate_proj(x)) * up_proj(x))
```

## Архитектура и поток данных

```text
input_ids [batch, seq_len]
    │
    ▼
embed_tokens(input_ids)
    → hidden_states [batch, seq_len, 896]
    │
    ▼
rotary_emb(hidden_states, position_ids)
    → cos, sin [batch, seq_len, 64]  # head_dim = 896/14 = 64
    │
    ▼
for layer in layers[0..23]:
    │
    ├─► residual = hidden_states
    │
    ├─► input_layernorm(hidden_states)  # RMSNorm
    │
    ├─► manual_attention:
    │   ├─ q = q_proj(hidden_states)     [batch, seq, 896]
    │   ├─ k = k_proj(hidden_states)     [batch, seq, 128]  # 2 KV heads
    │   ├─ v = v_proj(hidden_states)     [batch, seq, 128]
    │   ├─ reshape to [batch, heads, seq, head_dim]
    │   ├─ apply RoPE to q, k
    │   ├─ repeat k,v for GQA (2 → 14 heads)
    │   ├─ attn = softmax(q @ k.T / √64 + causal_mask) @ v
    │   └─ o_proj(attn)                  [batch, seq, 896]
    │
    ├─► hidden_states = residual + attention_output
    │
    ├─► residual = hidden_states
    │
    ├─► post_attention_layernorm(hidden_states)  # RMSNorm
    │
    ├─► manual_mlp:
    │   ├─ gate = gate_proj(hidden_states)  [batch, seq, 4864]
    │   ├─ up = up_proj(hidden_states)      [batch, seq, 4864]
    │   ├─ hidden = silu(gate) * up
    │   └─ down_proj(hidden)                [batch, seq, 896]
    │
    └─► hidden_states = residual + mlp_output
    │
    ▼
norm(hidden_states)  # Final RMSNorm
    │
    ▼
lm_head(hidden_states)
    → logits [batch, seq_len, 151936]
```

## Ключевые концепции

### RoPE (Rotary Position Embedding)
Кодирует позицию токена через поворот вектора в комплексной плоскости:
- Для каждой позиции вычисляются cos и sin
- Q и K поворачиваются на угол, пропорциональный позиции
- Позволяет модели понимать относительные позиции токенов

### Grouped-Query Attention (GQA)
Оптимизация памяти: вместо 14 голов K и V используются только 2.
Каждая K/V голова обслуживает 7 Q голов (14/2 = 7).

### SwiGLU
Улучшенный MLP: `silu(gate) * up` вместо простого `relu(up)`.
Gated activation улучшает качество модели.

## Требования

- Python 3.11
- Conda или Mamba
- ~1 ГБ дискового пространства для модели

## Установка

```bash
conda env create -f environment.yml
conda activate transformers-layers-demo
```

## Запуск

```bash
python inference.py
```

## Сравнение уровней абстракции

| Уровень           | Что скрыто                           | Что видно                    |
|-------------------|--------------------------------------|------------------------------|
| `model.generate()`| Всё                                  | Только вход/выход            |
| `model()`         | Attention, MLP, RoPE                 | Logits                       |
| `model.model()`   | Attention, MLP, RoPE                 | Hidden states                |
| **Этот код**      | Только Linear/RMSNorm                | Attention, RoPE, MLP, маски  |
