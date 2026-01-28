# Демонстрация инференса на уровне компонентов Transformer

Генерация текста с разделением модели на компоненты и ручной реализацией генерации.

## Описание

Проект демонстрирует работу с языковой моделью **Qwen2.5-0.5B-Instruct**, разделяя модель на основные компоненты:

- **model.model** (Qwen2Model) — transformer backbone
- **model.lm_head** — проекция на словарь
- **Ручной цикл авторегрессионной генерации** вместо `model.generate()`
- **Собственная реализация сэмплирования**: temperature scaling и nucleus (top-p) sampling

## Архитектура модели

```text
AutoModelForCausalLM (Qwen2ForCausalLM)
├── model (Qwen2Model) — transformer backbone
│   ├── embed_tokens: Embedding(vocab_size, hidden_size)
│   ├── rotary_emb: вычисляет (cos, sin) для RoPE
│   ├── layers: 24 × Qwen2DecoderLayer
│   │   ├── input_layernorm (RMSNorm)
│   │   ├── self_attn
│   │   │   ├── q_proj, k_proj, v_proj (Linear)
│   │   │   ├── RoPE rotation на Q и K
│   │   │   ├── scaled dot-product attention
│   │   │   └── o_proj (Linear)
│   │   ├── post_attention_layernorm (RMSNorm)
│   │   └── mlp
│   │       ├── gate_proj (Linear)
│   │       ├── up_proj (Linear)
│   │       ├── SiLU activation
│   │       └── down_proj (Linear)
│   └── norm: final RMSNorm
└── lm_head: Linear(hidden_size, vocab_size)
```

## Что показывает демо

1. Структуру компонентов модели и их типы
2. Разделение на backbone (model.model) и lm_head
3. Ручной цикл генерации токен за токеном
4. Реализацию temperature scaling для контроля "креативности"
5. Реализацию top-p (nucleus) sampling

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

## Сравнение с высокоуровневым подходом

| Аспект                | 02-transformers-model    | 01-transformers-layers        |
|-----------------------|--------------------------|-------------------------------|
| Генерация             | `model.generate()`       | Ручной цикл                   |
| Forward pass          | `model()` как чёрный ящик | `model.model()` + `lm_head()` |
| Сэмплирование         | Встроенное               | Своя реализация               |
| KV-cache              | Автоматический           | Отключён для ясности          |
| Вывод архитектуры     | Нет                      | Да                            |
| Код                   | ~20 строк                | ~150 строк                    |
