# Демонстрация инференса с Transformers

Генерация текста с использованием библиотеки Hugging Face `transformers`.

## Описание

Проект демонстрирует работу с языковой моделью **Qwen2.5-0.5B-Instruct** — компактной инструктивной моделью с 0.5 миллиарда параметров. Скрипт отправляет запрос на генерацию Python-функции для проверки простых чисел.

## Требования

- Python 3.11
- Conda или Mamba
- ~1 ГБ дискового пространства для модели

## Установка

```bash
conda env create -f environment.yml
conda activate transformers-model-demo
```

## Запуск

```bash
python inference.py
```

При первом запуске модель автоматически загружается с Hugging Face Hub. Используется автоматический выбор устройства (`device_map="auto"`): MPS (Metal) на Apple Silicon, CUDA на GPU или CPU.

## Параметры генерации

- `max_new_tokens`: 300
- `temperature`: 0.7
- `top_p`: 0.9
