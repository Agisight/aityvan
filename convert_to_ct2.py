#!/usr/bin/env python3
"""
Конвертация NLLB модели из PyTorch в CTranslate2
Запустить один раз — потом PyTorch не нужен.

Использование:
  pip install ctranslate2 transformers sentencepiece
  python convert_to_ct2.py
"""

import os
import ctranslate2

MODEL_URL = "slone/nllb-rus-tyv-v2-extvoc"
MODEL_URL_LOCAL = "/model/nllb-rus-tyv-v2-extvoc"
CT2_OUTPUT = "/model/nllb-rus-tyv-ct2"

if os.path.exists(MODEL_URL_LOCAL):
    source = MODEL_URL_LOCAL
else:
    source = MODEL_URL

print(f"Конвертация {source} → {CT2_OUTPUT}")
print("Это займёт 1-2 минуты...")

converter = ctranslate2.converters.TransformersConverter(
    source,
    load_as_float16=False,
)

converter.convert(
    CT2_OUTPUT,
    quantization="int8",  # int8 = быстрее и меньше памяти на CPU
    force=True,
)

# Копируем SentencePiece модель
import shutil
for fname in ["sentencepiece.bpe.model", "tokenizer.json"]:
    src_path = os.path.join(source, fname)
    if os.path.exists(src_path):
        shutil.copy2(src_path, os.path.join(CT2_OUTPUT, fname))
        print(f"  Скопирован {fname}")

print(f"\n✓ Готово! Модель сохранена в {CT2_OUTPUT}")
print(f"  Размер: {sum(os.path.getsize(os.path.join(CT2_OUTPUT, f)) for f in os.listdir(CT2_OUTPUT)) / 1024 / 1024:.0f} MB")
