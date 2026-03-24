# AI Tyvan — Tuvan-Russian Translator

A neural machine translation service between Tuvan and Russian languages, powered by [NLLB-200](https://huggingface.co/slone/nllb-rus-tyv-v2-extvoc) with Gradio UI and REST API.

Trained on [Agisight/tyv-rus-200k](https://huggingface.co/datasets/Agisight/tyv-rus-200k) — 296K parallel pairs collected via [tyvan.ru](https://tyvan.ru).

It is described in the post:
https://medium.com/@cointegrated/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865

## Prerequisites

Docker. A few gigabytes of memory.

If you want to speedup the building of the docker image, you can pre-download the model: run `download_model.py`
Or run like here:

```
python download_model.py
```

This requires the `huggingface_hub` library, install it if you don't have it yet:

```
pip install huggingface_hub
```

Please make sure that you have `git` and `git-lfs` installed first.

## How to run

Build a Docker image called "nllb" (from the current directory):

```
docker build -t nllb .
```

Run it:

```
docker run -it -p 7860:7860 nllb
```

Now open the browser at http://localhost:7860/docs.
It will show you a signature of the method you can use for translation.

## Translation engine

Two inference modes are supported:

| Mode | Speed | Memory | Leak-safe |
|------|-------|--------|-----------|
| **CTranslate2** (recommended) | 1-2 sec | ~600 MB | Yes |
| PyTorch (fallback) | 3-5 sec | ~2.5 GB | Fixed |

### Converting to CTranslate2 (recommended)

CTranslate2 is a C++ inference engine — faster and uses fixed memory (no leaks).

```
pip install ctranslate2 transformers sentencepiece
python convert_to_ct2.py
```

After conversion, `torch` and `transformers` can be removed — CTranslate2 runs standalone.

If the CT2 model is not found, the translator automatically falls back to PyTorch with memory leak protection (`torch.no_grad()`, explicit tensor cleanup, `gc.collect()`).

## Testing

Run the built-in test suite (no manual input needed):

```
bash test_aityvan.sh                     # default: localhost:7860
bash test_aityvan.sh 10.10.10.50:8000    # custom address
```

Includes 14 translations in both directions + memory leak check.

## API

```bash
# Russian → Tuvan
curl -X POST http://localhost:7860/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет!", "src_lang": "rus_Cyrl", "tgt_lang": "tyv_Cyrl"}'

# Tuvan → Russian
curl -X POST http://localhost:7860/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Экии!", "src_lang": "tyv_Cyrl", "tgt_lang": "rus_Cyrl"}'

# GET request (Russian → Tuvan)
curl "http://localhost:7860/translator?rus=Привет"

# List supported languages
curl http://localhost:7860/list-languages
```

## How to adapt

If you want to deploy another NLLB-based translation model,
just change the `MODEL_URL` in the `translation.py` file.
You may also want to adjust the `LANGUAGES` register in the same file.

## Preventing downloading for each docker build

To prevent downloading of the big model archive, download it once and place just over `place-nllb-rus-tyv-v2-extvoc` folder and use folder name as `MODEL_URL` in `translation.py`

## Memory leak fix

The previous version used PyTorch without `torch.no_grad()` — each translation accumulated computation graphs, growing memory by hundreds of MBs over time and eventually crashing the server.

Fixes applied:
1. **CTranslate2** — C++ engine with fixed memory allocation, cannot leak by design
2. **torch.no_grad()** — disables gradient accumulation (PyTorch fallback)
3. **del + gc.collect()** — explicit tensor cleanup after each translation
4. **systemd MemoryMax** — hard limit with automatic restart on exceed

## Author

Ali Kuzhuget — [tyvan.ru](https://tyvan.ru)

## License

MIT
