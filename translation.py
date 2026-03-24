"""
translation.py — исправленная версия без утечки памяти
Замена: PyTorch → CTranslate2 (C++ движок, фиксированная память)

Изменения:
  1. CTranslate2 вместо PyTorch model.generate() — нет утечки
  2. torch.no_grad() на случай если кто-то вызовет старый код
  3. Явное удаление тензоров после каждого перевода
  4. gc.collect() после каждого батча
"""

import gc
import os
import re
import sys
import typing as tp
import unicodedata

import ctranslate2
import sentencepiece as spm
from sacremoses import MosesPunctNormalizer
from sentence_splitter import SentenceSplitter

# Пути к модели — CTranslate2 формат
MODEL_URL = "slone/nllb-rus-tyv-v2-extvoc"
MODEL_URL_LOCAL = "/model/nllb-rus-tyv-v2-extvoc"
CT2_MODEL_DIR = "/model/nllb-rus-tyv-ct2"  # Конвертированная модель

if os.path.exists(MODEL_URL_LOCAL):
    MODEL_URL = MODEL_URL_LOCAL

LANGUAGES = {
    "Орус | Русский | Russian": "rus_Cyrl",
    "Тыва | Тувинский | Tyvan": "tyv_Cyrl",
}

# Маппинг языковых кодов на токены
LANG_TOKENS = {
    "rus_Cyrl": ">>rus_Cyrl<<",
    "tyv_Cyrl": ">>tyv_Cyrl<<",  
}


def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char


class TextPreprocessor:
    def __init__(self, lang="en"):
        self.mpn = MosesPunctNormalizer(lang=lang)
        self.mpn.substitutions = [
            (re.compile(r), sub) for r, sub in self.mpn.substitutions
        ]
        self.replace_nonprint = get_non_printing_char_replacer(" ")

    def __call__(self, text: str) -> str:
        clean = self.mpn.normalize(text)
        clean = self.replace_nonprint(clean)
        clean = unicodedata.normalize("NFKC", clean)
        return clean


def sentenize_with_fillers(text, splitter, fix_double_space=True, ignore_errors=False):
    if fix_double_space:
        text = re.sub(" +", " ", text)
    sentences = splitter.split(text)
    fillers = []
    i = 0
    for sentence in sentences:
        start_idx = text.find(sentence, i)
        if ignore_errors and start_idx == -1:
            start_idx = i + 1
        assert start_idx != -1, f"sent not found after {i}: `{sentence}`"
        fillers.append(text[i:start_idx])
        i = start_idx + len(sentence)
    fillers.append(text[i:])
    return sentences, fillers


class Translator:
    """
    Переводчик на CTranslate2 — без утечки памяти.
    
    CTranslate2 — это C++ движок с фиксированным потреблением RAM.
    В отличие от PyTorch, он не создаёт граф вычислений и не течёт.
    """

    def __init__(self):
        # Определяем пути
        model_dir = CT2_MODEL_DIR if os.path.exists(CT2_MODEL_DIR) else None
        
        if model_dir and os.path.exists(os.path.join(model_dir, "model.bin")):
            # CTranslate2 модель (быстрый путь, без утечки)
            print(f"Загрузка CTranslate2 модели из {model_dir}...")
            self.ct2_model = ctranslate2.Translator(
                model_dir,
                device="cpu",
                compute_type="int8",        # Быстрее и меньше памяти
                inter_threads=2,
                intra_threads=4,
            )
            self.sp_model = spm.SentencePieceProcessor()
            sp_path = os.path.join(model_dir, "sentencepiece.bpe.model")
            if not os.path.exists(sp_path):
                sp_path = os.path.join(MODEL_URL, "sentencepiece.bpe.model")
            self.sp_model.Load(sp_path)
            self.use_ct2 = True
            print("CTranslate2 модель загружена!")
        else:
            # Fallback на PyTorch (с защитой от утечки)
            print(f"CTranslate2 модель не найдена в {CT2_MODEL_DIR}")
            print(f"Загрузка PyTorch модели из {MODEL_URL}...")
            print("⚠ PyTorch режим — рекомендуется конвертировать в CT2")
            import torch
            from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
            
            self.torch_model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_URL, local_files_only=True
            )
            self.torch_model.eval()  # Важно! Отключает dropout и BatchNorm
            self.tokenizer = NllbTokenizer.from_pretrained(
                MODEL_URL, local_files_only=True
            )
            self._fix_tokenizer(self.tokenizer)
            self.use_ct2 = False
            print("PyTorch модель загружена (fallback режим)")

        self.splitter = SentenceSplitter("ru")
        self.preprocessor = TextPreprocessor()
        self.languages = LANGUAGES

    def _fix_tokenizer(self, tokenizer, new_lang="tyv_Cyrl"):
        """Add a new language token to the tokenizer vocabulary"""
        old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
        tokenizer.lang_code_to_id[new_lang] = old_len - 1
        tokenizer.id_to_lang_code[old_len - 1] = new_lang
        tokenizer.fairseq_tokens_to_ids["<mask>"] = (
            len(tokenizer.sp_model)
            + len(tokenizer.lang_code_to_id)
            + tokenizer.fairseq_offset
        )
        tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
        tokenizer.fairseq_ids_to_tokens = {
            v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()
        }
        if new_lang not in tokenizer._additional_special_tokens:
            tokenizer._additional_special_tokens.append(new_lang)
        tokenizer.added_tokens_encoder = {}
        tokenizer.added_tokens_decoder = {}

    def translate(
        self,
        text,
        src_lang="rus_Cyrl",
        tgt_lang="tyv_Cyrl",
        max_length="auto",
        num_beams=4,
        by_sentence=True,
        preprocess=True,
        **kwargs,
    ):
        """Translate text sentence by sentence, preserving fillers."""
        if by_sentence:
            sents, fillers = sentenize_with_fillers(
                text, splitter=self.splitter, ignore_errors=True
            )
        else:
            sents = [text]
            fillers = ["", ""]
        if preprocess:
            sents = [self.preprocessor(sent) for sent in sents]
        
        results = []
        for sent, sep in zip(sents, fillers):
            results.append(sep)
            results.append(
                self.translate_single(
                    sent,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    max_length=max_length,
                    num_beams=num_beams,
                    **kwargs,
                )
            )
        results.append(fillers[-1])
        return "".join(results)

    def translate_single(
        self,
        text,
        src_lang="rus_Cyrl",
        tgt_lang="tyv_Cyrl",
        max_length="auto",
        num_beams=4,
        n_out=None,
        **kwargs,
    ):
        if self.use_ct2:
            return self._translate_ct2(text, src_lang, tgt_lang, max_length, num_beams)
        else:
            return self._translate_pytorch(text, src_lang, tgt_lang, max_length, num_beams, n_out, **kwargs)

    def _translate_ct2(self, text, src_lang, tgt_lang, max_length, num_beams):
        """CTranslate2 перевод — быстрый, без утечки памяти"""
        # Токенизация через SentencePiece
        tokens = self.sp_model.Encode(text, out_type=str)
        tokens = [src_lang] + tokens + ["</s>"]
        
        target_prefix = [[tgt_lang]]
        
        if max_length == "auto":
            max_length = int(32 + 2.0 * len(tokens))
        
        results = self.ct2_model.translate_batch(
            [tokens],
            target_prefix=target_prefix,
            beam_size=num_beams,
            max_decoding_length=max_length,
        )
        
        output_tokens = results[0].hypotheses[0][1:]  # Убираем языковой токен
        return self.sp_model.Decode(output_tokens)

    def _translate_pytorch(self, text, src_lang, tgt_lang, max_length, num_beams, n_out=None, **kwargs):
        """
        PyTorch перевод — fallback с защитой от утечки.
        ИСПРАВЛЕНИЯ:
          1. torch.no_grad() — не создаёт граф вычислений
          2. del encoded, generated_tokens — явное удаление
          3. gc.collect() — принудительная сборка мусора
        """
        import torch
        
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        if max_length == "auto":
            max_length = int(32 + 2.0 * encoded.input_ids.shape[1])
        
        # ★ ИСПРАВЛЕНИЕ 1: torch.no_grad() предотвращает создание графа вычислений
        with torch.no_grad():
            generated_tokens = self.torch_model.generate(
                **encoded.to(self.torch_model.device),
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=n_out or 1,
                **kwargs,
            )
        
        out = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # ★ ИСПРАВЛЕНИЕ 2: Явное удаление тензоров
        del encoded
        del generated_tokens
        
        # ★ ИСПРАВЛЕНИЕ 3: Принудительная сборка мусора
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if isinstance(text, str) and n_out is None:
            return out[0]
        return out


if __name__ == "__main__":
    print("Initializing a translator to pre-download models...")
    translator = Translator()
    print("Initialization successful!")
