#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset_wiki_oscar_idemode.py

Версия скрипта, запускаемая из IDE (без argparse). Параметры вверху файла.
Скачивает русскую Википедию и (по возможности) OSCAR, разбивает на предложения (razdel),
для каждого предложения формирует clean/raw/char_labels и сохраняет в jsonl.

Установите зависимости:
    pip install datasets razdel tqdm

Запуск: просто нажмите Run в IDE после настройки переменных в блоке CONFIG.
"""

from datasets import load_dataset
from razdel import sentenize
from tqdm.auto import tqdm
import json
import re
import sys

# -----------------------------
# CONFIG: изменяйте здесь
# -----------------------------
OUT_PATH = "dataset_oscar.jsonl"  # куда сохранять
MAX_SENTENCES = 100000  # макс. записей в итоговом файле
WIKI_CONFIG = "20231101.ru"  # конфиг для wikimedia/wikipedia (попробуйте актуальную дату/конфиг)
STREAMING = True  # использовать streaming загрузку
SKIP_OSCAR = False  # если True — не пытаться грузить OSCAR
MAX_SENTENCE_LEN = 300  # максимальная длина предложения (в символах)
MIN_SENTENCE_LEN = 6  # минимальная длина предложения (в символах)
HF_AUTH_TOKEN = None  # если требуется токен HuggingFace, вставьте сюда строку, иначе None


# -----------------------------

def get_text_from_example(example):
    # Попытка извлечь текст из записи dataset
    for key in ("text", "content", "article", "body", "document", "raw_text", "description"):
        if key in example and example[key]:
            return example[key]
    if 'sections' in example and example['sections']:
        if isinstance(example['sections'], list):
            return " ".join([(s.get('text') if isinstance(s, dict) else str(s)) for s in example['sections'] if s])
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            return v
    return None


def sentence_to_raw_and_labels(sent: str):
    clean = " ".join(sent.split())
    if len(clean) == 0:
        return None
    raw = []
    char_labels = []
    raw_i = 0
    for ch in clean:
        if ch == " ":
            if raw_i > 0:
                while len(char_labels) < raw_i:
                    char_labels.append(0)
                char_labels[raw_i - 1] = 1
        else:
            raw.append(ch)
            raw_i += 1
    raw_s = "".join(raw)
    if len(char_labels) < len(raw_s):
        char_labels.extend([0] * (len(raw_s) - len(char_labels)))
    return {"clean": clean, "raw": raw_s, "char_labels": char_labels}


def process_dataset_split(ds_iterable, out_f, max_to_write, source_name, max_sentence_len, min_sentence_len):
    written = 0
    seen = 0
    pbar = tqdm(ds_iterable, desc=f"Processing {source_name}", unit="doc")
    for example in pbar:
        seen += 1
        text = None
        if isinstance(example, str):
            text = example
        elif isinstance(example, dict):
            text = get_text_from_example(example)
        if not text:
            continue
        text = re.sub(r"<[^>]+>", " ", text)
        text = text.replace("\r", " ").replace("\n", " ")
        try:
            for s in sentenize(text):
                sent = s.text.strip()
                if not sent:
                    continue
                L = len(sent)
                if L < min_sentence_len or L > max_sentence_len:
                    continue
                item = sentence_to_raw_and_labels(sent)
                if not item:
                    continue
                if len(item["raw"]) == 0:
                    continue
                out_obj = {
                    "source": source_name,
                    "clean": item["clean"],
                    "raw": item["raw"],
                    "char_labels": item["char_labels"]
                }
                out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                written += 1
                if written >= max_to_write:
                    pbar.close()
                    return written
        except Exception as e:
            if seen % 1000 == 0:
                tqdm.write(f"Warning: error processing example #{seen} from {source_name}: {e}", file=sys.stderr)
            continue
    return written


def process_russian_dialogues(ds_iterable, out_f, max_to_write):
    """
    Специальная обработка Den4ikAI/russian_dialogues:
    Берём поля 'question' и 'answer' (если есть) и объединяем в одну строку:
      text = question + ' ' + answer
    Затем применяем то же sentence_to_raw_and_labels и пишем запись.
    """
    written = 0
    seen = 0
    pbar = tqdm(ds_iterable, desc="Processing russian_dialogues", unit="item")
    for example in pbar:
        seen += 1
        # example expected to be dict with 'question' and 'answer'
        if not isinstance(example, dict):
            continue
        q = example.get("question") or example.get("q") or ""
        a = example.get("answer") or example.get("a") or ""
        # skip if both empty
        if not q and not a:
            continue
        # combine into one short text (как вы просили)
        # если один из них пуст — возьмём непустой
        if q and a:
            text = f"{q.strip()} {a.strip()}"
        else:
            text = (q or a).strip()
        # minimal cleaning
        text = re.sub(r"<[^>]+>", " ", text).replace("\r", " ").replace("\n", " ").strip()
        if len(text) < MIN_SENTENCE_LEN or len(text) > MAX_SENTENCE_LEN:
            continue
        item = sentence_to_raw_and_labels(text)
        if not item:
            continue
        if len(item["raw"]) == 0:
            continue
        out_obj = {
            "source": "Den4ikAI/russian_dialogues",
            "clean": item["clean"],
            "raw": item["raw"],
            "char_labels": item["char_labels"]
        }
        out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        written += 1
        if written >= max_to_write:
            pbar.close()
            return written
    return written


def main():
    total_written = 0
    print(
        f"Configuration:\n OUT_PATH={OUT_PATH}\n MAX_SENTENCES={MAX_SENTENCES}\n WIKI_CONFIG={WIKI_CONFIG}\n STREAMING={STREAMING}\n SKIP_OSCAR={SKIP_OSCAR}\n")
    with open(OUT_PATH, "w", encoding="utf8") as out_f:
        # # 1) Wikipedia
        # try:
        #     print(f"Loading Wikipedia (config={WIKI_CONFIG}) ...", file=sys.stderr)
        #     load_kwargs = {"streaming": STREAMING}
        #     if HF_AUTH_TOKEN:
        #         load_kwargs["use_auth_token"] = HF_AUTH_TOKEN
        #     wiki_ds = load_dataset("wikimedia/wikipedia", WIKI_CONFIG, split="train", **load_kwargs)
        #     written = process_dataset_split(wiki_ds, out_f, MAX_SENTENCES - total_written, "wikipedia", MAX_SENTENCE_LEN, MIN_SENTENCE_LEN)
        #     total_written += written
        #     print(f"Wikipedia done. Written: {written}. Total: {total_written}", file=sys.stderr)
        # except Exception as e:
        #     print(f"Failed to load/process Wikipedia: {e}", file=sys.stderr)

        # 2) russian_dialogues
        if total_written < MAX_SENTENCES:
            try:
                print("Loading Den4ikAI/russian_dialogues ...", file=sys.stderr)
                load_kwargs = {"streaming": STREAMING}
                if HF_AUTH_TOKEN:
                    load_kwargs["use_auth_token"] = HF_AUTH_TOKEN
                diag_ds = load_dataset("Den4ikAI/russian_dialogues", split="train", **load_kwargs)
                written = process_russian_dialogues(diag_ds, out_f, MAX_SENTENCES - total_written)
                total_written += written
                print(f"russian_dialogues done. Written: {written}. Total: {total_written}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to load/process Den4ikAI/russian_dialogues: {e}", file=sys.stderr)

        print(f"Finished. Total written: {total_written}. Output file: {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
