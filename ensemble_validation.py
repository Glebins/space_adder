"""
Validates ensembles out of 2 models
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from typing import List, Dict, Any, Set

from bpe import TokenBPEClassifier, BPESpaceDataset
from char_cnn import CharCNN, CharSpaceDataset, CharVocab
from utils import read_jsonl, positions_from_char_labels, f1_from_sets, collate_fn_bilstm, collate_fn_cnn, \
    predict_positions_for_raw_bpe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256
BATCH_SIZE = 16

CHAR_EMB_DIM = 64
CHAR_CONV_CHANNELS = 256
CHAR_KERNEL_SIZES = [1, 3, 5, 7]
CHAR_DROPOUT = 0.1

BPE_MODEL_DIR = "out_bpe_bilstm"
CNN_MODEL_PATH = "out_char_cnn/best.pt"
PATH_DATA = ["dataset_wiki.jsonl", "dataset_dialogues.jsonl"]

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

items = read_jsonl(PATH_DATA)
if len(items) == 0:
    raise RuntimeError("No data found in PATH_DATA")

import random

random.shuffle(items)
split_idx = int(len(items) * 0.95)
train_items = items[:split_idx]
val_items = items[split_idx:]

vocab = CharVocab(min_freq=1)
vocab.feed_texts([r["raw"] for r in train_items])
vocab.build()
print("Char vocab size:", len(vocab))

bpe_val_ds = BPESpaceDataset(val_items, tokenizer, max_length=MAX_LENGTH)
bpe_val_loader = DataLoader(bpe_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_bilstm)

char_val_ds = CharSpaceDataset(val_items, vocab, max_len=MAX_LENGTH)
char_val_loader = DataLoader(char_val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_cnn)

vocab_size = tokenizer.vocab_size
bpe_model = TokenBPEClassifier(vocab_size=vocab_size)
bpe_ckpt_path = Path(BPE_MODEL_DIR) / "best_model.pt"
if not bpe_ckpt_path.exists():
    raise RuntimeError(f"BPE model checkpoint not found: {bpe_ckpt_path}")
bpe_ckpt = torch.load(bpe_ckpt_path, map_location=DEVICE)
if isinstance(bpe_ckpt, dict) and "model_state_dict" in bpe_ckpt:
    bpe_model.load_state_dict(bpe_ckpt["model_state_dict"])
else:
    bpe_model.load_state_dict(bpe_ckpt)
bpe_model.to(DEVICE)
bpe_model.eval()

cnn_model = CharCNN(vocab_size=len(vocab), emb_dim=CHAR_EMB_DIM,
                    conv_channels=CHAR_CONV_CHANNELS, kernel_sizes=CHAR_KERNEL_SIZES,
                    dropout=CHAR_DROPOUT)
cnn_ckpt_path = Path(CNN_MODEL_PATH)
if not cnn_ckpt_path.exists():
    raise RuntimeError(f"CharCNN checkpoint not found: {cnn_ckpt_path}")
cnn_ckpt = torch.load(cnn_ckpt_path, map_location=DEVICE)
if isinstance(cnn_ckpt, dict):
    if "model_state_dict" in cnn_ckpt:
        cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
    elif "model" in cnn_ckpt:
        cnn_model.load_state_dict(cnn_ckpt["model"])
    elif "model_state" in cnn_ckpt:
        cnn_model.load_state_dict(cnn_ckpt["model_state"])
    else:
        try:
            cnn_model.load_state_dict(cnn_ckpt)
        except Exception as e:
            raise RuntimeError("Unknown CharCNN checkpoint format") from e
else:
    cnn_model.load_state_dict(cnn_ckpt)
cnn_model.to(DEVICE)
cnn_model.eval()


def predict_positions_for_raw_cnn_local(model: torch.nn.Module, vocab_obj: CharVocab,
                                        raw: str, device: str, max_length: int) -> Set[int]:
    ids = vocab_obj.encode(raw, max_length)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)  # (1, L)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)  # (1, L, 2)
        preds = logits.argmax(dim=-1).cpu().numpy()[0]  # (L,)
    Lraw = min(len(raw), max_length)
    predicted_positions = {i + 1 for i, p in enumerate(preds[:Lraw]) if int(p) == 1}
    return predicted_positions


def ensemble_predict(raw: str, bpe_model, cnn_model, tokenizer, device, vocab_obj, max_length: int) -> Set[int]:
    bpe_positions = predict_positions_for_raw_bpe(bpe_model, tokenizer, raw, device, max_length)
    char_positions = predict_positions_for_raw_cnn_local(cnn_model, vocab_obj, raw, device, max_length)
    return bpe_positions | char_positions


f1_scores = []
for rec in tqdm(val_items, desc="Ensemble validation"):
    raw = rec["raw"]
    true_positions = positions_from_char_labels(rec["char_labels"])
    pred_positions = ensemble_predict(raw, bpe_model, cnn_model, tokenizer, DEVICE, vocab, MAX_LENGTH)
    f1 = f1_from_sets(pred_positions, true_positions)
    f1_scores.append(f1)

avg_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0
print(f"Ensemble F1 on validation: {avg_f1:.4f}")
