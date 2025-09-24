#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a custom PyTorch token-classifier using a pretrained BPE fast-tokenizer.

- Uses tokenizer from transformers (use_fast=True) to obtain token offsets.
- Builds token-level labels from char-level labels.
- Custom model: Embedding -> BiLSTM -> Linear -> logits per token (2 classes).
- Computes evaluation F1 on positions (as in task): predicted positions = set of token-end offsets
  for which model predicts a space; ground-truth positions built from char_labels.
"""

import os
import json
import random
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ---------------------------
# CONFIG (меняй в IDE)
# ---------------------------
MODEL_NAME = "roberta-base"  # только для токенизатора (use_fast=True). Можно заменить на русскоязычный tokenizer.
PATH_DATA = "dataset.jsonl"  # jsonl с полями: clean, raw, char_labels
OUTPUT_DIR = "out_custom_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 6
LR = 1e-3  # т.к. модель тренируется с нуля, lr выше чем для fine-tuning
WEIGHT_DECAY = 1e-5
WARMUP_PROPORTION = 0.1
SEED = 42
EMBED_DIM = 128
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.2
VAL_FRACTION = 0.05
NUM_WORKERS = 2
# ---------------------------

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------
# Utilities: IO
# ---------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def split_data(items: List[Dict[str, Any]], val_frac=0.05, shuffle=True):
    if shuffle:
        random.shuffle(items)
    nval = int(len(items) * val_frac)
    if nval == 0:
        return items, []
    return items[nval:], items[:nval]


# ---------------------------
# Token-label alignment (как раньше)
# ---------------------------
def build_token_labels_for_raw(tokenizer, raw: str, char_labels: List[int], max_length: int):
    """
    Tokenize raw with add_special_tokens=False to get offsets.
    Build token-level labels (0/1). Then add special tokens (prefix/suffix) with -100 labels.
    Finally pad/truncate to max_length and return input_ids, attention_mask, labels (len=max_length).
    """
    enc = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
    input_ids_no_special = enc["input_ids"]
    offsets = enc["offset_mapping"]  # list of (st, ed)
    true_char_positions = {i for i, v in enumerate(char_labels) if
                           v == 1}  # positions: after char i -> corresponds to token end ed if ed-1 in this set
    token_labels = []
    for (st, ed) in offsets:
        if ed > 0 and (ed - 1) in true_char_positions:
            token_labels.append(1)
        else:
            token_labels.append(0)

    # special tokens
    num_special = tokenizer.num_special_tokens_to_add(False)
    prefix_special = 1 if tokenizer.cls_token_id is not None else 0
    suffix_special = num_special - prefix_special
    labels_with_special = ([-100] * prefix_special) + token_labels + ([-100] * suffix_special)

    input_ids_with_special = tokenizer.build_inputs_with_special_tokens(input_ids_no_special)
    attention_mask = [1] * len(input_ids_with_special)

    # pad/truncate
    if len(input_ids_with_special) > max_length:
        input_ids_with_special = input_ids_with_special[:max_length]
        attention_mask = attention_mask[:max_length]
        labels_with_special = labels_with_special[:max_length]
    else:
        pad_len = max_length - len(input_ids_with_special)
        input_ids_with_special = input_ids_with_special + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels_with_special = labels_with_special + ([-100] * pad_len)

    return {
        "input_ids": input_ids_with_special,
        "attention_mask": attention_mask,
        "labels": labels_with_special,
        "offsets": offsets,
        "num_token_no_special": len(input_ids_no_special)
    }


# ---------------------------
# Dataset
# ---------------------------
class BPESpaceDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], tokenizer, max_length=256, max_examples=None):
        self.rows = items if max_examples is None else items[:max_examples]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        raw = rec["raw"]
        char_labels = rec["char_labels"]
        # alignment
        aligned = build_token_labels_for_raw(self.tokenizer, raw, char_labels, self.max_length)
        return {
            "input_ids": torch.tensor(aligned["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(aligned["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned["labels"], dtype=torch.long),
            # keep raw and offsets for evaluation later
            "raw": raw,
            "char_labels": torch.tensor(char_labels, dtype=torch.long),
            "offsets": aligned["offsets"],  # list of (st,ed) for tokens without special tokens
            "num_token_no_special": aligned["num_token_no_special"]
        }


# ---------------------------
# Model: Embedding -> BiLSTM -> Linear
# ---------------------------
class TokenBPEClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=128, hidden_size=256, n_layers=2, dropout=0.2, pad_token_id=0,
                 num_labels=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=n_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, L)
        emb = self.emb(input_ids)  # (B, L, E)
        # Note: we do not use packing; padded tokens will produce outputs but labels for them are -100
        out, _ = self.lstm(emb)  # (B, L, 2*H)
        out = self.dropout(out)
        logits = self.classifier(out)  # (B, L, num_labels)
        return logits


# ---------------------------
# Training & Eval helpers
# ---------------------------
def compute_loss_and_backprop(model, batch, criterion, optimizer, scheduler=None):
    model.train()
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)  # (B, L) with -100 for ignore
    logits = model(input_ids, attention_mask)  # (B, L, C)
    B, L, C = logits.shape
    loss = criterion(logits.view(-1, C), labels.view(-1))
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()
    return loss.item()


def predict_positions_for_raw(model, tokenizer, raw: str, device) -> Set[int]:
    """
    Run model on a single raw example and return set of predicted positions (as defined in task).
    Steps:
      - tokenize raw with add_special_tokens=False -> offsets
      - build inputs_with_special_tokens and pad to MAX_LENGTH
      - run model -> logits
      - extract token logits corresponding to non-special tokens (slice using prefix_special)
      - token prediction: argmax over classes
      - predicted positions: for token i with pred==1 -> add offsets[i][1] (ed)
    """
    enc = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
    input_ids_no_special = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if len(input_ids_no_special) == 0:
        return set()
    input_ids_with_special = tokenizer.build_inputs_with_special_tokens(input_ids_no_special)
    # pad/truncate to MAX_LENGTH
    if len(input_ids_with_special) > MAX_LENGTH:
        input_ids_with_special = input_ids_with_special[:MAX_LENGTH]
    else:
        pad_len = MAX_LENGTH - len(input_ids_with_special)
        input_ids_with_special = input_ids_with_special + [tokenizer.pad_token_id] * pad_len
    input_tensor = torch.tensor([input_ids_with_special], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_tensor)  # (1, L, C)
        logits = logits.cpu().numpy()[0]  # (L, C)
    # compute prefix length
    num_special = tokenizer.num_special_tokens_to_add(False)
    prefix = 1 if tokenizer.cls_token_id is not None else 0
    # slice token logits corresponding to no-special tokens
    # token logits are at indices [prefix, prefix+len(input_ids_no_special)-1]
    token_logits = logits[prefix: prefix + len(input_ids_no_special)]
    token_preds = token_logits.argmax(axis=-1)  # 0/1 array length = len(input_ids_no_special)
    predicted_positions = set()
    for tok_idx, pred in enumerate(token_preds):
        if pred == 1:
            # offsets[tok_idx] corresponds to (st, ed)
            st, ed = offsets[tok_idx]
            # position index in task = ed (index where next char starts)
            predicted_positions.add(ed)
    return predicted_positions


def positions_from_char_labels(char_labels: List[int]) -> Set[int]:
    # char_labels[i] == 1 means space after char i -> position = i+1
    positions = set()
    for i, v in enumerate(char_labels):
        if v == 1:
            positions.add(i + 1)
    return positions


def f1_from_sets(pred: Set[int], true: Set[int]) -> float:
    if len(pred) == 0 and len(true) == 0:
        return 1.0
    if len(pred) == 0 or len(true) == 0:
        return 0.0
    tp = len(pred & true)
    prec = tp / len(pred) if len(pred) > 0 else 0.0
    rec = tp / len(true) if len(true) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# Добавь эту функцию где-нибудь вверху (рядом с определением датасета)
def collate_fn(batch):
    """
    batch: list of dicts returned by __getitem__
    Собираем батч: stack для фиксированных тензоров, остальные поля - в списки.
    """
    # Стек для тензоров, которые имеют одинаковую форму (паддятся в __getitem__)
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)

    # Сохраняем прочие поля как списки (offsets, raw, char_labels и т.д.)
    others = {}
    for key in batch[0].keys():
        if key in ("input_ids", "attention_mask", "labels"):
            continue
        others[key] = [item[key] for item in batch]

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    out.update(others)
    return out


# ---------------------------
# Main training flow
# ---------------------------
def main():
    print("Loading tokenizer (fast) ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    # ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    print("Loading dataset ...", PATH_DATA)
    items = read_jsonl(PATH_DATA)
    if len(items) == 0:
        raise RuntimeError("No data found in PATH_DATA")
    print(f"Loaded {len(items)} examples.")

    train_items, val_items = split_data(items, val_frac=VAL_FRACTION)
    print(f"Train size: {len(train_items)}, Val size: {len(val_items)}")

    train_ds = BPESpaceDataset(train_items, tokenizer, max_length=MAX_LENGTH)
    val_ds = BPESpaceDataset(val_items, tokenizer, max_length=MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id

    model = TokenBPEClassifier(vocab_size=vocab_size, emb_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE,
                               n_layers=NUM_LSTM_LAYERS, dropout=DROPOUT, pad_token_id=pad_id, num_labels=2)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(WARMUP_PROPORTION * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    best_f1 = -1.0
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for batch in pbar:
            loss = compute_loss_and_backprop(model, batch, criterion, optimizer, scheduler)
            running_loss += loss
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}"})
        # Evaluation: compute average F1 over val set (we'll do per-example inference for correctness)
        model.eval()
        f1s = []
        # iterate val dataset example-by-example for precise offsets handling
        pbar_val = tqdm(val_ds, desc="Validation", total=len(val_ds))
        for rec in pbar_val:
            raw = rec["raw"]
            char_labels = rec["char_labels"].tolist()
            true_positions = positions_from_char_labels(char_labels)
            pred_positions = predict_positions_for_raw(model, tokenizer, raw, DEVICE)
            f1 = f1_from_sets(pred_positions, true_positions)
            f1s.append(f1)
            pbar_val.set_postfix({"avg_f1": f"{(sum(f1s) / len(f1s)):.4f}"})
        avg_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
        print(f"Epoch {epoch} completed. Avg F1 on val (positions): {avg_f1:.4f}")

        # save best
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer": tokenizer.get_vocab(),
                "config": {
                    "MODEL_NAME": MODEL_NAME,
                    "EMBED_DIM": EMBED_DIM,
                    "HIDDEN_SIZE": HIDDEN_SIZE,
                    "NUM_LSTM_LAYERS": NUM_LSTM_LAYERS
                }
            }, out_dir / "best_model.pt")
            # also save tokenizer files for later use
            tokenizer.save_pretrained(out_dir)
            print(f"Saved best model with F1={best_f1:.4f} to {out_dir}")

    print("Training finished.")


if __name__ == "__main__":
    main()
