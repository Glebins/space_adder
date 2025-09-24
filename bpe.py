"""
Train a custom PyTorch token-classifier using a pretrained BPE fast-tokenizer.

- Uses tokenizer from transformers to obtain token offsets.
- Builds token-level labels from char-level labels.
- Custom model: Embedding -> BiLSTM -> Linear -> logits per token (2 classes).
- Computes evaluation F1 on positions (as in task): predicted positions = set of token-end offsets
  for which model predicts a space; ground-truth positions built from char_labels.
"""

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
from utils import read_jsonl, f1_from_sets, collate_fn_bilstm, positions_from_char_labels, split_data, \
    build_token_labels_for_raw, predict_positions_for_raw_bpe

MODEL_NAME = "roberta-base"
PATH_DATA = ["dataset_wiki.jsonl", "dataset_dialogues.jsonl"]
OUTPUT_DIR = "out_bpe_bilstm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_PROPORTION = 0.1
SEED = 42
EMBED_DIM = 128
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT = 0.1
VAL_FRACTION = 0.05
NUM_WORKERS = 2

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


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
        aligned = build_token_labels_for_raw(self.tokenizer, raw, char_labels, self.max_length)
        return {
            "input_ids": torch.tensor(aligned["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(aligned["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(aligned["labels"], dtype=torch.long),
            "raw": raw,
            "char_labels": torch.tensor(char_labels, dtype=torch.long),
            "offsets": aligned["offsets"],
            "num_token_no_special": aligned["num_token_no_special"]
        }


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
        out, _ = self.lstm(emb)  # (B, L, 2*H)
        out = self.dropout(out)
        logits = self.classifier(out)  # (B, L, num_labels)
        return logits


def compute_loss_and_backprop(model, batch, criterion, optimizer, scheduler=None):
    model.train()
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    logits = model(input_ids, attention_mask)  # (B, L, C)
    B, L, C = logits.shape
    loss = criterion(logits.view(-1, C), labels.view(-1))
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()
    return loss.item()


def main():
    print("Loading tokenizer (fast) ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              collate_fn=collate_fn_bilstm)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=collate_fn_bilstm)

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
        model.eval()
        f1s = []
        pbar_val = tqdm(val_ds, desc="Validation", total=len(val_ds))
        for rec in pbar_val:
            raw = rec["raw"]
            char_labels = rec["char_labels"].tolist()
            true_positions = positions_from_char_labels(char_labels)
            pred_positions = predict_positions_for_raw_bpe(model, tokenizer, raw, DEVICE, MAX_LENGTH)
            f1 = f1_from_sets(pred_positions, true_positions)
            f1s.append(f1)
            pbar_val.set_postfix({"avg_f1": f"{(sum(f1s) / len(f1s)):.4f}"})
        avg_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
        print(f"Epoch {epoch} completed. Avg F1 on val (positions): {avg_f1:.4f}")

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
            tokenizer.save_pretrained(out_dir)
            print(f"Saved best model with F1={best_f1:.4f} to {out_dir}")

    print("Training finished.")


if __name__ == "__main__":
    main()
