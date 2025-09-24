"""
Train a custom PyTorch token-classifier using a 1D CNN.
"""

import json, random
from pathlib import Path
from typing import List, Dict, Any, Set
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from utils import read_jsonl, f1_from_sets, positions_from_char_labels, collate_fn_cnn

PATH_DATA = ["dataset_wiki.jsonl", "dataset_dialogues.jsonl"]
OUT_DIR = "out_char_cnn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256
BATCH_SIZE = 32
NUM_EPOCHS = 3
LR = 5e-4
EMBED_DIM = 64
CONV_CHANNELS = 256
KERNEL_SIZES = [1, 3, 5, 7]
DROPOUT = 0.1
WEIGHT_DECAY = 1e-5
SEED = 42
NUM_WORKERS = 4

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class CharVocab:
    def __init__(self, min_freq=1):
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        self.ch2i = {self.PAD: 0, self.UNK: 1}
        self.i2ch = {0: self.PAD, 1: self.UNK}
        self.counter = {}
        self.min_freq = min_freq

    def feed_texts(self, texts: List[str]):
        for t in texts:
            for ch in t:
                self.counter[ch] = self.counter.get(ch, 0) + 1

    def build(self):
        idx = len(self.ch2i)
        for ch, cnt in sorted(self.counter.items(), key=lambda x: -x[1]):
            if cnt < self.min_freq: continue
            if ch in self.ch2i: continue
            self.ch2i[ch] = idx
            self.i2ch[idx] = ch
            idx += 1

    def encode(self, s: str, max_len: int) -> List[int]:
        res = []
        for ch in s[:max_len]:
            res.append(self.ch2i.get(ch, self.ch2i[self.UNK]))
        if len(res) < max_len:
            res += [self.ch2i[self.PAD]] * (max_len - len(res))
        return res

    def __len__(self):
        return len(self.ch2i)


class CharSpaceDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], vocab: CharVocab, max_len: int):
        self.rows = rows
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        raw = rec["raw"]
        char_labels = rec["char_labels"]
        lab = char_labels[:self.max_len]
        if len(lab) < self.max_len:
            lab = lab + [-100] * (self.max_len - len(lab))
        ids = self.vocab.encode(raw, self.max_len)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "raw": raw,
            "char_labels": char_labels
        }


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, conv_channels: int, kernel_sizes: List[int], dropout: float):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            pad = (k - 1) // 2
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels=emb_dim, out_channels=conv_channels, kernel_size=k, padding=pad),
                nn.ReLU(),
                nn.BatchNorm1d(conv_channels)
            ))
        total_ch = conv_channels * len(kernel_sizes)
        self.conv_mix = nn.Sequential(
            nn.Conv1d(total_ch, total_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(total_ch)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(total_ch, 2)

    def forward(self, input_ids):
        # input_ids: (B, L)
        emb = self.emb(input_ids)  # (B, L, E)
        x = emb.permute(0, 2, 1)  # (B, E, L)
        outs = []
        for conv in self.convs:
            outs.append(conv(x))  # (B, C, L)
        xcat = torch.cat(outs, dim=1)  # (B, C_total, L)
        xmix = self.conv_mix(xcat)  # (B, C_total, L)
        xmix = xmix.permute(0, 2, 1)  # (B, L, C_total)
        xmix = self.dropout(xmix)
        logits = self.classifier(xmix)  # (B, L, 2)
        return logits


def train_epoch(model, loader, opt, criterion, device):
    model.train()
    total_loss = 0.0
    p = tqdm(loader, desc="train")
    for batch in p:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)  # (B, L, 2)
        B, L, C = logits.shape
        loss = criterion(logits.view(-1, C), labels.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        p.set_postfix({"loss": total_loss / (p.n + 1)})
    return total_loss / len(loader)


def evaluate_positions_f1(model, dataset, device, sample_limit=None):
    model.eval()
    f1s = []
    with torch.no_grad():
        it = range(len(dataset)) if sample_limit is None else range(min(len(dataset), sample_limit))
        for i in tqdm(it, desc="val"):
            rec = dataset[i]
            raw = rec["raw"]
            true_char_labels = rec["char_labels"]
            ids = torch.tensor(dataset.vocab.encode(raw, dataset.max_len), dtype=torch.long).unsqueeze(0).to(device)
            logits = model(ids)  # (1, L, 2)
            preds = logits.argmax(-1).cpu().numpy()[0]
            pred_positions = {i + 1 for i, p in enumerate(preds[:len(raw)]) if p == 1}
            true_positions = positions_from_char_labels(true_char_labels)
            f1s.append(f1_from_sets(pred_positions, true_positions))
    return float(np.mean(f1s)) if len(f1s) > 0 else 0.0


def main():
    print("Loading data", PATH_DATA)
    rows = read_jsonl(PATH_DATA)
    print("Total rows:", len(rows))
    random.shuffle(rows)
    n = len(rows)
    split = int(n * 0.95)
    train_rows = rows[:split]
    val_rows = rows[split:]

    vocab = CharVocab(min_freq=1)
    vocab.feed_texts([r["raw"] for r in train_rows])
    vocab.build()
    print("Vocab size:", len(vocab))

    train_ds = CharSpaceDataset(train_rows, vocab, max_len=MAX_LENGTH)
    val_ds = CharSpaceDataset(val_rows, vocab, max_len=MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              collate_fn=collate_fn_cnn)
    model = CharCNN(vocab_size=len(vocab), emb_dim=EMBED_DIM, conv_channels=CONV_CHANNELS, kernel_sizes=KERNEL_SIZES,
                    dropout=DROPOUT)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1 = -1.0
    outp = Path(OUT_DIR)
    outp.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, opt, criterion, DEVICE)
        val_f1 = evaluate_positions_f1(model, val_ds, DEVICE)
        print(f"Epoch {epoch} train_loss={train_loss:.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({"model": model.state_dict(), "vocab": vocab.ch2i}, outp / "best.pt")
            print("Saved best model, f1=", best_f1)

    print("Done. Best F1:", best_f1)


if __name__ == "__main__":
    main()
