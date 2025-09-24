"""
Different helping functions
"""

import json, random
from typing import List, Dict, Any, Set
import torch
import torch.nn as nn


def read_jsonl(paths) -> List[Dict[str, Any]]:
    items = []
    for path in paths:
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))

    random.shuffle(items)
    return items


def f1_from_sets(pred: Set[int], true: Set[int]) -> float:
    if len(pred) == 0 and len(true) == 0: return 1.0
    if len(pred) == 0 or len(true) == 0: return 0.0
    tp = len(pred & true)
    p = tp / len(pred) if len(pred) > 0 else 0.0
    r = tp / len(true) if len(true) > 0 else 0.0
    if p + r == 0: return 0.0
    return 2 * p * r / (p + r)


def positions_from_char_labels(char_labels: List[int]) -> Set[int]:
    return {i + 1 for i, v in enumerate(char_labels) if v == 1}


def collate_fn_cnn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    raws = [b["raw"] for b in batch]
    char_labels = [b["char_labels"] for b in batch]
    return {"input_ids": input_ids, "labels": labels, "raw": raws, "char_labels": char_labels}


def collate_fn_bilstm(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)

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


def split_data(items: List[Dict[str, Any]], val_frac=0.05, shuffle=True):
    if shuffle:
        random.shuffle(items)
    nval = int(len(items) * val_frac)
    if nval == 0:
        return items, []
    return items[nval:], items[:nval]


def build_token_labels_for_raw(tokenizer, raw: str, char_labels: List[int], max_length: int):
    enc = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
    input_ids_no_special = enc["input_ids"]
    offsets = enc["offset_mapping"]
    true_char_positions = {i for i, v in enumerate(char_labels) if v == 1}
    token_labels = []
    for (st, ed) in offsets:
        if ed > 0 and (ed - 1) in true_char_positions:
            token_labels.append(1)
        else:
            token_labels.append(0)

    num_special = tokenizer.num_special_tokens_to_add(False)
    prefix_special = 1 if tokenizer.cls_token_id is not None else 0
    suffix_special = num_special - prefix_special
    labels_with_special = ([-100] * prefix_special) + token_labels + ([-100] * suffix_special)

    input_ids_with_special = tokenizer.build_inputs_with_special_tokens(input_ids_no_special)
    attention_mask = [1] * len(input_ids_with_special)

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


def predict_positions_for_raw_bpe(model, tokenizer, raw: str, device, max_len) -> Set[int]:
    enc = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
    input_ids_no_special = enc["input_ids"]
    offsets = enc["offset_mapping"]
    if len(input_ids_no_special) == 0:
        return set()
    input_ids_with_special = tokenizer.build_inputs_with_special_tokens(input_ids_no_special)
    if len(input_ids_with_special) > max_len:
        input_ids_with_special = input_ids_with_special[:max_len]
    else:
        pad_len = max_len - len(input_ids_with_special)
        input_ids_with_special = input_ids_with_special + [tokenizer.pad_token_id] * pad_len
    input_tensor = torch.tensor([input_ids_with_special], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_tensor)  # (1, L, C)
        logits = logits.cpu().numpy()[0]  # (L, C)
    prefix = 1 if tokenizer.cls_token_id is not None else 0
    token_logits = logits[prefix: prefix + len(input_ids_no_special)]
    token_preds = token_logits.argmax(axis=-1)
    predicted_positions = set()
    for tok_idx, pred in enumerate(token_preds):
        if pred == 1:
            st, ed = offsets[tok_idx]
            predicted_positions.add(ed)
    return predicted_positions


def predict_positions_for_raw_cnn(model: nn.Module, raw: str, device: str,
                                  char2idx: dict, max_length: int = 256) -> Set[int]:
    input_ids = [char2idx.get(c, char2idx.get("<unk>", 0)) for c in raw]
    input_ids = input_ids[:max_length]
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [char2idx.get("<pad>", 0)] * pad_len

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)  # (1, L)
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)  # (1, L, 2)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()[0]  # (L,)

    positions = set()
    for i, p in enumerate(preds[:len(raw)]):
        if p == 1:
            positions.add(i + 1)
    return positions
