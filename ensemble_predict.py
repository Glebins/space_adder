"""
Generates submission for sending to the platform
"""

import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import torch
from char_cnn import CharCNN, CharVocab
from bpe import TokenBPEClassifier
from transformers import AutoTokenizer
from utils import predict_positions_for_raw_bpe

INPUT_CSV = "dataset_test_1.txt"
OUTPUT_CSV = "submission_1.csv"
BPE_MODEL_DIR = "out_bpe_bilstm"
CNN_MODEL_PATH = "out_char_cnn/best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 256

CHAR_EMB_DIM = 64
CHAR_CONV_CHANNELS = 256
CHAR_KERNEL_SIZES = [1,3,5,7]
CHAR_DROPOUT = 0.1

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

bpe_model = TokenBPEClassifier(vocab_size=tokenizer.vocab_size)
bpe_ckpt_path = Path(BPE_MODEL_DIR) / "best_model.pt"
bpe_ckpt = torch.load(bpe_ckpt_path, map_location=DEVICE)
if isinstance(bpe_ckpt, dict) and "model_state_dict" in bpe_ckpt:
    bpe_model.load_state_dict(bpe_ckpt["model_state_dict"])
else:
    bpe_model.load_state_dict(bpe_ckpt)
bpe_model.to(DEVICE).eval()

cnn_ckpt_path = Path(CNN_MODEL_PATH)
cnn_ckpt = torch.load(cnn_ckpt_path, map_location=DEVICE)

if "vocab" in cnn_ckpt:
    char_vocab = CharVocab()
    char_vocab.ch2i = cnn_ckpt["vocab"]
    char_vocab.i2ch = {i: c for c, i in char_vocab.ch2i.items()}
else:
    raise RuntimeError("CharCNN checkpoint missing vocab")

cnn_model = CharCNN(
    vocab_size=len(char_vocab),
    emb_dim=CHAR_EMB_DIM,
    conv_channels=CHAR_CONV_CHANNELS,
    kernel_sizes=CHAR_KERNEL_SIZES,
    dropout=CHAR_DROPOUT
)

if isinstance(cnn_ckpt, dict):
    if "model_state_dict" in cnn_ckpt:
        cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
    elif "model" in cnn_ckpt:
        cnn_model.load_state_dict(cnn_ckpt["model"])
    else:
        cnn_model.load_state_dict(cnn_ckpt)
    if "vocab" in cnn_ckpt:
        char_vocab = CharVocab()
        char_vocab.ch2i = cnn_ckpt["vocab"]
        char_vocab.i2ch = {i: c for c, i in char_vocab.ch2i.items()}
    else:
        raise RuntimeError("CharCNN checkpoint missing vocab")
else:
    raise RuntimeError("Unknown CharCNN checkpoint format")
cnn_model.to(DEVICE).eval()

def predict_positions_for_raw_cnn_local(model: torch.nn.Module, vocab_obj: CharVocab,
                                        raw: str, device: str, max_length: int):
    ids = vocab_obj.encode(raw, max_length)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_tensor)
        preds = logits.argmax(dim=-1).cpu().numpy()[0]
    Lraw = min(len(raw), max_length)
    return {i+1 for i, p in enumerate(preds[:Lraw]) if int(p)==1}

def ensemble_predict(raw: str):
    bpe_positions = predict_positions_for_raw_bpe(bpe_model, tokenizer, raw, DEVICE, MAX_LENGTH)
    char_positions = predict_positions_for_raw_cnn_local(cnn_model, char_vocab, raw, DEVICE, MAX_LENGTH)

    positions = bpe_positions | char_positions

    # ------------------------ !!! Explicit spaces before and after dashes: -----------------

    for i, ch in enumerate(raw):
        if ch == "-":
            if i > 0:
                positions.add(i)
            positions.add(i + 1)

    # --------- todo You may delete it without breaking anything ----------------------------

    return sorted(list(positions))

rows = []
with open(INPUT_CSV, encoding="utf8") as f:
    next(f)
    for line in f:
        line = line.strip()
        if not line:
            continue
        first_comma = line.find(",")
        row_id = line[:first_comma]
        text = line[first_comma + 1 :]
        rows.append({"id": row_id, "text_no_spaces": text})
df = pd.DataFrame(rows)

if "id" not in df.columns or "text_no_spaces" not in df.columns:
    raise ValueError("Input CSV must contain columns: id, text_no_spaces")

out_rows = []
for _id, text in tqdm(zip(df["id"], df["text_no_spaces"]), total=len(df), desc="Predicting"):
    text = "" if pd.isna(text) else str(text)
    positions = ensemble_predict(text)
    out_rows.append({"id": _id, "predicted_positions": json.dumps(positions)})

out_df = pd.DataFrame(out_rows)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}. Rows: {len(out_df)}")
