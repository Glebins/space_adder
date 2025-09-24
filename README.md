# Space adder

Here, models were created that insert spaces in the correct places in Russian sentences.
This project was done as part of a test assignment during the selection process for an internship at Avito.

# Datasets

Every row has the following pattern:
clean: original sentence, raw: sentence with spaces removed, char_labels: char_labels[i] = 1 if space goes after i-th char

Were downloaded from HF: wikimedia/wikipedia - russian wiki-articles (for clean-official russian lang),
Den4ikAI/russian_dialogues (for colloquial russian lang)

# Pretrained tokenizer + BiLSTM

## Architecture

- BPE fast tokenizer
- Embedding (B, L) -> (B, L, E)
- BiLSTM (input_size=E, hidden_size=H, num_layers=N, dropout=P, bidirectorial=True) -> (B, L, 2 * H)
- Dropout (p=P)
- Linear (B, L, 2 * H) -> (B, L, 2)

Where
- B = 16
- L = 256
- E = 128
- H = 256
- N = 2
- P = 0.2

For every BPE token, we predict whether space shall be placed after it or not.

## Performance

Accuracy after 1 epoch of training on RU Wiki dataset (95 000 rows for training, 5 000 for validation).
Here we are going to use pretrained tokenizer only. Using model is very costly.

|              Model              | F1 score | Loss in the end |
|:-------------------------------:|:--------:|:---------------:|
|          roberta-base           |  0.950   |      0.079      |
|        xlm-roberta-base         |  0.915   |      0.159      |
| blinoff/roberta-base-russian-v0 |  0.861   |      0.157      |
|       deepvk/roberta-base       |  0.822   |      0.147      |


The other experiments have shown that f1 of different models increase at the same pace as when more epochs are added. 
But I was eager to know which performance I would get if I used pretrained model as well (did it here: https://colab.research.google.com/drive/1wP9q2vaBJkXuf3G7KTZgtQ0Nkxjux1_y?usp=sharing).
Here are the results after 1 epoch.
Model is roberta-base

|             Type             | F1 score | Training time |
|:----------------------------:|:--------:|:-------------:|
|   tokenizer + custom model   |  0.950   |    6 mins     |
| tokenizer + pretrained model |  0.978   |    73 mins    |

The authors asked for a tiny model. Pretrained model doesn't show oVeRwHeLmInG superiority over smaller models, so it won't be used.

Note: I understand that removing BPE tokenizer may increase the model performance, but it won't be that significant, so leave it as it is for ~~showing off~~ variety.

# Char-CNN

## Architecture

- Embedding (B, L) -> (B, L, E)
- Parallel: [Conv1d(E -> C, kernel_size=k, padding=(k-1)//2), ReLU(), BatchNorm1d()] for k in [1, 3, 5, 7] -> each one produces (B, C, L)
- Concat: (B, C_total, L), C_total = 4 * C
- Conv1d(C_total -> C_total, kernel_size=3, padding=1) -> (B, C_total, L)
- Dropout(p=P)
- Linear: (B, C_total, L) -> permute -> (B, L, 2)

Where:

- B = 32
- L = 192 (in the end, I decided to increase it to 256 but all metrics were measured already, so I'll leave it 192)
- E = 64
- P = 0.1

Char-level classification: 1 if space after current char is needed, 0 — if not

## Performance

Wiki dataset (100 000 rows) is used as well.

| Epoch | F1 score |
|:-----:|:--------:|
|   1   |  0.947   |
|   2   |  0.956   |
|   3   |  0.962   |
|   4   |  0.964   |
|   5   |  0.967   |

The time of training for 1 epoch is around 90 – 95 seconds (150 for L = 256).

# Final measures

After merging datasets, we got 200k rows. We used 95% (190k) for training and 5% (10k) for validation.
For both BiLSTM and CNN models 3 epochs were used.

The following results were obtained:

| Epoch | Val F1 - BiLSTM | Val F1 - CNN |
|:-----:|:---------------:|:------------:|
|   1   |      0.946      |    0.947     |
|   2   |      0.954      |    0.955     |
|   3   |      0.957      |    0.962     |

Time spent on training: 34m 12s vs 13m 26s
Time spent on validation: 8m 50s vs 51s
Total: 43m 02s vs 14m 17s

Then we build an ensemble out of these models (keep in mind that there are more sophisticated ways of doing it):

|               Type                | F1 score |
|:---------------------------------:|:--------:|
| Union (ensemble = set_a OR set_b) |  0.978   |
|  Intersection = set_a AND set_b   |  0.949   |

So we'll use union.

# Predictions

The first submission has F1 score = 0.933. The second — 0.932. It took 20 seconds to generate them.</br>
In the second solution I put spaces explicitly before and after dashes.
It's surprising that F1 stayed almost the same. So let it be so. This block of code is framed in "ensemble_predict.py"

---
Files, folders, and their functions:

- out_bpe_bilstm, out_char_cnn contain weights of models
- bpe.py - (BPE + BiLSTM architecture) + training + validation + saving weights
- char_cnn.py - (Char-CNN) + training + validation + saving weights
- compose_dataset.py - downloads and saves "ru wiki" and "russian dialogues" datasets
- dataset_dialogues.jsonl, dataset_wiki.jsonl — datasets actually. I know it's a bad practice, but what will you do to me?
- ensemble_predict.py contains generation submission.csv file - space adding
- ensemble_validation.py gives F1 score of two models' collaboration
- utils.py - miscellaneous funcs.

---
If you actually read all of this, I'm sorry.
