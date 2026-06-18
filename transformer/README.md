# English-Vietnamese Transformer

This project implements a small encoder-decoder Transformer from scratch in PyTorch and uses it for English-to-Vietnamese machine translation.

The code is organized as a learning project: each Transformer component lives in its own file, and the training pipeline is split into dataset, train, test, and main entry-point files.

## Project Structure

```text
.
├── main.py                  # Training entry point
├── train.py                 # Model builder, train loop, validation loop, checkpoint saving
├── test.py                  # Load checkpoint and translate a sentence
├── dataset.py               # Tokenizer, vocab, dataset loader, collate function
├── config.json              # Default config for the full dataset training run
├── model/
│   ├── transformer.py       # Full Transformer wrapper
│   ├── encoder_decoder.py   # Encoder-decoder stack
│   ├── encoder.py           # Encoder block
│   ├── decoder.py           # Decoder block
│   ├── self_attention.py    # Multi-head self-attention
│   ├── cross_attention.py   # Multi-head cross-attention
│   └── preprocess.py        # Embedding + positional encoding
└── data/
    ├── en_vi_ted2020.tsv        # Full OPUS TED2020 EN-VI dataset
    ├── en_vi_ted2020_10k.tsv    # Small 10k sample for quick experiments
    └── en_vi_ted2020.zip        # Original downloaded zip
```

## Dataset

The dataset used here is OPUS TED2020 English-Vietnamese.

The training pipeline expects parallel data in one of these formats:

- `.tsv` or `.txt`: first column is English, second column is Vietnamese
- `.csv`: defaults to columns `en` and `vi`
- `.jsonl`: defaults to keys `en` and `vi`, or fallback keys `src/source` and `tgt/target`

Example `.tsv` row:

```text
I love machine learning	Tôi thích học máy
```

Two ready-to-use files are available after dataset download:

```text
data/en_vi_ted2020_10k.tsv
data/en_vi_ted2020.tsv
```

Use the full file for the default training config. The 10k file is useful for quick smoke tests.

## Setup

Install PyTorch before running training. For CPU-only:

```bash
pip install torch
```

If you use CUDA, install the PyTorch build matching your CUDA version from the official PyTorch install guide.

## Train

Train with the default full-dataset config:

```bash
python3 main.py
```

By default, `main.py` reads:

```text
config.json
```

Change the number of epochs in that config file:

```json
{
  "epochs": 5
}
```

You can also pass another config file:

```bash
python3 main.py --config config.json
```

## W&B Logging

Weights & Biases logging is optional and disabled by default.

Install it:

```bash
pip install wandb
```

Login once:

```bash
wandb login
```

Enable W&B in `config.json`:

```json
{
  "use_wandb": true,
  "wandb_project": "en-vi-transformer",
  "wandb_run_name": "train-10k"
}
```

When enabled, the training loop logs loss only:

```text
train/loss
val/loss
```

Train on the full TED2020 dataset:

```json
{
  "train_path": "data/en_vi_ted2020.tsv",
  "epochs": 10,
  "d_model": 256,
  "d_embed": 256,
  "d_ff": 1024,
  "num_head": 8,
  "num_layer": 4,
  "batch_size": 32
}
```

With the default full-dataset config, the best checkpoint is saved to:

```text
checkpoints/en_vi_transformer_full.pt
```

## Translate

After training, run:

```bash
python3 test.py \
  --checkpoint checkpoints/en_vi_transformer.pt \
  --sentence "I love machine learning"
```

## Important Config Fields

```text
train_path        Path to train data
valid_path        Optional validation file
valid_ratio       Validation split ratio if no valid file is given
max_src_len       Max English sequence length
max_tgt_len       Max Vietnamese sequence length
src_vocab_size    Max English vocabulary size
tgt_vocab_size    Max Vietnamese vocabulary size
min_freq          Minimum token frequency kept in vocab
d_model           Transformer hidden size
d_embed           Embedding size before projection to d_model
d_ff              Feed-forward hidden size
num_head          Number of attention heads
num_layer         Number of encoder/decoder layers
batch_size        Batch size
epochs            Number of training epochs
lr                Learning rate
checkpoint        Path to save the best checkpoint
use_wandb         Enable or disable W&B logging
wandb_project     W&B project name
wandb_entity      Optional W&B team/user entity
wandb_run_name    Optional run name
wandb_mode        Optional W&B mode, e.g. offline
wandb_watch       Log gradients with wandb.watch
```

## Current Limitations

- Tokenization is simple word-level regex tokenization, not BPE/SentencePiece.
- Padding mask is not fully wired through encoder and cross-attention yet.
- Decoding uses greedy search, not beam search.
- This is a learning implementation, so it prioritizes clarity over speed.

Good next improvements:

- Add source padding masks.
- Replace word-level vocab with BPE or SentencePiece.
- Add BLEU evaluation.
- Add beam search decoding.
- Add resume training from checkpoint.
