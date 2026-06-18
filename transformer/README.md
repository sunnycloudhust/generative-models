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

Use the 10k file first to check that the pipeline works.

## Setup

Install PyTorch before running training. For CPU-only:

```bash
pip install torch
```

If you use CUDA, install the PyTorch build matching your CUDA version from the official PyTorch install guide.

## Train

Quick smoke test on 10k sentence pairs:

```bash
python3 main.py \
  --train-path data/en_vi_ted2020_10k.tsv \
  --min-freq 1 \
  --d-model 128 \
  --d-embed 128 \
  --d-ff 512 \
  --num-head 4 \
  --num-layer 2 \
  --batch-size 16 \
  --epochs 5
```

Train on the full TED2020 dataset:

```bash
python3 main.py \
  --train-path data/en_vi_ted2020.tsv \
  --min-freq 2 \
  --d-model 256 \
  --d-embed 256 \
  --d-ff 1024 \
  --num-head 8 \
  --num-layer 4 \
  --batch-size 32 \
  --epochs 10
```

By default, the best checkpoint is saved to:

```text
checkpoints/en_vi_transformer.pt
```

## Translate

After training, run:

```bash
python3 test.py \
  --checkpoint checkpoints/en_vi_transformer.pt \
  --sentence "I love machine learning"
```

## Important Arguments

```text
--train-path        Path to train data
--valid-path        Optional validation file
--valid-ratio       Validation split ratio if no valid file is given
--max-src-len       Max English sequence length
--max-tgt-len       Max Vietnamese sequence length
--src-vocab-size    Max English vocabulary size
--tgt-vocab-size    Max Vietnamese vocabulary size
--min-freq          Minimum token frequency kept in vocab
--d-model           Transformer hidden size
--d-embed           Embedding size before projection to d_model
--d-ff              Feed-forward hidden size
--num-head          Number of attention heads
--num-layer         Number of encoder/decoder layers
--batch-size        Batch size
--epochs            Number of training epochs
--lr                Learning rate
--checkpoint        Path to save the best checkpoint
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
