import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


class Vocab:
    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, texts, max_size=30000, min_freq=2):
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        for token, freq in counter.most_common():
            if freq < min_freq:
                break
            if token not in tokens:
                tokens.append(token)
            if len(tokens) >= max_size:
                break
        return cls(tokens)

    @classmethod
    def from_tokens(cls, tokens):
        return cls(tokens)

    def encode(self, text):
        return [self.stoi.get(token, UNK_IDX) for token in tokenize(text)]

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, BOS_IDX):
                continue
            tokens.append(self.itos[idx] if idx < len(self.itos) else UNK_TOKEN)
        return " ".join(tokens)


def load_parallel_data(path, src_col="en", tgt_col="vi"):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".tsv", ".txt"}:
        pairs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    if suffix == ".csv":
        pairs = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                if src_col in row and tgt_col in row:
                    pairs.append((row[src_col], row[tgt_col]))
                elif len(fieldnames) >= 2:
                    pairs.append((row[fieldnames[0]], row[fieldnames[1]]))
        return pairs

    if suffix == ".jsonl":
        pairs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                src = item.get(src_col) or item.get("src") or item.get("source")
                tgt = item.get(tgt_col) or item.get("tgt") or item.get("target")
                if src and tgt:
                    pairs.append((src, tgt))
        return pairs

    raise ValueError("Supported data formats: .tsv, .txt, .csv, .jsonl")


class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_src_len=80, max_tgt_len=80):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]

        src_ids = self.src_vocab.encode(src_text)[:self.max_src_len - 1]
        src_ids = src_ids + [EOS_IDX]

        tgt_ids = self.tgt_vocab.encode(tgt_text)[:self.max_tgt_len - 2]
        tgt_full = [BOS_IDX] + tgt_ids + [EOS_IDX]

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input": torch.tensor(tgt_full[:-1], dtype=torch.long),
            "tgt_label": torch.tensor(tgt_full[1:], dtype=torch.long),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def collate_translation_batch(batch):
    src = nn.utils.rnn.pad_sequence(
        [item["src"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX
    )
    tgt_input = nn.utils.rnn.pad_sequence(
        [item["tgt_input"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX
    )
    tgt_label = nn.utils.rnn.pad_sequence(
        [item["tgt_label"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX
    )
    return src, tgt_input, tgt_label
