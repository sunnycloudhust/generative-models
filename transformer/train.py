import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PAD_IDX, TranslationDataset, Vocab, collate_translation_batch, load_parallel_data
from model.cross_attention import CrossAttention
from model.decoder import Decoder
from model.encoder import Encoder
from model.encoder_decoder import EncoderDecoder
from model.preprocess import Preprocess
from model.self_attention import SelfAttention
from model.transformer import Transformer


def build_model(
    src_vocab_size,
    tgt_vocab_size,
    d_model=256,
    d_embed=256,
    d_ff=1024,
    num_head=8,
    num_layer=4,
    dropout=0.1,
    bias=True
):
    src_preprocess = Preprocess(src_vocab_size, d_model, d_embed, bias, dropout)
    tgt_preprocess = Preprocess(tgt_vocab_size, d_model, d_embed, bias, dropout)

    encoder = Encoder(
        SelfAttention(d_model, num_head, bias, dropout),
        d_model,
        d_ff,
        num_head,
        bias,
        dropout
    )
    decoder = Decoder(
        SelfAttention(d_model, num_head, bias, dropout),
        CrossAttention(d_model, num_head, bias, dropout),
        d_model,
        d_ff,
        num_head,
        bias,
        dropout
    )
    encoder_decoder = EncoderDecoder(encoder, decoder, num_layer)

    return Transformer(
        src_preprocess,
        tgt_preprocess,
        encoder_decoder,
        d_model,
        tgt_vocab_size,
        bias
    )


def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0

    for src, tgt_input, tgt_label in dataloader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_label = tgt_label.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(src, tgt_input)
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            tgt_label.reshape(-1)
        )
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    for src, tgt_input, tgt_label in dataloader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_label = tgt_label.to(device)

        logits = model(src, tgt_input)
        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            tgt_label.reshape(-1)
        )
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def save_checkpoint(path, model, optimizer, src_vocab, tgt_vocab, config, epoch, val_loss):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "src_vocab": src_vocab.itos,
            "tgt_vocab": tgt_vocab.itos,
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path
    )


def init_wandb(args, config, model):
    if not getattr(args, "use_wandb", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("W&B is enabled. Install it with: pip install wandb") from exc

    init_kwargs = {
        "project": getattr(args, "wandb_project", "en-vi-transformer"),
        "name": getattr(args, "wandb_run_name", None),
        "config": config,
    }

    wandb_entity = getattr(args, "wandb_entity", None)
    wandb_mode = getattr(args, "wandb_mode", None)
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    if wandb_mode:
        init_kwargs["mode"] = wandb_mode

    run = wandb.init(**init_kwargs)
    if getattr(args, "wandb_watch", False):
        wandb.watch(
            model,
            log="gradients",
            log_freq=getattr(args, "wandb_log_freq", 100)
        )

    return run


def prepare_dataloaders(args):
    pairs = load_parallel_data(args.train_path, args.src_col, args.tgt_col)
    if not pairs:
        raise ValueError("No sentence pairs found in train data.")

    if args.valid_path:
        train_pairs = pairs
        valid_pairs = load_parallel_data(args.valid_path, args.src_col, args.tgt_col)
    else:
        random.shuffle(pairs)
        valid_size = max(1, int(len(pairs) * args.valid_ratio))
        valid_pairs = pairs[:valid_size]
        train_pairs = pairs[valid_size:]

    src_vocab = Vocab.build(
        [src for src, _ in train_pairs],
        max_size=args.src_vocab_size,
        min_freq=args.min_freq
    )
    tgt_vocab = Vocab.build(
        [tgt for _, tgt in train_pairs],
        max_size=args.tgt_vocab_size,
        min_freq=args.min_freq
    )

    train_dataset = TranslationDataset(
        train_pairs,
        src_vocab,
        tgt_vocab,
        args.max_src_len,
        args.max_tgt_len
    )
    valid_dataset = TranslationDataset(
        valid_pairs,
        src_vocab,
        tgt_vocab,
        args.max_src_len,
        args.max_tgt_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_translation_batch
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_translation_batch
    )

    return train_loader, valid_loader, src_vocab, tgt_vocab, train_pairs, valid_pairs


def run_training(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, valid_loader, src_vocab, tgt_vocab, train_pairs, valid_pairs = prepare_dataloaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        len(src_vocab),
        len(tgt_vocab),
        args.d_model,
        args.d_embed,
        args.d_ff,
        args.num_head,
        args.num_layer,
        args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = vars(args).copy()
    config["actual_src_vocab_size"] = len(src_vocab)
    config["actual_tgt_vocab_size"] = len(tgt_vocab)
    config["device"] = str(device)

    best_val_loss = float("inf")
    wandb_run = init_wandb(args, config, model)

    print(f"device: {device}")
    print(f"train pairs: {len(train_pairs)} | valid pairs: {len(valid_pairs)}")
    print(f"src vocab: {len(src_vocab)} | tgt vocab: {len(tgt_vocab)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.grad_clip
        )
        val_loss = evaluate_loss(model, valid_loader, criterion, device)

        print(
            f"epoch {epoch:02d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                src_vocab,
                tgt_vocab,
                config,
                epoch,
                val_loss
            )
            print(f"saved checkpoint: {args.checkpoint}")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                },
                step=epoch
            )

    if wandb_run is not None:
        wandb_run.finish()

    return model, src_vocab, tgt_vocab
