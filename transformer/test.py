import argparse

import torch

from dataset import BOS_IDX, EOS_IDX, Vocab


@torch.no_grad()
def greedy_translate(model, sentence, src_vocab, tgt_vocab, device, max_len=80):
    model.eval()
    src_ids = src_vocab.encode(sentence)[:max_len - 1] + [EOS_IDX]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    tgt = torch.tensor([[BOS_IDX]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        logits = model(src, tgt)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        tgt = torch.cat(
            [tgt, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1
        )
        if next_id == EOS_IDX:
            break

    return tgt_vocab.decode(tgt[0].tolist())


def load_checkpoint(checkpoint_path, device):
    from train import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    src_vocab = Vocab.from_tokens(checkpoint["src_vocab"])
    tgt_vocab = Vocab.from_tokens(checkpoint["tgt_vocab"])

    model = build_model(
        len(src_vocab),
        len(tgt_vocab),
        config["d_model"],
        config["d_embed"],
        config["d_ff"],
        config["num_head"],
        config["num_layer"],
        config["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, src_vocab, tgt_vocab, config


def parse_args():
    parser = argparse.ArgumentParser(description="Translate English text with a trained EN-VI Transformer.")
    parser.add_argument("--checkpoint", default="checkpoints/en_vi_transformer.pt")
    parser.add_argument("--sentence", required=True)
    parser.add_argument("--max-len", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src_vocab, tgt_vocab, _ = load_checkpoint(args.checkpoint, device)
    translation = greedy_translate(
        model,
        args.sentence,
        src_vocab,
        tgt_vocab,
        device,
        args.max_len
    )
    print(translation)


if __name__ == "__main__":
    main()
