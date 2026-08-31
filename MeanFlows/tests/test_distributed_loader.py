from pathlib import Path

from data import build_loader


def test_build_loader_supports_distributed_sampler(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.png"
    image_path.write_bytes(b"fake png")

    loader = build_loader(
        image_dir,
        image_size=64,
        batch_size=2,
        workers=0,
        distributed=True,
        rank=0,
        world_size=2,
    )

    assert loader.sampler is not None
    assert hasattr(loader.sampler, "set_epoch")
