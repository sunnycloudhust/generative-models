import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi


def build_hf_bundle(checkpoint_path: str, output_dir: str = "hf_bundle"):
    project_root = Path(__file__).resolve().parents[1]
    src = Path(checkpoint_path)
    if not src.is_absolute():
        src = project_root / src
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")

    dst = Path(output_dir)
    if not dst.is_absolute():
        dst = project_root / dst
    dst.mkdir(parents=True, exist_ok=True)

    modeling_src = project_root / "modeling.py"
    if not modeling_src.exists():
        modeling_src = project_root / "huggingface" / "modeling.py"
    if modeling_src.exists():
        shutil.copy2(modeling_src, dst / "modeling.py")

    readme_src = project_root / "huggingface" / "README.md"
    config_src = project_root / "huggingface" / "config.json"
    if readme_src.exists():
        shutil.copy2(readme_src, dst / "README.md")
    if config_src.exists():
        shutil.copy2(config_src, dst / "config.json")

    shutil.copy2(src, dst / "pytorch_model.bin")
    print(f"Prepared bundle at: {dst}")
    return dst


def upload_to_hf(repo_id: str, checkpoint_path: str, output_dir: str = "hf_bundle"):
    bundle = build_hf_bundle(checkpoint_path, output_dir)
    api = HfApi()
    api.upload_folder(
        folder_path=str(bundle),
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package and upload a flow-matching checkpoint to Hugging Face.")
    parser.add_argument("--repo-id", type=str, default="sunnycloudhust/CelebA-flow-matching", help="Example: your-username/celeba-flow-matching")
    parser.add_argument("--checkpoint", type=str, default="runs/celeba_flow/model_final.pt", help="Path to the trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="hf_bundle", help="Temporary folder used before upload")
    args = parser.parse_args()

    upload_to_hf(args.repo_id, args.checkpoint, args.output_dir)
