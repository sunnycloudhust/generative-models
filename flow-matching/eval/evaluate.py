from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
try:
    from torchvision.models import Inception3_Weights, inception_v3
except ImportError:
    from torchvision.models import inception_v3

    Inception3_Weights = None

from config import CONFIG
from sample import load_model
from train import sample


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
INCEPTION_MEAN = [0.485, 0.456, 0.406]
INCEPTION_STD = [0.229, 0.224, 0.225]
DATA_ROOT = Path(CONFIG["data_root"])
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT
CHECKPOINT_PATH = PROJECT_ROOT / "runs/celeba_flow/model_final.pt"
NUM_SAMPLES = 1000
BATCH_SIZE = CONFIG["batch_size"]
SAMPLE_STEPS = CONFIG["sample_steps"]


class EvaluationDataset(Dataset):
    def __init__(self, root, transform):
        root = Path(root)
        self.paths = sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(self.paths) < 2:
            raise RuntimeError(f"At least two images are required in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)


def get_features(images, model, device):
    return model(images.to(device)).detach().cpu().double()


def calculate_statistics(features):
    return features.mean(dim=0), torch.cov(features.T) #return mean, covariance


def calculate_fid(real_features, fake_features):
    real_mean, real_covariance = calculate_statistics(real_features)
    fake_mean, fake_covariance = calculate_statistics(fake_features)
    mean_difference = real_mean - fake_mean
    covariance_product = real_covariance @ fake_covariance
    eigenvalues = torch.linalg.eigvals(covariance_product).real.clamp_min(0)
    covariance_distance = torch.trace(real_covariance + fake_covariance)
    covariance_distance -= 2 * eigenvalues.sqrt().sum()
    return (mean_difference @ mean_difference + covariance_distance).item()


@torch.inference_mode()
def main():
    device_name = CONFIG["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if Inception3_Weights is None:
        feature_model = inception_v3(pretrained=True, transform_input=False)
    else:
        feature_model = inception_v3(
            weights=Inception3_Weights.DEFAULT, transform_input=False
        )
    feature_model.fc = torch.nn.Identity()
    feature_model.eval().to(device)

    real_transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(INCEPTION_MEAN, INCEPTION_STD),
        ]
    )
    dataset = EvaluationDataset(DATA_ROOT, real_transform)
    num_samples = min(NUM_SAMPLES, len(dataset))
    if num_samples < 2:
        raise RuntimeError("NUM_SAMPLES must be at least 2 for FID evaluation")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    real_features = []
    for images in loader:
        real_features.append(get_features(images, feature_model, device))
    real_features = torch.cat(real_features)[:num_samples]

    model = load_model(CHECKPOINT_PATH, CONFIG, device).eval()
    fake_features = []
    remaining = num_samples
    resize = transforms.Resize((299, 299))
    normalize = transforms.Normalize(INCEPTION_MEAN, INCEPTION_STD)
    while remaining > 0:
        current_batch_size = min(BATCH_SIZE, remaining)
        images = sample(
            model=model,
            device=device,
            n_samples=current_batch_size,
            image_size=CONFIG["image_size"],
            steps=SAMPLE_STEPS,
        )
        images = normalize(resize((images + 1) * 0.5))
        fake_features.append(get_features(images, feature_model, device))
        remaining -= current_batch_size

    fid = calculate_fid(real_features, torch.cat(fake_features))
    print(f"FID ({num_samples} samples): {fid:.4f}")


if __name__ == "__main__":
    main()