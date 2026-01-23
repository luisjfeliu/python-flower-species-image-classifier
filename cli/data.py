import os

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

try:
    from torchvision.transforms import v2 as transforms_v2
except ImportError:  # torchvision < 0.15
    transforms_v2 = None

_DTYPE_MAP = {
    "float16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
}

_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
_NORMALIZE_STD = [0.229, 0.224, 0.225]
_INFER_RESIZE = 256
_INFER_CROP = 224
_INFER_DTYPE = "float32"
_INFER_SCALE = True


def load_data(data_dir: str):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(_INFER_CROP),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_NORMALIZE_MEAN, _NORMALIZE_STD),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize(_INFER_RESIZE),
            transforms.CenterCrop(_INFER_CROP),
            transforms.ToTensor(),
            transforms.Normalize(_NORMALIZE_MEAN, _NORMALIZE_STD),
        ]
    )

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=eval_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=eval_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, train_loader, valid_loader, test_loader


def build_inference_transform_from_checkpoint(checkpoint: dict):
    cfg = checkpoint["input_transform"]
    dtype_str = cfg.get("dtype", "float32")
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype in checkpoint: {dtype_str}")

    if transforms_v2 is not None:
        return transforms_v2.Compose(
            [
                transforms_v2.Resize(size=cfg["resize"]),
                transforms_v2.CenterCrop(size=cfg["crop"]),
                transforms_v2.ToImage(),
                transforms_v2.ToDtype(_DTYPE_MAP[dtype_str], scale=cfg.get("scale", True)),
                transforms_v2.Normalize(
                    mean=cfg["normalize_mean"], std=cfg["normalize_std"]
                ),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(cfg["resize"]),
            transforms.CenterCrop(cfg["crop"]),
            transforms.ToTensor(),
            transforms.Normalize(cfg["normalize_mean"], cfg["normalize_std"]),
        ]
    )


def get_input_transform_config() -> dict:
    return {
        "resize": _INFER_RESIZE,
        "crop": _INFER_CROP,
        "dtype": _INFER_DTYPE,
        "scale": _INFER_SCALE,
        "normalize_mean": _NORMALIZE_MEAN,
        "normalize_std": _NORMALIZE_STD,
    }


def process_image(image_path: str, transform=None) -> np.ndarray:
    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(_INFER_RESIZE),
                transforms.CenterCrop(_INFER_CROP),
                transforms.ToTensor(),
                transforms.Normalize(_NORMALIZE_MEAN, _NORMALIZE_STD),
            ]
        )

    with Image.open(image_path).convert("RGB") as im:
        processed_image = transform(im)
        if isinstance(processed_image, torch.Tensor):
            image_array = processed_image.numpy()
        else:
            image_array = np.asarray(processed_image)

    return image_array
