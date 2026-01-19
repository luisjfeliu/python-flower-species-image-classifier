import argparse
import json

import torch
from PIL import Image

from cli.train import AVAILABLE_ARCHS, build_model


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Return top K most likely classes",
    )
    parser.add_argument(
        "--category_names",
        default=None,
        help="Path to JSON mapping of categories to names",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")


def process_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((256, 256))
    left = (256 - 224) / 2
    upper = (256 - 224) / 2
    right = left + 224
    lower = upper + 224
    image = image.crop((left, upper, right, lower))

    image = torch.tensor(list(image.getdata())).reshape((224, 224, 3))
    image = image.permute((2, 0, 1)).float() / 255.0

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = (image - mean) / std

    return image


def load_checkpoint(path: str) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu")
    arch = checkpoint["arch"]
    hidden_units = checkpoint["hidden_units"]

    if arch not in AVAILABLE_ARCHS:
        raise ValueError(f"Unsupported architecture in checkpoint: {arch}")

    model = build_model(arch, hidden_units)
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    return model


def predict(image_path: str, model: torch.nn.Module, topk: int, device: torch.device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = process_image(image_path).unsqueeze(0).to(device)
        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)

    top_p = top_p.squeeze().tolist()
    top_class = top_class.squeeze().tolist()

    if not isinstance(top_p, list):
        top_p = [top_p]
    if not isinstance(top_class, list):
        top_class = [top_class]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[idx] for idx in top_class]
    return top_p, top_labels


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if args.gpu and device.type != "cuda":
        print("GPU requested but not available; using CPU.")

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.image_path, model, args.top_k, device)

    if args.category_names:
        with open(args.category_names, encoding="utf-8") as handle:
            cat_to_name = json.load(handle)
        labels = [cat_to_name.get(cls, cls) for cls in classes]
    else:
        labels = classes

    for label, prob in zip(labels, probs, strict=False):
        print(f"{label}: {prob:.4f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    add_args(parser)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
