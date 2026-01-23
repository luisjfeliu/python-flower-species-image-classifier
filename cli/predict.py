import argparse
import json

import torch

from cli.data import build_inference_transform_from_checkpoint
from cli.model import load_checkpoint, predict
from cli.plot import save_prediction_plot


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
    parser.add_argument(
        "--plot_path",
        default=None,
        help="Optional path to save prediction plot image",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if args.gpu and device.type != "cuda":
        print("GPU requested but not available; using CPU.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    transform = None
    if "input_transform" in checkpoint:
        transform = build_inference_transform_from_checkpoint(checkpoint)

    model = load_checkpoint(args.checkpoint, checkpoint=checkpoint)
    probs, classes = predict(args.image_path, model, args.top_k, device, transform=transform)

    if args.category_names:
        with open(args.category_names, encoding="utf-8") as handle:
            cat_to_name = json.load(handle)
        labels = [cat_to_name.get(cls, cls) for cls in classes]
    else:
        labels = classes

    for label, prob in zip(labels, probs, strict=False):
        print(f"{label}: {prob:.4f}")

    if args.plot_path:
        save_prediction_plot(args.image_path, probs, labels, args.plot_path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    add_args(parser)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
