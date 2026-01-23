import argparse

import torch
from torch import nn, optim

from cli.data import load_data
from cli.model import AVAILABLE_ARCHS, build_model, save_checkpoint, train


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data_dir", help="Path to dataset directory")
    parser.add_argument(
        "--save_dir",
        default=".",
        help="Directory to save checkpoints (default: current directory)",
    )
    parser.add_argument(
        "--arch",
        default="resnet50",
        choices=sorted(AVAILABLE_ARCHS.keys()),
        help="Model architecture",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=512,
        help="Hidden units in classifier",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")


def run(args: argparse.Namespace) -> None:
    train_data, train_loader, valid_loader, _ = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    model.class_to_idx = train_data.class_to_idx

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if args.gpu and device.type != "cuda":
        print("GPU requested but not available; using CPU.")

    loss_fn = nn.CrossEntropyLoss()
    classifier_params = (
        model.classifier.parameters()
        if hasattr(model, "classifier")
        else model.fc.parameters()
    )
    optimizer = optim.Adam(classifier_params, lr=args.learning_rate)

    train(model, train_loader, valid_loader, loss_fn, optimizer, device, args.epochs)
    save_checkpoint(model, args, optimizer=optimizer)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train a new network")
    add_args(parser)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
