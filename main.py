"""Root CLI entry point for training and prediction.

Subcommands:
- train
- predict

Usage:
  python main.py train <data_dir> [options]
  python main.py predict <image_path> <checkpoint> [options]

Examples:
  python main.py train flowers --arch vgg13 --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu
  python main.py train flowers --save_dir checkpoints
  python main.py predict flowers/test/1/image_06752.jpg flower_classifier.pth --top_k 3 --category_names cat_to_name.json
  python main.py predict flowers/test/1/image_06752.jpg flower_classifier.pth --gpu
"""

import argparse

from cli import predict as predict_module
from cli import train as train_module


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Flower species classifier CLI")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_module.add_args(train_parser)

    predict_parser = subparsers.add_parser("predict", help="Predict from an image")
    predict_module.add_args(predict_parser)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        parser.exit(2)
    if args.command == "train":
        train_module.run(args)
    elif args.command == "predict":
        predict_module.run(args)


if __name__ == "__main__":
    main()
