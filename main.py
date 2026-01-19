import argparse

from cli import predict as predict_module
from cli import train as train_module


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Flower classifier CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_module.add_args(train_parser)

    predict_parser = subparsers.add_parser("predict", help="Predict from an image")
    predict_module.add_args(predict_parser)

    args = parser.parse_args(argv)

    if args.command == "train":
        train_module.run(args)
    elif args.command == "predict":
        predict_module.run(args)


if __name__ == "__main__":
    main()
