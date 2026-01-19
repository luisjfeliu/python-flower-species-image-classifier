import argparse
import os

import torch
from torch import nn, optim
from torchvision import datasets, models, transforms

AVAILABLE_ARCHS = {
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "densenet121": models.densenet121,
}


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data_dir", help="Path to dataset directory")
    parser.add_argument(
        "--save_dir",
        default=".",
        help="Directory to save checkpoints (default: current directory)",
    )
    parser.add_argument(
        "--arch",
        default="vgg16",
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


def build_model(arch: str, hidden_units: int) -> torch.nn.Module:
    model = AVAILABLE_ARCHS[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    if arch.startswith("vgg"):
        input_features = model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1),
        )
        model.classifier = classifier
    elif arch.startswith("densenet"):
        input_features = model.classifier.in_features
        classifier = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1),
        )
        model.classifier = classifier
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model


def load_data(data_dir: str):
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=eval_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=eval_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, train_loader, valid_loader, test_loader


def validate(model, loader, criterion, device):
    model.eval()
    loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += criterion(output, labels).item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.float)).item()
    return loss / len(loader), accuracy / len(loader)


def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{epochs}.. Train loss: {running_loss / len(train_loader):.3f}.. Validation loss: {val_loss:.3f}.. Validation accuracy: {val_accuracy:.3f}"
        )


def save_checkpoint(model, args):
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")
    checkpoint = {
        "arch": args.arch,
        "hidden_units": args.hidden_units,
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def run(args: argparse.Namespace) -> None:
    train_data, train_loader, valid_loader, _ = load_data(args.data_dir)
    model = build_model(args.arch, args.hidden_units)
    model.class_to_idx = train_data.class_to_idx

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    if args.gpu and device.type != "cuda":
        print("GPU requested but not available; using CPU.")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train(model, train_loader, valid_loader, criterion, optimizer, device, args.epochs)
    save_checkpoint(model, args)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train a new network")
    add_args(parser)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
