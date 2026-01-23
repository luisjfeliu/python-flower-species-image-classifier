import os
from collections.abc import Callable

import torch
import torchvision
from torch import nn
from torchvision import models

try:
    from torchvision.models import list_models
except ImportError:  # Older torchvision versions may not have list_models.
    list_models = None

from cli.data import get_input_transform_config, process_image


def _get_available_archs() -> dict[str, Callable[..., nn.Module]]:
    if list_models is None:
        model_names = [
            name
            for name in dir(models)
            if name.islower() and callable(getattr(models, name))
        ]
    else:
        model_names = list_models(module=models)
    return {name: getattr(models, name) for name in model_names}


AVAILABLE_ARCHS = _get_available_archs()


def _classifier_input_features(classifier: nn.Module) -> int | None:
    if isinstance(classifier, nn.Linear):
        return classifier.in_features
    if isinstance(classifier, nn.Sequential):
        for module in classifier:
            if isinstance(module, nn.Linear):
                return module.in_features
    return None


def _classifier_output_features(classifier: nn.Module) -> int | None:
    if isinstance(classifier, nn.Linear):
        return classifier.out_features
    if isinstance(classifier, nn.Sequential):
        for module in reversed(classifier):
            if isinstance(module, nn.Linear):
                return module.out_features
    return None


def _build_classifier(
    input_features: int, hidden_units: int, num_classes: int = 102
) -> nn.Sequential:
    hidden_units2 = max(1, hidden_units // 2)
    return nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_units, hidden_units2),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_units2, num_classes),
    )


def build_model(arch: str, hidden_units: int) -> torch.nn.Module:
    model = AVAILABLE_ARCHS[arch](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "classifier"):
        input_features = _classifier_input_features(model.classifier)
        if input_features is None:
            raise ValueError(
                f"Unsupported architecture: {arch} (classifier head is not linear)"
            )
        model.classifier = _build_classifier(input_features, hidden_units)
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        input_features = model.fc.in_features
        model.fc = _build_classifier(input_features, hidden_units)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model


def validate(model, loader, loss_fn, device):
    model.eval()
    loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += loss_fn(output, labels).item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.float)).item()
    return loss / len(loader), accuracy / len(loader)


def train(model, train_loader, valid_loader, loss_fn, optimizer, device, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss, val_accuracy = validate(model, valid_loader, loss_fn, device)
        print(
            f"Epoch {epoch + 1}/{epochs}.. Train loss: {running_loss / len(train_loader):.3f}.. Validation loss: {val_loss:.3f}.. Validation accuracy: {val_accuracy:.3f}"
        )


def save_checkpoint(model, args, optimizer=None):
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, "checkpoint.pth")
    classifier = model.classifier if hasattr(model, "classifier") else model.fc
    in_features = _classifier_input_features(classifier)
    num_classes = _classifier_output_features(classifier)
    checkpoint = {
        "arch": args.arch,
        "framework": "pytorch",
        "torchvision_version": torchvision.__version__,
        "in_features": in_features,
        "num_classes": num_classes,
        "epochs_trained": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "class_to_idx": model.class_to_idx,
        "class_names": list(model.class_to_idx.keys()),
        "input_transform": get_input_transform_config(),
        "hidden_units": args.hidden_units,
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(path: str, checkpoint: dict | None = None) -> torch.nn.Module:
    if checkpoint is None:
        checkpoint = torch.load(path, map_location="cpu")

    arch = checkpoint.get("arch", "resnet50")
    if arch not in AVAILABLE_ARCHS:
        raise ValueError(f"Unsupported architecture in checkpoint: {arch}")

    if "state_dict" in checkpoint and "hidden_units" in checkpoint:
        hidden_units = checkpoint["hidden_units"]
        model = build_model(arch, hidden_units)
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        try:
            model = AVAILABLE_ARCHS[arch](weights=None)
        except TypeError:  # Older torchvision expects pretrained flag.
            model = AVAILABLE_ARCHS[arch](pretrained=False)

        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Sequential(
                nn.Linear(checkpoint["in_features"], 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, checkpoint["num_classes"]),
            )
        else:
            raise ValueError(f"Unsupported architecture in checkpoint: {arch}")

        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("Unsupported checkpoint format: missing model weights.")

    model.class_to_idx = checkpoint["class_to_idx"]
    return model


def predict(
    image_path: str,
    model: torch.nn.Module,
    topk: int,
    device: torch.device,
    transform=None,
):
    model.to(device)
    model.eval()
    with torch.no_grad():
        image_array = process_image(image_path, transform=transform)
        image = torch.from_numpy(image_array).unsqueeze(0).float().to(device)
        logits = model(image)
        ps = torch.softmax(logits, dim=1)
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
