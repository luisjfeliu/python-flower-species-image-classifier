import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


def save_prediction_plot(image_path: str, probs, labels, output_path: str) -> None:
    image = Image.open(image_path).convert("RGB")

    fig, (ax_image, ax_bar) = plt.subplots(nrows=2, figsize=(6, 8))
    ax_image.imshow(image)
    ax_image.axis("off")

    ax_bar.barh(labels, probs)
    ax_bar.set_xlim(0, 1)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Probability")

    fig.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
