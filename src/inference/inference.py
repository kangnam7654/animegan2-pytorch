import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms

from src.models import Generator  # Noqa

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = ArgumentParser(description="ONNX Inference for AnimeGAN2")
    parser.add_argument("--weight", type=str)
    parser.add_argument("--image", type=str, default="examples/ffhq_100.png")

    return parser.parse_args()


def main(args):
    model = Generator()
    if args.weight:
        model.load_state_dict(torch.load(args.weight, map_location="cpu"))

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # (C, H, W) [0, 1]
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # [-1, 1]
        ]
    )

    input_tensor = transform(image).unsqueeze(0)  # (1, 3, 256, 256)
    with torch.no_grad():
        output = model(input_tensor)
    output = output.squeeze(0).permute(1, 2, 0).numpy()
    output = ((output + 1) * 127.5).clip(0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imread("output.png", output)


if __name__ == "__main__":
    args = get_args()
    main(args)
