import cv2
import numpy as np
import torch
import gradio as gr
from torchvision.utils import save_image

from models.generator import Generator

model = Generator()
ckpt = torch.load("./weights/personai_cartoon.pt")["G"]
model.load_state_dict(ckpt)
model.eval()


def preprocess(image):
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    image = (image / 127.5) - 1
    return image


def tensor_to_image(image):
    return image


def inference(image):
    image = preprocess(image)
    logit = model(image)
    save_image(logit, "temp_save.jpg", normalize=True)
    return "temp_save.jpg"


if __name__ == "__main__":
    demo = gr.Interface(fn=inference, inputs=gr.Image(), outputs="image")
    demo.launch()
