import numpy as np
import torch
from torchvision.transforms import v2
import io
from PIL import Image
from fastapi import FastAPI
from fastapi import UploadFile, File
from pydantic import BaseModel
from models.generator import Generator

import uvicorn

# Define Transform
T = v2.Compose(
    [
        v2.Resize((256, 256), interpolation=v2.InterpolationMode.LANCZOS),
        v2.ToTensor(),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Model Load
model = Generator()
model.eval()

app = FastAPI()


def imread(image: UploadFile):
    image = Image.open(io.BytesIO(image)).convert("RGB")
    return image


def preprocess_image(image):
    processed_image = T(image)
    return processed_image.unsqueeze(0)


def postprocess_image(data):
    data = data.squeeze()
    data = ((data + 1) * 127.5).to(torch.uint8)
    data = torch.permute(data, (1, 2, 0))
    data = data.detach().cpu().numpy()
    data = data.tobytes()
    return data


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    bytes = await file.read()
    image = imread(bytes)
    image = preprocess_image(image)
    prediction = model(image)
    prediction = postprocess_image(prediction)
    return {"prediction": prediction}
