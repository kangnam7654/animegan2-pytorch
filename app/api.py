import io
import base64

import numpy as np
import torch
from PIL import Image
from fastapi import UploadFile, File, FastAPI
from torchvision.transforms import v2
import uvicorn

from models.generator import Generator


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
    data = torch.clip(data, -1, 1)
    data = data.permute(1, 2, 0)
    data = data.detach().cpu().numpy()
    data = (data + 1) * 127.5
    data = data.astype(np.uint8).tobytes()
    data = base64.b64encode(data).decode("utf-8")
    return data


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    bytes = await file.read()
    image = imread(bytes)
    image = preprocess_image(image)
    prediction = model(image)
    prediction = postprocess_image(prediction)
    return {"prediction": prediction}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
