import cv2
import numpy as np


def preprocess_image(image_bytes, target_size=(256, 256)):
    # Load image
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image data")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    image = image.astype(np.float32) / 127.5 - 1.0  # [0, 255] → [-1, 1]
    image = image.transpose(2, 0, 1)  # HWC → CHW
    image = np.expand_dims(image, axis=0)
    return image
