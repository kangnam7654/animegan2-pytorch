import base64
import logging
from pathlib import Path

import cv2
import numpy as np
import requests
import tritonclient.http as httpclient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def single_test(
    triton_client: httpclient.InferenceServerClient,
    image_path,
    preprocess_url,
    params=None,
    convert_to_bgr=True,
):
    if image_path.startswith("http"):
        # If the image path is a URL, fetch the image
        response = requests.get(image_path)
        response.raise_for_status()
        image_bytes = response.content
    else:
        # If the image path is a local file, read the file
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
    files = {"file": image_bytes}
    response = requests.post(preprocess_url, files=files, params=params)
    response.raise_for_status()
    base_64_image = response.json().get("data")
    decoded_image = base64.b64decode(base_64_image)

    image = np.frombuffer(decoded_image, np.float32)  # flattened array
    image = image.reshape(1, 3, 256, 256)

    input_tensor = httpclient.InferInput("input", image.shape, "FP32")
    input_tensor.set_data_from_numpy(image)
    result = triton_client.infer(model_name="animegan", inputs=[input_tensor])
    output = result.as_numpy("output")  # output shape: (1, 3, 256, 256)

    output_image = output[0].transpose(1, 2, 0)  # Convert to HWC format #type: ignore
    output_image = np.clip((output_image + 1) * 127.5, 0, 255).astype(
        np.uint8
    )  # Convert to [0, 255] range

    if convert_to_bgr:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    return output_image


def test_k8s_end_to_end(
    triton_url=None,
    preprocess_url=None,
    local_image_path=None,
    online_image_path=None,
):
    triton_client = httpclient.InferenceServerClient(triton_url)
    # Request parameters
    params = {
        "target_width": 256,
        "target_height": 256,
        "return_format": "base64",  # "base64" or "numpy"
    }

    # ==============
    # | Local file |
    # ==============
    local_image = single_test(
        triton_client=triton_client,
        image_path=local_image_path,
        preprocess_url=preprocess_url,
        params=params,
        convert_to_bgr=True,
    )
    # ===============
    # | Online file |
    # ===============
    online_image = single_test(
        triton_client=triton_client,
        image_path=online_image_path,
        preprocess_url=preprocess_url,
        params=params,
        convert_to_bgr=True,
    )

    save_dir = Path("test_result")
    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_dir.joinpath("local_image.png").as_posix(), local_image)
    cv2.imwrite(save_dir.joinpath("online_image.png").as_posix(), online_image)


if __name__ == "__main__":
    triton_url = "http://localhost:8000"
    preprocess_url = "http://localhost:9004/preprocess"

    local_image_path = "examples/ffhq_100.png"
    online_image_path = "https://efrosgans.eecs.berkeley.edu/SwappingAutoencoder/results_for_paper_with_new_ffhq2/ffhq/input_style/00001__001.png"

    test_k8s_end_to_end(
        triton_url=triton_url,
        preprocess_url=preprocess_url,
        local_image_path=local_image_path,
        online_image_path=online_image_path,
    )
