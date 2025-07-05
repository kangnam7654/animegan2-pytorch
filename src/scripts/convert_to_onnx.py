import torch
from models.generator import Generator


def to_onnx():
    # Initialize the generator model
    model = Generator()
    state_dict = torch.load(
        "/home/kangnam/projects/animegan2/weights/face_paint_512_v1.pt"
    )
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode

    # Create a dummy input tensor with the appropriate shape
    dummy_input = torch.randn(1, 3, 256, 256)  # Example input shape

    # Define the output ONNX file path
    onnx_file_path = "generator.onnx"

    # Export the model to ONNX format
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        # export_params=True,
        # opset_version=11,
        # do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print(f"Model exported to {onnx_file_path}")


if __name__ == "__main__":
    to_onnx()
