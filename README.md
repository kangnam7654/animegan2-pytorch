# AnimeGAN2 - PyTorch

PyTorch implementation of [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2) with improved project structure and modern practices.

*This code is forked from [bryandlee's animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)*

## Project Structure

```
animegan2/
├── animegan2/                    # Main package
│   ├── models/                   # Model definitions
│   ├── data/                     # Data loading and preprocessing
│   ├── training/                 # Training pipeline
│   ├── inference/                # Inference and model conversion
│   ├── utils/                    # Utility functions
│   └── serving/                  # Model serving
├── scripts/                      # Entry point scripts
├── configs/                      # Configuration files
├── tests/                        # Test files
├── deployment/                   # Deployment configurations
├── examples/                     # Examples and demos
└── weights/                      # Pre-trained models
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/animegan2.git
cd animegan2

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Training

#### Step 1: Prepare edge smoothing data
```bash
python scripts/edge_smooth.py --root_dir {ANIME_IMAGE_DIRECTORY} --out_dir {EDGE_SMOOTH_SAVE_DIRECTORY} --image_size {EDGE_SMOOTH_IMAGE_SIZE}
```

#### Step 2: Train the model
```bash
python scripts/train.py --photo_dir {PHOTO_IMAGE_DIRECTORY} --anime_dir {ANIME_IMAGE_DIRECTORY} --smooth_dir {SMOOTH_IMAGE_DIRECTORY}
```

### Inference

```bash
python scripts/inference.py
```

### Model Serving

```bash
python scripts/convert_to_onnx.py  # Convert to ONNX format
python -m animegan2.serving.app    # Start serving API
```

## Features

- ✅ Clean, modular project structure
- ✅ PyTorch Lightning integration
- ✅ ONNX model conversion and inference
- ✅ FastAPI serving endpoints
- ✅ Docker deployment support
- ✅ Kubernetes deployment configurations
- ✅ Comprehensive logging and monitoring
- ✅ Configuration management

## Trained Results

![Training Results](./examples/00025000.jpg)

## Requirements

- Python >= 3.8
- PyTorch >= 1.7.1
- PyTorch Lightning
- See `setup.py` for full requirements

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original AnimeGAN2 implementation: [TachibanaYoshino/AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2)
- PyTorch implementation: [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)