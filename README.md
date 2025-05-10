# Deepfake Detection Lab

A modular platform for training, evaluating, and comparing deepfake detectors across different modalities. Currently focused on image-based detection with plans to extend to video and audio modalities.

## Project Structure

```
deepfake-detection-lab/
├── configs/                    # Configuration files
│   ├── preprocessing.yaml      # Data preprocessing configurations
│   ├── training.yaml          # Model training configurations
│   └── evaluation.yaml        # Evaluation configurations
├── src/
│   ├── data/                  # Data handling modules
│   │   ├── __init__.py
│   │   ├── dataset.py        # Dataset classes
│   │   ├── preprocessing.py   # Data preprocessing utilities
│   │   └── transforms.py      # Data transformations
│   ├── models/               # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py          # Base model interface
│   │   └── detectors/       # Different detector implementations
│   ├── training/            # Training related code
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training loop
│   │   └── metrics.py       # Evaluation metrics
│   └── utils/               # Utility functions
│       ├── __init__.py
│       ├── config.py        # Configuration management
│       └── logging.py       # Logging utilities
├── scripts/                 # Executable scripts
│   ├── preprocess.py        # Data preprocessing script
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── tests/                  # Unit tests
├── requirements.txt        # Project dependencies
└── setup.py               # Package setup file
```

## Features

- Modular architecture for easy extension to different modalities
- Configurable preprocessing pipeline
- Flexible model training framework
- Comprehensive evaluation suite
- Support for various deepfake detection models

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing
```bash
python scripts/preprocess.py --config configs/preprocessing.yaml
```

### Training
```bash
python scripts/train.py --config configs/training.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --config configs/evaluation.yaml
```

## Configuration

The project uses YAML configuration files to manage different aspects:
- `preprocessing.yaml`: Data preprocessing parameters
- `training.yaml`: Model training parameters
- `evaluation.yaml`: Evaluation parameters

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.