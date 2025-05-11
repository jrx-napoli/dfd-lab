# Deepfake Detection Lab

A modular platform for training, evaluating, and comparing deepfake detectors across different modalities. Currently focused on image-based detection with plans to extend to video and audio modalities.

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
python preprocess.py --config configs/preprocessing.yaml
```

### Training
```bash
python train.py --config configs/training.yaml
```

### Evaluation
```bash
python evaluate.py --config configs/evaluation.yaml
```

## Configuration

The project uses YAML configuration files to manage different aspects:
- `preprocessing.yaml`: Data preprocessing parameters
- `training.yaml`: Model training parameters
- `evaluation.yaml`: Evaluation parameters
