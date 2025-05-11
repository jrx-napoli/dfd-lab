import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import DeepfakeDataset
from src.models.base import BaseDetector
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train deepfake detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing processed dataset"
    )
    return parser.parse_args()

def create_model(config: dict) -> BaseDetector:
    """Create model based on configuration."""
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    
    if model_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=config["model"]["pretrained"])
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=config["model"]["pretrained"])
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def create_data_loaders(config: dict, data_dir: str) -> tuple:
    """Create training and validation data loaders."""
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split="train",
        data_format=config["data"]["format"]
    )
    val_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split="val",
        data_format=config["data"]["format"]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"]
    )
    
    return train_loader, val_loader

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config, args.data_dir)
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    print("Training completed!")

if __name__ == "__main__":
    main() 