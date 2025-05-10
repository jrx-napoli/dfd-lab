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
from src.training.metrics import MetricsCalculator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing test dataset"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    return parser.parse_args()

def load_model(config: dict, checkpoint_path: str) -> BaseDetector:
    """Load model from checkpoint."""
    # Create model
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    
    if model_name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model

def create_data_loader(config: dict, data_dir: str) -> DataLoader:
    """Create test data loader."""
    # Create dataset
    test_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split="test",
        data_format=config["data"]["format"]
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"]
    )
    
    return test_loader

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config["model"]["device"])
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(config, args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # Create data loader
    print("Creating data loader...")
    test_loader = create_data_loader(config, args.data_dir)
    
    # Create metrics calculator
    print("Initializing metrics calculator...")
    metrics_calculator = MetricsCalculator(config)
    
    # Evaluate model
    print("Evaluating model...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1)
            metrics_calculator.update(pred, target, probs)
    
    # Compute metrics
    metrics = metrics_calculator.compute()
    print("\nEvaluation metrics:")
    for metric, value in metrics.items():
        if metric != "confusion_matrix":
            print(f"{metric}: {value:.4f}")
    
    # Generate visualizations
    if config["evaluation"]["visualization"]["enabled"]:
        print("\nGenerating visualizations...")
        metrics_calculator.visualize(config["evaluation"]["visualization"]["save_dir"])
    
    # Perform error analysis
    if config["evaluation"]["error_analysis"]["enabled"]:
        print("\nPerforming error analysis...")
        misclassified = metrics_calculator.analyze_errors(test_loader, model, device)
        print(f"Found {len(misclassified)} misclassified samples")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main() 