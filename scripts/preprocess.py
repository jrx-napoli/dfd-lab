import argparse
import yaml
from pathlib import Path
from src.data.preprocessing import DataPreprocessor

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess deepfake detection dataset")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to preprocessing configuration file"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input videos"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save processed data"
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Process dataset
    print(f"Processing dataset from {args.input_dir}")
    preprocessor.process_dataset(args.input_dir, args.output_dir)
    print(f"Processed data saved to {args.output_dir}")

if __name__ == "__main__":
    main() 