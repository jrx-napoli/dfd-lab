import argparse

import yaml

from src.data.fakeavceleb_preprocessor import FakeAVCelebPreprocessor


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
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to dataset metadata CSV file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fakeavceleb",
        choices=["fakeavceleb"],
        help="Dataset to preprocess"
    )
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create preprocessor based on dataset
    if args.dataset == "fakeavceleb":
        preprocessor = FakeAVCelebPreprocessor(config)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Process dataset
    print(f"Processing {args.dataset} dataset from {args.input_dir}")
    preprocessor.process_dataset(args.input_dir, args.output_dir, args.metadata)
    print(f"Processed data saved to {args.output_dir}")

if __name__ == "__main__":
    main() 