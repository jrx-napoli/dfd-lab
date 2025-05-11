import argparse

import yaml

from src.data.fakeavceleb_preprocessor import FakeAVCelebPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess deepfake detection dataset")
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
    with open("configs\\preprocessing.yaml", "r") as f:
        config = yaml.safe_load(f)

    input_dir = config["data"]["input_dir"]
    output_dir = config["data"]["output_dir"]

    # Create preprocessor based on dataset
    if args.dataset == "fakeavceleb":
        preprocessor = FakeAVCelebPreprocessor(config)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Process dataset
    print(f"Processing {args.dataset} dataset from {input_dir}")
    preprocessor.process_dataset(input_dir, output_dir)
    print(f"Processed data saved to {output_dir}")


if __name__ == "__main__":
    main()
