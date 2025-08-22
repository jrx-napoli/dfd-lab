"""
Utility functions for working with LMDB-stored preprocessed data.
Provides efficient data loading for training and evaluation.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.lmdb_storage import LMDBStorage, LMDBDataset


def load_lmdb_dataset(db_path: str, transform=None) -> LMDBDataset:
    """Load a dataset from LMDB storage.
    
    Args:
        db_path: Path to the LMDB database
        transform: Optional transform to apply to samples
        
    Returns:
        LMDBDataset instance
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"LMDB database not found: {db_path}")
    
    return LMDBDataset(db_path, transform=transform)


def create_data_loader(
    db_path: str, 
    batch_size: int = 32, 
    shuffle: bool = True, 
    num_workers: int = 4,
    transform=None,
    **kwargs
) -> DataLoader:
    """Create a PyTorch DataLoader from LMDB storage.
    
    Args:
        db_path: Path to the LMDB database
        batch_size: Batch size for the data loader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        transform: Optional transform to apply to samples
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
    dataset = load_lmdb_dataset(db_path, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )


def get_dataset_statistics(db_path: str) -> Dict[str, Any]:
    """Get statistics about the dataset stored in LMDB.
    
    Args:
        db_path: Path to the LMDB database
        
    Returns:
        Dictionary containing dataset statistics
    """
    with LMDBStorage(db_path) as storage:
        return storage.get_dataset_info()


def query_samples_by_label(db_path: str, label: int) -> List[str]:
    """Query samples by their label.
    
    Args:
        db_path: Path to the LMDB database
        label: Label to search for (0 for real, 1 for fake)
        
    Returns:
        List of sample keys with the specified label
    """
    with LMDBStorage(db_path) as storage:
        return storage.query_by_index("label_index", label)


def query_samples_by_category(db_path: str, category: str) -> List[str]:
    """Query samples by their category.
    
    Args:
        db_path: Path to the LMDB database
        category: Category to search for
        
    Returns:
        List of sample keys with the specified category
    """
    with LMDBStorage(db_path) as storage:
        return storage.query_by_index("category_index", category)


def query_samples_by_method(db_path: str, method: str) -> List[str]:
    """Query samples by their generation method.
    
    Args:
        db_path: Path to the LMDB database
        method: Method to search for
        
    Returns:
        List of sample keys with the specified method
    """
    with LMDBStorage(db_path) as storage:
        return storage.query_by_index("method_index", method)


def load_metadata(db_path: str) -> Dict[str, Any]:
    """Load metadata for the dataset.
    
    Args:
        db_path: Path to the LMDB database directory (not the .lmdb file)
        
    Returns:
        Dictionary containing metadata for all samples
    """
    metadata_path = os.path.join(os.path.dirname(db_path), "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


class LMDBDataTransform:
    """Transform class for LMDB data that converts to PyTorch tensors and normalizes."""

    def __init__(
        self,
        target_device: str = "cpu",
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        """Initialize the transform.

        Args:
            target_device: Target device for tensors ("cpu" or "cuda")
            normalize: Whether to apply ImageNet-style normalization
            mean: Optional mean for normalization (length-3 RGB)
            std: Optional std for normalization (length-3 RGB)
        """
        self.target_device = target_device
        self.normalize = normalize
        if self.normalize:
            # Default to ImageNet stats if not provided
            self.mean = torch.tensor(
                mean if mean is not None else [0.485, 0.456, 0.406],
                dtype=torch.float32,
            )
            self.std = torch.tensor(
                std if std is not None else [0.229, 0.224, 0.225],
                dtype=torch.float32,
            )
        else:
            self.mean = None
            self.std = None

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the transform to a sample.

        Args:
            sample: Sample dictionary from LMDB

        Returns:
            Transformed sample
        """
        transformed_sample: Dict[str, Any] = {}

        for key, value in sample.items():
            if key == "data":
                # Convert numpy array to torch tensor
                if isinstance(value, np.ndarray):
                    # Expect uint8; convert to float and normalize on the fly
                    if value.dtype == np.uint8:
                        tensor = torch.from_numpy(value).to(torch.float32).div_(255.0)
                    else:
                        tensor = torch.from_numpy(value).to(torch.float32)
                elif isinstance(value, torch.Tensor):
                    tensor = value.to(torch.float32)
                    if tensor.dtype != torch.float32:
                        tensor = tensor.float()
                else:
                    raise TypeError(f"Unexpected data type for 'data': {type(value)}")

                # Apply normalization if requested (assumes last dim is channels RGB)
                if self.normalize:
                    # Ensure mean/std on correct device and broadcast to (1,1,1,C)
                    mean = self.mean.to(tensor.device).view(1, 1, 1, -1)
                    std = self.std.to(tensor.device).view(1, 1, 1, -1)
                    tensor = (tensor - mean) / std

                # Move to target device
                tensor = tensor.to(self.target_device)
                transformed_sample[key] = tensor

            elif key == "label":
                transformed_sample[key] = torch.tensor(value, dtype=torch.long, device=self.target_device)

            else:
                transformed_sample[key] = value

        return transformed_sample


def create_train_val_loaders(
    train_db_path: str,
    val_db_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    target_device: str = "cpu",
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.
    
    Args:
        train_db_path: Path to training LMDB database
        val_db_path: Path to validation LMDB database
        batch_size: Batch size for both loaders
        num_workers: Number of worker processes
        target_device: Target device for tensors
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create transforms
    train_transform = LMDBDataTransform(target_device)
    val_transform = LMDBDataTransform(target_device)
    
    # Create data loaders
    train_loader = create_data_loader(
        train_db_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        transform=train_transform,
        **kwargs
    )
    
    val_loader = create_data_loader(
        val_db_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        transform=val_transform,
        **kwargs
    )
    
    return train_loader, val_loader


def benchmark_lmdb_performance(db_path: str, num_samples: int = 1000) -> Dict[str, float]:
    """Benchmark LMDB read performance.
    
    Args:
        db_path: Path to the LMDB database
        num_samples: Number of samples to test
        
    Returns:
        Dictionary containing performance metrics
    """
    import time
    
    with LMDBStorage(db_path) as storage:
        keys = storage.get_all_keys()
        if len(keys) == 0:
            return {"error": "No samples found in database"}
        
        # Test random access performance
        test_keys = np.random.choice(keys, min(num_samples, len(keys)), replace=False)
        
        # Warm up
        for key in test_keys[:10]:
            storage.retrieve_sample(key)
        
        # Benchmark random access
        start_time = time.time()
        for key in test_keys:
            storage.retrieve_sample(key)
        random_access_time = time.time() - start_time
        
        # Benchmark sequential access
        start_time = time.time()
        for key in keys[:num_samples]:
            storage.retrieve_sample(key)
        sequential_time = time.time() - start_time
        
        # Calculate metrics
        avg_random_time = random_access_time / len(test_keys)
        avg_sequential_time = sequential_time / min(num_samples, len(keys))
        
        return {
            "total_samples": len(keys),
            "tested_samples": len(test_keys),
            "avg_random_access_time_ms": avg_random_time * 1000,
            "avg_sequential_access_time_ms": avg_sequential_time * 1000,
            "total_random_access_time": random_access_time,
            "total_sequential_time": sequential_time,
            "samples_per_second_random": len(test_keys) / random_access_time,
            "samples_per_second_sequential": min(num_samples, len(keys)) / sequential_time
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="LMDB utility functions")
    parser.add_argument("--db_path", type=str, required=True, help="Path to LMDB database")
    parser.add_argument("--action", type=str, choices=["info", "benchmark", "query"], default="info")
    parser.add_argument("--label", type=int, help="Label to query for")
    parser.add_argument("--category", type=str, help="Category to query for")
    parser.add_argument("--method", type=str, help="Method to query for")
    
    args = parser.parse_args()
    
    if args.action == "info":
        stats = get_dataset_statistics(args.db_path)
        print("Dataset Statistics:")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Sample info: {stats['sample_info']}")
        print(f"Example keys: {stats['keys']}")
        
    elif args.action == "benchmark":
        perf = benchmark_lmdb_performance(args.db_path)
        print("Performance Benchmark:")
        for key, value in perf.items():
            print(f"{key}: {value}")
            
    elif args.action == "query":
        if args.label is not None:
            keys = query_samples_by_label(args.db_path, args.label)
            print(f"Samples with label {args.label}: {len(keys)}")
            print(f"First 10 keys: {keys[:10]}")
        elif args.category is not None:
            keys = query_samples_by_category(args.db_path, args.category)
            print(f"Samples with category {args.category}: {len(keys)}")
            print(f"First 10 keys: {keys[:10]}")
        elif args.method is not None:
            keys = query_samples_by_method(args.db_path, args.method)
            print(f"Samples with method {args.method}: {len(keys)}")
            print(f"First 10 keys: {keys[:10]}")
        else:
            print("Please specify --label, --category, or --method for querying")
