# LMDB Storage for Preprocessed Data

This document explains how to use the LMDB (Lightning Memory-Mapped Database) storage system for efficiently storing and accessing preprocessed deepfake detection data.

## Overview

LMDB provides high-performance, memory-mapped database storage that is ideal for large datasets. It offers:

- **Fast random access**: O(1) lookup time for any sample
- **Memory efficiency**: Only loads data into memory when accessed
- **Scalability**: Handles datasets of any size efficiently
- **ACID compliance**: Transaction-based operations ensure data integrity
- **Cross-platform**: Works on Windows, Linux, and macOS

## Installation

The LMDB dependency is already included in the project. If you need to install it separately:

```bash
pip install lmdb
```

## Configuration

Update your `configs/preprocessing.yaml` to use LMDB format:

```yaml
output:
  format: "lmdb"  # Options: "hdf5", "numpy", "torch", "lmdb"
  
  # LMDB-specific configuration
  lmdb:
    map_size_gb: 10  # Maximum database size in GB
    max_readers: 1024
    create_indexes: true  # Whether to create indexes for efficient querying
    index_fields: ["label", "category", "method"]  # Fields to index
```

## Usage

### 1. Preprocessing Data to LMDB

Run the preprocessing pipeline with LMDB output format:

```bash
python preprocess.py --dataset fakeavceleb
```

This will create LMDB databases in the following structure:
```
data/processed/
└── FakeAVCeleb_v1.2/
    ├── train/
    │   ├── data.lmdb/          # LMDB database
    │   └── metadata.json       # Sample metadata
    ├── val/
    │   ├── data.lmdb/
    │   └── metadata.json
    └── test/
        ├── data.lmdb/
        └── metadata.json
```

### 2. Loading Data for Training

Use the provided utilities to create PyTorch DataLoaders:

```python
from src.utils.lmdb_utils import create_train_val_loaders

# Create data loaders
train_loader, val_loader = create_train_val_loaders(
    train_db_path="data/processed/FakeAVCeleb_v1.2/train/data.lmdb",
    val_db_path="data/processed/FakeAVCeleb_v1.2/val/data.lmdb",
    batch_size=32,
    num_workers=4,
    target_device="cuda"  # or "cpu"
)

# Use in training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        data = batch['data']      # Shape: (batch, channels, frames, height, width)
        labels = batch['label']   # Shape: (batch,)
        # ... training code ...
```

### 3. Direct LMDB Access

For custom data loading or evaluation:

```python
from src.data.lmdb_storage import LMDBStorage

# Open database
with LMDBStorage("data/processed/FakeAVCeleb_v1.2/train/data.lmdb") as storage:
    # Get all sample keys
    keys = storage.get_all_keys()
    print(f"Total samples: {len(keys)}")
    
    # Load a specific sample
    sample = storage.retrieve_sample("sample_000001")
    print(f"Sample data shape: {sample['data'].shape}")
    print(f"Sample label: {sample['label']}")
    
    # Load multiple samples
    batch_keys = keys[:10]
    samples = storage.retrieve_batch(batch_keys)
```

### 4. Querying by Index

Use indexes for efficient filtering:

```python
from src.utils.lmdb_utils import query_samples_by_label, query_samples_by_category

# Query samples by label
real_samples = query_samples_by_label("data/processed/FakeAVCeleb_v1.2/train/data.lmdb", 0)
fake_samples = query_samples_by_label("data/processed/FakeAVCeleb_v1.2/train/data.lmdb", 1)

print(f"Real samples: {len(real_samples)}")
print(f"Fake samples: {len(fake_samples)}")

# Query by category
category_a_samples = query_samples_by_category("data/processed/FakeAVCeleb_v1.2/train/data.lmdb", "A")
print(f"Category A samples: {len(category_a_samples)}")
```

### 5. Performance Benchmarking

Benchmark LMDB read performance:

```python
from src.utils.lmdb_utils import benchmark_lmdb_performance

# Benchmark performance
perf = benchmark_lmdb_performance("data/processed/FakeAVCeleb_v1.2/train/data.lmdb", num_samples=1000)
print(f"Random access: {perf['avg_random_access_time_ms']:.2f} ms per sample")
print(f"Sequential access: {perf['avg_sequential_access_time_ms']:.2f} ms per sample")
print(f"Throughput: {perf['samples_per_second_random']:.0f} samples/second")
```

## Data Structure

Each sample in the LMDB database contains:

```python
{
    "data": np.ndarray,           # Preprocessed video frames
    "label": int,                 # 0 for real, 1 for fake
    "metadata": {
        "num_frames": int,        # Number of frames
        "video_path": str,        # Original video path
        "category": str,          # Dataset category (A, B, C, D)
        "method": str,            # Deepfake generation method
        "source": str,            # Source identity
        "target1": str,           # Target identity 1
        "target2": str,           # Target identity 2
        "type": str,              # Video type
        "gender": str,            # Gender
        "race": str               # Race
    }
}
```

## Advanced Features

### Custom Transforms

Create custom transforms for data augmentation:

```python
from src.utils.lmdb_utils import LMDBDataTransform
import torchvision.transforms as transforms

class CustomTransform(LMDBDataTransform):
    def __init__(self, target_device="cpu"):
        super().__init__(target_device)
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            # Add more augmentations as needed
        ])
    
    def __call__(self, sample):
        sample = super().__call__(sample)
        # Apply custom augmentations
        sample['data'] = self.augmentation(sample['data'])
        return sample

# Use in data loader
transform = CustomTransform(target_device="cuda")
train_loader = create_data_loader(
    db_path="data/processed/FakeAVCeleb_v1.2/train/data.lmdb",
    transform=transform,
    batch_size=32
)
```

### Batch Processing

Process multiple samples efficiently:

```python
from src.data.lmdb_storage import LMDBStorage

with LMDBStorage("data/processed/FakeAVCeleb_v1.2/train/data.lmdb") as storage:
    # Get all keys
    keys = storage.get_all_keys()
    
    # Process in batches
    batch_size = 100
    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i + batch_size]
        samples = storage.retrieve_batch(batch_keys)
        
        # Process batch
        for sample in samples:
            if sample is not None:
                # ... processing code ...
                pass
```
