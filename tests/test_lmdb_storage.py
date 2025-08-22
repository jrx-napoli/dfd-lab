"""
Test script for LMDB storage functionality.
Run this to verify that the LMDB implementation works correctly.
"""

import os
import tempfile
import numpy as np
import torch
import pytest

from src.data.lmdb_storage import LMDBStorage, LMDBDataset


def test_lmdb_storage_basic():
    """Test basic LMDB storage functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db")
        
        # Test data
        test_data = {
            "data": np.random.randn(10, 3, 224, 224).astype(np.float32),
            "label": 1,
            "metadata": {
                "num_frames": 10,
                "category": "A",
                "method": "test_method"
            }
        }
        
        # Store data
        with LMDBStorage(db_path) as storage:
            storage.store_sample("sample_001", test_data)
            
            # Retrieve data
            retrieved_data = storage.retrieve_sample("sample_001")
            
            # Verify data integrity
            assert retrieved_data is not None
            assert np.array_equal(retrieved_data["data"], test_data["data"])
            assert retrieved_data["label"] == test_data["label"]
            assert retrieved_data["metadata"]["category"] == test_data["metadata"]["category"]
            
            # Test batch operations
            test_data_2 = {
                "data": np.random.randn(5, 3, 224, 224).astype(np.float32),
                "label": 0,
                "metadata": {"num_frames": 5, "category": "B"}
            }
            
            samples = [("sample_002", test_data_2)]
            storage.store_batch(samples)
            
            # Test batch retrieval
            retrieved_batch = storage.retrieve_batch(["sample_001", "sample_002"])
            assert len(retrieved_batch) == 2
            assert retrieved_batch[0] is not None
            assert retrieved_batch[1] is not None
            
            # Test get all keys
            keys = storage.get_all_keys()
            assert len(keys) == 2
            assert "sample_001" in keys
            assert "sample_002" in keys


def test_lmdb_storage_torch_tensors():
    """Test LMDB storage with PyTorch tensors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db_torch")
        
        # Test data with torch tensors
        test_data = {
            "data": torch.randn(10, 3, 224, 224),
            "label": torch.tensor(1),
            "metadata": {"num_frames": 10}
        }
        
        with LMDBStorage(db_path) as storage:
            storage.store_sample("sample_torch", test_data)
            retrieved_data = storage.retrieve_sample("sample_torch")
            
            # Verify tensor data
            assert isinstance(retrieved_data["data"], torch.Tensor)
            assert torch.equal(retrieved_data["data"], test_data["data"])
            assert retrieved_data["label"] == test_data["label"]


def test_lmdb_storage_indexes():
    """Test LMDB index functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db_index")
        
        # Test data with different labels
        test_samples = [
            ("sample_001", {"data": np.random.randn(5, 3, 224, 224), "label": 0, "metadata": {"category": "A"}}),
            ("sample_002", {"data": np.random.randn(5, 3, 224, 224), "label": 1, "metadata": {"category": "B"}}),
            ("sample_003", {"data": np.random.randn(5, 3, 224, 224), "label": 0, "metadata": {"category": "A"}}),
        ]
        
        with LMDBStorage(db_path) as storage:
            # Store samples
            storage.store_batch(test_samples)
            
            # Create indexes
            def label_index_func(sample):
                return sample.get("label", "unknown")
            
            def category_index_func(sample):
                return sample.get("metadata", {}).get("category", "unknown")
            
            storage.create_index("label_index", label_index_func)
            storage.create_index("category_index", category_index_func)
            
            # Test queries
            label_0_keys = storage.query_by_index("label_index", 0)
            label_1_keys = storage.query_by_index("label_index", 1)
            category_a_keys = storage.query_by_index("category_index", "A")
            category_b_keys = storage.query_by_index("category_index", "B")
            
            assert len(label_0_keys) == 2
            assert len(label_1_keys) == 1
            assert len(category_a_keys) == 2
            assert len(category_b_keys) == 1


def test_lmdb_dataset():
    """Test LMDBDataset class."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db_dataset")
        
        # Create test data
        test_samples = []
        for i in range(10):
            test_samples.append((
                f"sample_{i:03d}",
                {
                    "data": np.random.randn(5, 3, 224, 224).astype(np.float32),
                    "label": i % 2,
                    "metadata": {"index": i}
                }
            ))
        
        # Store data
        with LMDBStorage(db_path) as storage:
            storage.store_batch(test_samples)
        
        # Test dataset
        dataset = LMDBDataset(db_path)
        
        # Test length
        assert len(dataset) == 10
        
        # Test indexing
        sample = dataset[0]
        assert sample is not None
        assert "data" in sample
        assert "label" in sample
        assert "metadata" in sample
        
        # Test key-based access
        sample_by_key = dataset.get_sample_by_key("sample_000")
        assert sample_by_key is not None


def test_lmdb_storage_large_data():
    """Test LMDB storage with larger data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db_large")
        
        # Create larger test data
        large_data = {
            "data": np.random.randn(100, 3, 224, 224).astype(np.float32),
            "label": 1,
            "metadata": {"size": "large", "frames": 100}
        }
        
        with LMDBStorage(db_path) as storage:
            # Store multiple large samples
            for i in range(5):
                storage.store_sample(f"large_sample_{i}", large_data)
            
            # Verify storage
            keys = storage.get_all_keys()
            assert len(keys) == 5
            
            # Test retrieval
            retrieved = storage.retrieve_sample("large_sample_0")
            assert retrieved is not None
            assert retrieved["data"].shape == (100, 3, 224, 224)


def test_lmdb_storage_context_manager():
    """Test LMDB storage context manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db_context")
        
        # Test context manager
        with LMDBStorage(db_path) as storage:
            storage.store_sample("test", {"data": np.array([1, 2, 3]), "label": 0})
            
            # Verify data is accessible within context
            retrieved = storage.retrieve_sample("test")
            assert retrieved is not None
        
        # Verify database is closed after context exit
        # This would raise an error if we tried to access it


def test_lmdb_storage_serialization():
    """Test data serialization and deserialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_db_serialization")
        
        # Test various data types
        test_data = {
            "numpy_array": np.random.randn(10, 10),
            "torch_tensor": torch.randn(5, 5),
            "list_data": [1, 2, 3, "test"],
            "dict_data": {"key": "value", "number": 42},
            "primitive": 123.45
        }
        
        with LMDBStorage(db_path) as storage:
            storage.store_sample("serialization_test", test_data)
            retrieved = storage.retrieve_sample("serialization_test")
            
            # Verify all data types are preserved
            assert isinstance(retrieved["numpy_array"], np.ndarray)
            assert isinstance(retrieved["torch_tensor"], torch.Tensor)
            assert isinstance(retrieved["list_data"], list)
            assert isinstance(retrieved["dict_data"], dict)
            assert isinstance(retrieved["primitive"], float)
            
            # Verify values
            assert np.array_equal(retrieved["numpy_array"], test_data["numpy_array"])
            assert torch.equal(retrieved["torch_tensor"], test_data["torch_tensor"])
            assert retrieved["list_data"] == test_data["list_data"]
            assert retrieved["dict_data"] == test_data["dict_data"]
            assert retrieved["primitive"] == test_data["primitive"]


if __name__ == "__main__":
    print("Running LMDB storage tests...")
    
    try:
        test_lmdb_storage_basic()
        print("✓ Basic storage test passed")
        
        test_lmdb_storage_torch_tensors()
        print("✓ Torch tensor test passed")
        
        test_lmdb_storage_indexes()
        print("✓ Index test passed")
        
        test_lmdb_dataset()
        print("✓ Dataset test passed")
        
        test_lmdb_storage_large_data()
        print("✓ Large data test passed")
        
        test_lmdb_storage_context_manager()
        print("✓ Context manager test passed")
        
        test_lmdb_storage_serialization()
        print("✓ Serialization test passed")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()
