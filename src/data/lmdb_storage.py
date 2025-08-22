import json
import os
import pickle
from typing import Dict, Any, List, Optional, Tuple

import lmdb
import numpy as np
import torch

import zstandard as zstd
import zlib


class LMDBStorage:
    """LMDB-based storage for preprocessed deepfake detection data.
    
    Provides efficient storage and retrieval of large datasets with fast random access.
    """
    
    def __init__(self, db_path: str, map_size: int = 1024 * 1024 * 1024 * 10, compression: Optional[str] = None, compression_level: int = 0):  # 10GB default
        """Initialize LMDB storage.
        
        Args:
            db_path: Path to the LMDB database directory
            map_size: Maximum size of the database in bytes
        """
        self.db_path = db_path
        self.map_size = map_size
        self.compression = (compression or '').lower() or None
        self.compression_level = compression_level
        self.env = None
        self._ensure_db_directory()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        os.makedirs(self.db_path, exist_ok=True)
    
    def open(self):
        """Open the LMDB environment."""
        if self.env is None:
            self.env = lmdb.open(
                self.db_path,
                map_size=self.map_size,
                subdir=True,
                readonly=False,
                meminit=False,
                map_async=True,
                max_readers=1024,
                max_dbs=128  # Allow up to 128 named databases for indexes
            )
    
    def close(self):
        """Close the LMDB environment."""
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def store_sample(self, key: str, data: Dict[str, Any], txn=None):
        """Store a single sample in the database.
        
        Args:
            key: Unique identifier for the sample
            data: Dictionary containing the sample data
            txn: Optional transaction object for batch operations
        """
        if self.env is None:
            raise RuntimeError("Database not open. Call open() first or use context manager.")
        
        # Serialize the data
        serialized_data = self._serialize_data(data)
        
        # Store in database
        if txn is None:
            with self.env.begin(write=True) as txn:
                txn.put(key.encode(), serialized_data)
        else:
            txn.put(key.encode(), serialized_data)
    
    def store_batch(self, samples: List[Tuple[str, Dict[str, Any]]]):
        """Store multiple samples in a batch transaction.
        
        Args:
            samples: List of (key, data) tuples
        """
        if self.env is None:
            raise RuntimeError("Database not open. Call open() first or use context manager.")
        
        with self.env.begin(write=True) as txn:
            for key, data in samples:
                self.store_sample(key, data, txn)
    
    def retrieve_sample(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single sample from the database.
        
        Args:
            key: Unique identifier for the sample
            
        Returns:
            Sample data dictionary or None if not found
        """
        if self.env is None:
            raise RuntimeError("Database not open. Call open() first or use context manager.")
        
        with self.env.begin(write=False) as txn:
            value = txn.get(key.encode())
            if value is None:
                return None
            
            return self._deserialize_data(value)
    
    def retrieve_batch(self, keys: List[str]) -> List[Optional[Dict[str, Any]]]:
        """Retrieve multiple samples from the database.
        
        Args:
            keys: List of sample identifiers
            
        Returns:
            List of sample data dictionaries (None for missing samples)
        """
        if self.env is None:
            raise RuntimeError("Database not open. Call open() first or use context manager.")
        
        results = []
        with self.env.begin(write=False) as txn:
            for key in keys:
                value = txn.get(key.encode())
                if value is None:
                    results.append(None)
                else:
                    results.append(self._deserialize_data(value))
        
        return results
    
    def get_all_keys(self) -> List[str]:
        """Get all keys in the database.
        
        Returns:
            List of all sample keys
        """
        if self.env is None:
            raise RuntimeError("Database not open. Call open() first or use context manager.")
        
        keys = []
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                keys.append(key.decode())
        
        return sorted(keys)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset.
        
        Returns:
            Dictionary containing dataset statistics and metadata
        """
        if self.env is None:
            raise RuntimeError("Database not open. Call open() first or use context manager.")
        
        keys = self.get_all_keys()
        total_samples = len(keys)
        
        # Get sample info from first few samples
        sample_info = {}
        if keys:
            sample = self.retrieve_sample(keys[0])
            if sample:
                sample_info = {
                    'data_shape': sample.get('data', {}).shape if hasattr(sample.get('data', {}), 'shape') else None,
                    'data_type': type(sample.get('data')).__name__,
                    'has_metadata': 'metadata' in sample,
                    'metadata_keys': list(sample.get('metadata', {}).keys()) if 'metadata' in sample else []
                }
        
        return {
            'total_samples': total_samples,
            'sample_info': sample_info,
            'keys': keys[:10] if keys else []  # First 10 keys as examples
        }
    
    def _serialize_data(self, data: Dict[str, Any]) -> bytes:
        """Serialize data for storage in LMDB.
        
        Args:
            data: Data dictionary to serialize
            
        Returns:
            Serialized bytes
        """
        serialized_data: Dict[str, Any] = {}

        for key, value in data.items():
            item: Dict[str, Any]
            # Prepare raw payload and metadata
            if isinstance(value, np.ndarray):
                raw = value.tobytes()
                meta = {
                    'type': 'numpy',
                    'shape': value.shape,
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, torch.Tensor):
                np_value = value.detach().cpu().numpy()
                raw = np_value.tobytes()
                meta = {
                    'type': 'torch',
                    'shape': tuple(value.shape),
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, (dict, list)):
                raw = json.dumps(value, default=str).encode('utf-8')
                meta = {
                    'type': 'json',
                    'encoding': 'utf-8'
                }
            else:
                # Use pickle for primitives/others for robustness
                raw = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                meta = {
                    'type': 'primitive'
                }

            # Optionally compress
            if self.compression in ("zstd", "zstandard") and zstd is not None:
                cctx = zstd.ZstdCompressor(level=self.compression_level or 3)
                payload = cctx.compress(raw)
                meta['compressed'] = True
                meta['algorithm'] = 'zstd'
            elif self.compression in ("zlib", "gzip"):
                payload = zlib.compress(raw, self.compression_level or 6)
                meta['compressed'] = True
                meta['algorithm'] = 'zlib'
            else:
                payload = raw
                meta['compressed'] = False

            item = {
                'meta': meta,
                'data': payload
            }
            serialized_data[key] = item

        return pickle.dumps(serialized_data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_data(self, serialized_bytes: bytes) -> Dict[str, Any]:
        """Deserialize data from LMDB storage.
        
        Args:
            serialized_bytes: Serialized data bytes
            
        Returns:
            Deserialized data dictionary
        """
        serialized_data = pickle.loads(serialized_bytes)
        deserialized_data: Dict[str, Any] = {}

        for key, item in serialized_data.items():
            meta = item['meta']
            payload = item['data']

            # Decompress if needed
            if meta.get('compressed'):
                algo = meta.get('algorithm')
                if algo == 'zstd' and zstd is not None:
                    dctx = zstd.ZstdDecompressor()
                    raw = dctx.decompress(payload)
                elif algo == 'zlib':
                    raw = zlib.decompress(payload)
                else:
                    # Unknown compression or missing backend
                    raise RuntimeError(f"Unsupported compression algorithm or backend not available: {algo}")
            else:
                raw = payload

            data_type = meta['type']

            if data_type == 'numpy':
                shape = tuple(meta['shape'])
                dtype = np.dtype(meta['dtype'])
                deserialized_data[key] = np.frombuffer(raw, dtype=dtype).reshape(shape)

            elif data_type == 'torch':
                shape = tuple(meta['shape'])
                torch_dtype_str = meta['dtype']
                if torch_dtype_str.startswith('torch.'):
                    torch_to_numpy_dtype = {
                        'torch.float32': np.float32,
                        'torch.float64': np.float64,
                        'torch.int32': np.int32,
                        'torch.int64': np.int64,
                        'torch.uint8': np.uint8,
                        'torch.int8': np.int8,
                        'torch.int16': np.int16,
                        'torch.bool': np.bool_,
                    }
                    numpy_dtype = torch_to_numpy_dtype.get(torch_dtype_str, np.float32)
                else:
                    numpy_dtype = np.dtype(torch_dtype_str)

                numpy_array = np.frombuffer(raw, dtype=numpy_dtype).reshape(shape)
                numpy_array = numpy_array.copy()
                deserialized_data[key] = torch.from_numpy(numpy_array)

            elif data_type == 'json':
                encoding = meta.get('encoding', 'utf-8')
                deserialized_data[key] = json.loads(raw.decode(encoding))

            elif data_type == 'primitive':
                deserialized_data[key] = pickle.loads(raw)

            else:
                raise ValueError(f"Unknown data type: {data_type}")

        return deserialized_data
    
    def create_index(self, index_name: str, index_func):
        """Create an index for efficient querying.
        
        Args:
            index_name: Name of the index
            index_func: Function that extracts index key from sample data
        """
        if self.env is None:
            raise RuntimeError("Database not open. Call open() first or use context manager.")
        
        index_db = self.env.open_db(index_name.encode())
        
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(db=index_db)
            main_cursor = txn.cursor(db=None)
            
            # Only process keys that look like sample keys (not index keys)
            for key, value in main_cursor:
                key_str = key.decode()
                # Skip keys that are likely index database keys
                if key_str in [index_name] or b'_index' in key:
                    continue
                    
                try:
                    sample = self._deserialize_data(value)
                    index_key = index_func(sample)
                    if index_key is not None:
                        index_key_bytes = str(index_key).encode()
                        
                        # Get existing values for this index key
                        existing_value = cursor.get(index_key_bytes)
                        if existing_value:
                            # Append to existing list
                            existing_keys = existing_value.decode().split(',')
                            if key_str not in existing_keys:
                                existing_keys.append(key_str)
                            new_value = ','.join(existing_keys).encode()
                        else:
                            # First entry for this index key
                            new_value = key
                        
                        cursor.put(index_key_bytes, new_value)
                except Exception as e:
                    # Skip entries that can't be deserialized 
                    continue
    
    def query_by_index(self, index_name: str, index_key: Any) -> List[str]:
        """Query samples using an index.
        
        Args:
            index_name: Name of the index to use
            index_key: Key to search for
            
        Returns:
            List of sample keys matching the index key
        """
        if self.env is None:
            raise RuntimeError("Database not open. Call open() first or use context manager.")
        
        index_db = self.env.open_db(index_name.encode())
        matching_keys = []
        
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor(db=index_db)
            index_key_str = str(index_key).encode()
            
            # Find all matches
            for key, value in cursor:
                if key == index_key_str:
                    # Split comma-separated values
                    keys_list = value.decode().split(',')
                    matching_keys.extend(keys_list)
        
        return matching_keys


class LMDBDataset:
    """PyTorch-style dataset interface for LMDB storage."""
    
    def __init__(self, db_path: str, transform=None):
        """Initialize the dataset.
        
        Args:
            db_path: Path to the LMDB database
            transform: Optional transform to apply to samples
        """
        self.db_path = db_path
        self.transform = transform
        self.storage = LMDBStorage(db_path)
        self.keys = None
        self._load_keys()
    
    def _load_keys(self):
        """Load all keys from the database."""
        with self.storage:
            self.keys = self.storage.get_all_keys()
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.keys)
    
    def __getitem__(self, idx):
        """Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Sample data
        """
        if idx >= len(self.keys):
            raise IndexError(f"Index {idx} out of range")
        
        key = self.keys[idx]
        with self.storage:
            sample = self.storage.retrieve_sample(key)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_sample_by_key(self, key: str):
        """Get a sample by its key.
        
        Args:
            key: Sample key
            
        Returns:
            Sample data
        """
        with self.storage:
            sample = self.storage.retrieve_sample(key)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
