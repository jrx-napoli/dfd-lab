from typing import Tuple, Dict, Any, Optional
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import h5py
import json

class DeepfakeDataset(Dataset):
    """Dataset class for deepfake detection."""
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Any] = None,
        split: str = "train",
        data_format: str = "hdf5"
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            transform: Optional transforms to apply to the images
            split: Dataset split ("train", "val", or "test")
            data_format: Format of the data files ("hdf5", "numpy", or "torch")
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.split = split
        self.data_format = data_format
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.samples = self._load_samples()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata."""
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}
    
    def _load_samples(self) -> list:
        """Load dataset samples."""
        if self.data_format == "hdf5":
            return self._load_hdf5_samples()
        elif self.data_format == "numpy":
            return self._load_numpy_samples()
        elif self.data_format == "torch":
            return self._load_torch_samples()
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")
    
    def _load_hdf5_samples(self) -> list:
        """Load samples from HDF5 file."""
        samples = []
        h5_path = os.path.join(self.data_dir, "data.h5")
        with h5py.File(h5_path, "r") as f:
            for key in f.keys():
                samples.append({
                    "id": key,
                    "data": f[key][:],
                    "label": f[key].attrs["label"]
                })
        return samples
    
    def _load_numpy_samples(self) -> list:
        """Load samples from numpy files."""
        samples = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".npy"):
                data = np.load(os.path.join(self.data_dir, filename))
                label = int(filename.split("_")[0])  # Assuming filename format: "label_id.npy"
                samples.append({
                    "id": filename,
                    "data": data,
                    "label": label
                })
        return samples
    
    def _load_torch_samples(self) -> list:
        """Load samples from torch files."""
        samples = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".pt"):
                data = torch.load(os.path.join(self.data_dir, filename))
                label = int(filename.split("_")[0])  # Assuming filename format: "label_id.pt"
                samples.append({
                    "id": filename,
                    "data": data,
                    "label": label
                })
        return samples
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to get
            
        Returns:
            Tuple of (image tensor, label)
        """
        sample = self.samples[idx]
        
        # Convert data to tensor if needed
        if isinstance(sample["data"], np.ndarray):
            image = torch.from_numpy(sample["data"])
        elif isinstance(sample["data"], torch.Tensor):
            image = sample["data"]
        else:
            raise ValueError(f"Unsupported data type: {type(sample['data'])}")
        
        # Apply transforms if specified
        if self.transform is not None:
            image = self.transform(image)
        
        return image, sample["label"]
    
    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing sample metadata
        """
        sample = self.samples[idx]
        return {
            "id": sample["id"],
            "label": sample["label"],
            **self.metadata.get(sample["id"], {})
        } 