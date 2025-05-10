from typing import Dict, Any, List, Tuple, Optional
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
from .preprocessing import DataPreprocessor

class FakeAVCelebPreprocessor(DataPreprocessor):
    """Preprocessor specifically for the FakeAVCeleb dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the FakeAVCeleb preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        super().__init__(config)
        self.metadata = None
        self.category_mapping = {
            'A': 'RealVideo-RealAudio',
            'B': 'RealVideo-FakeAudio',
            'C': 'FakeVideo-RealAudio',
            'D': 'FakeVideo-FakeAudio'
        }
    
    def load_metadata(self, metadata_path: str):
        """Load dataset metadata from CSV file.
        
        Args:
            metadata_path: Path to the metadata CSV file
        """
        self.metadata = pd.read_csv(metadata_path)
    
    def get_video_label(self, video_path: str) -> Tuple[int, Dict[str, Any]]:
        """Get label and metadata for a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (label, metadata)
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
        
        # Extract filename from path
        filename = os.path.basename(video_path)
        
        # Find matching row in metadata
        row = self.metadata[self.metadata['filename'] == filename]
        if len(row) == 0:
            raise ValueError(f"No metadata found for video: {filename}")
        
        row = row.iloc[0]
        
        # Determine label (0 for real, 1 for fake)
        label = 0 if row['category'] == 'A' else 1
        
        # Create metadata dictionary
        metadata = {
            'source': row['source'],
            'target1': row['target1'],
            'target2': row['target2'],
            'method': row['method'],
            'category': row['category'],
            'type': row['type'],
            'gender': row['gender'],
            'race': row['race'],
            'filename': row['filename'],
            'path': row['path']
        }
        
        return label, metadata
    
    def process_dataset(self, input_dir: str, output_dir: str, metadata_path: str):
        """Process the FakeAVCeleb dataset.
        
        Args:
            input_dir: Directory containing the dataset
            output_dir: Directory to save processed data
            metadata_path: Path to the metadata CSV file
        """
        # Load metadata
        print("Loading metadata...")
        self.load_metadata(metadata_path)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each category
        processed_data = []
        for category in ['A', 'B', 'C', 'D']:
            category_dir = os.path.join(input_dir, self.category_mapping[category])
            if not os.path.exists(category_dir):
                continue
            
            print(f"Processing category {category} ({self.category_mapping[category]})...")
            
            # Process each video in the category
            for root, _, files in os.walk(category_dir):
                for file in files:
                    if not file.endswith('.mp4'):
                        continue
                    
                    video_path = os.path.join(root, file)
                    try:
                        # Get label and metadata
                        label, metadata = self.get_video_label(video_path)
                        
                        # Process video
                        result = self.process_video(video_path, label)
                        if result is not None:
                            # Add metadata to result
                            result['metadata'].update(metadata)
                            processed_data.append(result)
                            
                    except Exception as e:
                        print(f"Error processing {video_path}: {str(e)}")
                        continue
        
        # Save processed data
        print("Saving processed data...")
        output_format = self.config["output"]["format"]
        if output_format == "hdf5":
            self._save_hdf5(processed_data, output_dir)
        elif output_format == "numpy":
            self._save_numpy(processed_data, output_dir)
        elif output_format == "torch":
            self._save_torch(processed_data, output_dir)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        # Save dataset statistics
        self._save_dataset_statistics(processed_data, output_dir)
    
    def _save_dataset_statistics(self, processed_data: List[Dict[str, Any]], output_dir: str):
        """Save dataset statistics.
        
        Args:
            processed_data: List of processed data samples
            output_dir: Directory to save statistics
        """
        stats = {
            'total_samples': len(processed_data),
            'categories': {},
            'methods': {},
            'gender_distribution': {},
            'race_distribution': {}
        }
        
        for sample in processed_data:
            metadata = sample['metadata']
            
            # Update category statistics
            category = metadata['category']
            stats['categories'][category] = stats['categories'].get(category, 0) + 1
            
            # Update method statistics
            method = metadata['method']
            stats['methods'][method] = stats['methods'].get(method, 0) + 1
            
            # Update gender statistics
            gender = metadata['gender']
            stats['gender_distribution'][gender] = stats['gender_distribution'].get(gender, 0) + 1
            
            # Update race statistics
            race = metadata['race']
            stats['race_distribution'][race] = stats['race_distribution'].get(race, 0) + 1
        
        # Save statistics
        stats_path = os.path.join(output_dir, 'dataset_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2) 