import json
import os
import random
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.base_preprocessor import DataPreprocessor
from src.data.lmdb_storage import LMDBStorage
from src.data.shard_writer import ShardWriter


class FakeAVCelebPreprocessor(DataPreprocessor):
    """Preprocessor for the FakeAVCeleb dataset."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the FakeAVCeleb preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        super().__init__(config)
        self.dataset_name = "FakeAVCeleb_v1.2"
        self.metadata_filename = "meta_data.csv"
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

    def process_dataset(self, input_dir: str, output_dir: str):
        """Process the FakeAVCeleb dataset.
        
        Args:
            input_dir: Directory containing the dataset
            output_dir: Directory to save processed data
        """
        # Load metadata
        print("Loading metadata...")
        metadata_path = os.path.join(input_dir, self.dataset_name, self.metadata_filename)
        self.load_metadata(metadata_path)

        # Create output directory
        output_dir = os.path.join(output_dir, self.dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize statistics tracking
        stats = {
            'total_samples': 0,
            'categories': {},
            'methods': {},
            'gender_distribution': {},
            'race_distribution': {}
        }

        # Initialize output format specific storage
        output_format = self.config["output"]["format"]
        # Generic counter for processed items (videos). Used for IDs/logging across formats
        self.sample_index = 0
        self._initialize_output_storage(output_dir, output_format)

        # Process each category
        for category in ['A', 'B', 'C', 'D']:
            category_dir = os.path.join(input_dir, self.dataset_name, self.category_mapping[category])
            if not os.path.exists(category_dir):
                continue

            print(f"\nProcessing category {category} ({self.category_mapping[category]})...")

            # Get list of all video files in the category
            video_files = []
            for root, _, files in os.walk(category_dir):
                for file in files:
                    if file.endswith('.mp4'):
                        video_files.append(os.path.join(root, file))

            # Process each video in the category with progress bar
            for video_path in tqdm(video_files, desc=f"Category {category}", unit="video"):
                try:
                    # Get label and metadata
                    label, metadata = self.get_video_label(video_path)

                    # Process video
                    result = self.process_video(video_path, label)
                    if result is not None:
                        # Add metadata to result
                        result['metadata'].update(metadata)
                        
                        # Save result incrementally
                        self._save_incremental(result, output_dir, output_format)
                        
                        # Update statistics
                        self._update_statistics(stats, result['metadata'])
                        stats['total_samples'] += 1

                except Exception as e:
                    # Check if this is our MDB_MAP_FULL error that should stop processing
                    if "MDB_MAP_FULL" in str(e) or "Environment mapsize limit reached" in str(e):
                        error_msg = (
                            f"\n\n{'=' * 64}\n"
                            f"ERROR: LMDB storage limit reached!\n"
                            f"{'=' * 64}\n"
                            f"The LMDB database has reached its maximum size limit.\n"
                            f"Current map size: {self.lmdb_storage.map_size / (1024 ** 3):.1f} GB\n"
                            f"Processed samples: {getattr(self, 'sample_index', getattr(self, 'lmdb_index', 0))}\n"
                            # f"Last attempted sample: {key}\n\n"
                            f"To continue processing, please:\n"
                            f"1. Increase the 'map_size_gb' value in your configuration file\n"
                            f"2. Restart the preprocessing with a larger map size\n"
                            f"{'=' * 64}\n"
                        )
                        print(error_msg)

                        # Close the storage cleanly
                        self.lmdb_storage.close()
                        return
                    else:
                        # Other errors - log and continue with next video
                        print(f"\nError processing {video_path}: {str(e)}")
                        continue

        # Finalize output storage
        self._finalize_output_storage(output_dir, output_format)

        # Save dataset statistics
        self._save_dataset_statistics(stats, output_dir)

    def _initialize_output_storage(self, output_dir: str, output_format: str):
        """Initialize storage for the specified output format.
        
        Args:
            output_dir: Directory to save processed data
            output_format: Format to save data in
        """
        if output_format == "lmdb":
            # Initialize LMDB storage
            lmdb_path = os.path.join(output_dir, 'processed_data.lmdb')
            map_size = self.config["output"]["lmdb"]["map_size_gb"] * 1024 * 1024 * 1024
            compression = self.config["output"].get("lmdb", {}).get("compression")
            compression_level = self.config["output"].get("lmdb", {}).get("compression_level", 0)
            self.lmdb_storage = LMDBStorage(
                lmdb_path,
                map_size=map_size,
                compression=compression,
                compression_level=compression_level,
            )
            self.lmdb_storage.open()
            self.lmdb_index = 0
        elif output_format == "webdataset":
            wds_cfg = self.config["output"]["webdataset"]
            shard_dir = os.path.join(output_dir, 'shards')
            self.shard_writer = ShardWriter(
                output_dir=shard_dir,
                shard_prefix=wds_cfg.get("prefix", "shard"),
                max_shard_size_bytes=int(wds_cfg.get("max_shard_size_mb", 1024)) * 1024 * 1024,
                image_codec=wds_cfg.get("image_codec", "webp"),
                image_quality=int(wds_cfg.get("image_quality", 90)),
                index_filename=wds_cfg.get("index_filename", "index.csv"),
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _save_incremental(self, result: Dict[str, Any], output_dir: str, output_format: str):
        """Save a single result incrementally.
        
        Args:
            result: Processed data sample
            output_dir: Directory to save processed data
            output_format: Format to save data in
        """
        if output_format == "lmdb":
            # Save data to LMDB
            key = f'sample_{self.lmdb_index:06d}'
            self.lmdb_storage.store_sample(key, result)
            self.lmdb_index += 1
            self.sample_index += 1
        elif output_format == "webdataset":
            # Save as a clip sample
            clip_len = int(self.config["output"]["webdataset"].get("clip_length", 16))
            clip_stride = int(self.config["output"]["webdataset"].get("clip_stride", clip_len))
            frames: np.ndarray = result["data"]  # (T,H,W,3) uint8
            label = int(result["label"])
            meta = result["metadata"].copy()
            # Create clips
            t = frames.shape[0]
            start = 0
            while start + clip_len <= t:
                clip = frames[start:start + clip_len]
                sample_id = f"{self.sample_index:06d}_{start:06d}"
                self.shard_writer.add_sample(sample_id, clip, label, meta)
                start += clip_stride
            self.sample_index += 1

    def _finalize_output_storage(self, output_dir: str, output_format: str):
        """Finalize storage and save any remaining data.
        
        Args:
            output_dir: Directory to save processed data
            output_format: Format to save data in
        """
        if output_format == "lmdb":
            self.lmdb_storage.close()
        elif output_format == "webdataset":
            self.shard_writer.close()
            # After writing shards and index.csv, generate stratified train/val index files
            try:
                self._stratified_split_webdataset_indexes(output_dir)
            except Exception as e:
                print(f"Warning: failed to create train/val indexes: {e}")

    def _stratified_split_webdataset_indexes(self, output_dir: str) -> None:
        """Create index_train.csv and index_val.csv stratified by label.

        Uses data.train_split and data.val_split from config. The two ratios
        are renormalized to sum to 1.0. Randomness is controlled by data.random_seed.
        """
        shard_dir = os.path.join(output_dir, 'shards')
        wds_cfg = self.config["output"]["webdataset"]
        index_filename = wds_cfg.get("index_filename", "index.csv")
        index_path = os.path.join(shard_dir, index_filename)

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"WebDataset index not found: {index_path}")

        with open(index_path, "r", encoding="utf-8") as f:
            header = f.readline()
            lines = [line.rstrip("\n") for line in f]

        # Group by label (5th column: sample_id,shard,dir,num_frames,label,metadata)
        by_label: Dict[int, List[str]] = {}
        for line in lines:
            parts = line.split(",", 5)
            if len(parts) < 6:
                continue
            try:
                label = int(parts[4])
            except ValueError:
                # Skip malformed line
                continue
            by_label.setdefault(label, []).append(line)

        train_ratio_cfg = float(self.config["data"].get("train_split", 0.7))
        val_ratio_cfg = float(self.config["data"].get("val_split", 0.3))
        total = max(train_ratio_cfg + val_ratio_cfg, 1e-9)
        train_ratio = train_ratio_cfg / total
        val_ratio = val_ratio_cfg / total
        seed = int(self.config["data"].get("random_seed", 42))
        rng = random.Random(seed)

        train_lines: List[str] = []
        val_lines: List[str] = []
        for _, group in by_label.items():
            rng.shuffle(group)
            n_val = int(round(len(group) * val_ratio))
            val_lines.extend(group[:n_val])
            train_lines.extend(group[n_val:])

        # Write outputs
        train_out = os.path.join(shard_dir, "index_train.csv")
        val_out = os.path.join(shard_dir, "index_val.csv")
        with open(train_out, "w", encoding="utf-8") as f:
            f.write(header)
            for line in train_lines:
                f.write(line + "\n")
        with open(val_out, "w", encoding="utf-8") as f:
            f.write(header)
            for line in val_lines:
                f.write(line + "\n")
        print(f"Created stratified indexes: {os.path.basename(train_out)} ({len(train_lines)}), {os.path.basename(val_out)} ({len(val_lines)})")

    def _update_statistics(self, stats: Dict[str, Any], metadata: Dict[str, Any]):
        """Update statistics with metadata from a processed sample.
        
        Args:
            stats: Current statistics dictionary
            metadata: Metadata from processed sample
        """
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

    def _save_dataset_statistics(self, stats: Dict[str, Any], output_dir: str):
        """Save dataset statistics.
        
        Args:
            stats: Statistics dictionary
            output_dir: Directory to save statistics
        """
        stats_path = os.path.join(output_dir, 'dataset_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
