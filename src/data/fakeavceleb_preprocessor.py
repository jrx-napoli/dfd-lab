import json
import os
import random
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.base_preprocessor import DataPreprocessor
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

        # Initialize shards-only storage
        self.sample_index = 0
        self._initialize_output_storage(output_dir)

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
                        self._save_incremental(result, output_dir)
                        
                        # Update statistics
                        self._update_statistics(stats, result['metadata'])
                        stats['total_samples'] += 1

                except Exception as e:
                    # Log and continue with next video
                    print(f"\nError processing {video_path}: {str(e)}")
                    continue

        # Finalize output storage
        self._finalize_output_storage(output_dir)

        # Save dataset statistics
        self._save_dataset_statistics(stats, output_dir)

    def _initialize_output_storage(self, output_dir: str):
        """Initialize shards-only storage."""
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

    def _save_incremental(self, result: Dict[str, Any], output_dir: str):
        """Save a single result incrementally (shards-only)."""
        # Save as clip samples with optional aligned audio mel-spectrogram slices
        wds_cfg = self.config["output"]["webdataset"]
        clip_len = int(wds_cfg.get("clip_length", 16))
        clip_stride = int(wds_cfg.get("clip_stride", clip_len))
        frames: np.ndarray = result["data"]  # (T,H,W,3) uint8
        label = int(result["label"])
        meta_base = result["metadata"].copy()

        t = frames.shape[0]
        fps = float(meta_base.get("fps", 25.0))
        audio_mel_full = result.get("audio_mel_full")  # [n_mels, time] float16 np.ndarray or None
        audio_cfg = self.config.get("preprocessing", {}).get("audio_processing", {})
        sr = int(audio_cfg.get("sample_rate", 16000))
        hop = int(audio_cfg.get("hop_length", 512))

        # If audio present, compute mapping from frame indices to mel frame indices via time
        mel_per_second = sr / hop if hop > 0 else 0.0

        start = 0
        while start + clip_len <= t:
            clip = frames[start:start + clip_len]
            sample_id = f"{self.sample_index:06d}_{start:06d}"
            meta = meta_base.copy()
            meta["clip_start_frame"] = int(start)
            meta["clip_length"] = int(clip_len)

            mel_clip: np.ndarray | None = None
            if audio_mel_full is not None and mel_per_second > 0 and fps > 0:
                # Time range for the clip in seconds
                t0 = start / fps
                t1 = (start + clip_len) / fps
                # Map to mel frame indices
                m0 = int(np.floor(t0 * mel_per_second))
                m1 = int(np.ceil(t1 * mel_per_second))
                m0 = max(0, m0)
                m1 = min(audio_mel_full.shape[1], m1)
                if m1 > m0:
                    mel_clip = audio_mel_full[:, m0:m1]

            self.shard_writer.add_sample(sample_id, clip, label, meta, mel_clip=mel_clip)
            start += clip_stride
        self.sample_index += 1

    def _finalize_output_storage(self, output_dir: str):
        """Finalize shards storage and save any remaining data."""
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

        train_ratio_cfg = float(self.config["data"].get("train_split", 0.8))
        val_ratio_cfg = float(self.config["data"].get("val_split", 0.2))
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
