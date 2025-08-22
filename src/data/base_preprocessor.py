import json
import os
from typing import Dict, Any, List, Tuple

import cv2
# import face_alignment
import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.data.lmdb_storage import LMDBStorage


class DataPreprocessor:
    """Class for preprocessing deepfake detection data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.face_detector = self._initialize_face_detector()
        
    def _initialize_face_detector(self):
        """Initialize face detector based on configuration."""
        detector_type = self.config["preprocessing"]["face_detection"]["detector"]
        # if detector_type == "dlib":
        #     return face_alignment.FaceAlignment(
        #         face_alignment.LandmarksType._2D,
        #         flip_input=False,
        #         device="cuda" if torch.cuda.is_available() else "cpu"
        #     )
        if detector_type == "opencv":
            return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        else:
            raise ValueError(f"Unsupported face detector: {detector_type}")
    
    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        fps = self.config["preprocessing"]["frame_extraction"]["fps"]
        max_frames = self.config["preprocessing"]["frame_extraction"]["max_frames"]
        
        frames = []
        frame_count = 0
        frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            frame_count += 1
            
        cap.release()
        return frames
    
    def detect_face(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Detect face in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (cropped face, bounding box)
        """
        detector_type = self.config["preprocessing"]["face_detection"]["detector"]
        min_face_size = self.config["preprocessing"]["face_detection"]["min_face_size"]
        margin = self.config["preprocessing"]["face_detection"]["margin"]
        
        if detector_type == "dlib":
            landmarks = self.face_detector.get_landmarks(frame)
            if landmarks is None:
                return None, None
                
            landmarks = landmarks[0]
            x_min, y_min = landmarks.min(axis=0)
            x_max, y_max = landmarks.max(axis=0)
            
        elif detector_type == "opencv":
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size)
            )
            
            if len(faces) == 0:
                return None, None
                
            x, y, w, h = faces[0]
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h
        
        # Add margin
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, int(x_min - width * margin))
        y_min = max(0, int(y_min - height * margin))
        x_max = min(frame.shape[1], int(x_max + width * margin))
        y_max = min(frame.shape[0], int(y_max + height * margin))
        
        face = frame[y_min:y_max, x_min:x_max]
        return face, (x_min, y_min, x_max, y_max)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Detect and crop face
        face, bbox = self.detect_face(frame)
        if face is None:
            return None
            
        # Resize to target size
        target_size = self.config["preprocessing"]["image_processing"]["target_size"]
        face = cv2.resize(face, target_size)
        
        # Normalize
        mean = np.array(self.config["preprocessing"]["image_processing"]["normalization"]["mean"])
        std = np.array(self.config["preprocessing"]["image_processing"]["normalization"]["std"])
        face = (face / 255.0 - mean) / std
        
        return face
    
    def process_video(self, video_path: str, label: int) -> Dict[str, Any]:
        """Process a single video.
        
        Args:
            video_path: Path to the video file
            label: Label of the video (0 for real, 1 for fake)
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Process frames
        processed_frames = []
        for frame in frames:
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is not None:
                processed_frames.append(processed_frame)
        
        if not processed_frames:
            return None
            
        return {
            "data": np.stack(processed_frames),
            "label": label,
            "metadata": {
                "num_frames": len(processed_frames),
                "video_path": video_path
            }
        }
    
    def process_dataset(self, input_dir: str, output_dir: str):
        """Process entire dataset.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Process videos
        processed_data = []
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(input_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            print(f"Processing {split} split...")
            for video_file in tqdm(os.listdir(split_dir)):
                if not video_file.endswith((".mp4", ".avi", ".mov")):
                    continue
                    
                video_path = os.path.join(split_dir, video_file)
                label = 1 if "fake" in video_file.lower() else 0
                
                result = self.process_video(video_path, label)
                if result is not None:
                    processed_data.append(result)
        
        # Save processed data
        output_format = self.config["output"]["format"]
        if output_format == "hdf5":
            self._save_hdf5(processed_data, output_dir)
        elif output_format == "numpy":
            self._save_numpy(processed_data, output_dir)
        elif output_format == "torch":
            self._save_torch(processed_data, output_dir)
        elif output_format == "lmdb":
            self._save_lmdb(processed_data, output_dir)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _save_hdf5(self, data: List[Dict[str, Any]], output_dir: str):
        """Save processed data in HDF5 format."""
        for split in ["train", "val", "test"]:
            split_data = [d for d in data if d["metadata"]["video_path"].split(os.sep)[-2] == split]
            if not split_data:
                continue
                
            output_path = os.path.join(output_dir, f"{split}.h5")
            with h5py.File(output_path, "w") as f:
                for i, item in enumerate(split_data):
                    f.create_dataset(
                        f"sample_{i}",
                        data=item["data"],
                        compression=self.config["output"]["compression"],
                        compression_opts=self.config["output"]["compression_level"]
                    )
                    f[f"sample_{i}"].attrs["label"] = item["label"]
                    
            # Save metadata
            metadata = {f"sample_{i}": item["metadata"] for i, item in enumerate(split_data)}
            with open(os.path.join(output_dir, f"{split}_metadata.json"), "w") as f:
                json.dump(metadata, f)
    
    def _save_numpy(self, data: List[Dict[str, Any]], output_dir: str):
        """Save processed data in NumPy format."""
        for split in ["train", "val", "test"]:
            split_data = [d for d in data if d["metadata"]["video_path"].split(os.sep)[-2] == split]
            if not split_data:
                continue
                
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            for i, item in enumerate(split_data):
                np.save(
                    os.path.join(split_dir, f"{item['label']}_{i}.npy"),
                    item["data"]
                )
    
    def _save_torch(self, data: List[Dict[str, Any]], output_dir: str):
        """Save processed data in PyTorch format."""
        for split in ["train", "val", "test"]:
            split_data = [d for d in data if d["metadata"]["video_path"].split(os.sep)[-2] == split]
            if not split_data:
                continue
                
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            for i, item in enumerate(split_data):
                torch.save(
                    torch.from_numpy(item["data"]),
                    os.path.join(split_dir, f"{item['label']}_{i}.pt")
                ) 
    
    def _save_lmdb(self, data: List[Dict[str, Any]], output_dir: str):
        """Save processed data in LMDB format."""
        for split in ["train", "val", "test"]:
            split_data = [d for d in data if d["metadata"]["video_path"].split(os.sep)[-2] == split]
            if not split_data:
                continue
                
            split_output_dir = os.path.join(output_dir, split)
            os.makedirs(split_output_dir, exist_ok=True)
            
            # Initialize LMDB storage
            lmdb_path = os.path.join(split_output_dir, "data.lmdb")
            map_size = self.config["output"]["lmdb"]["map_size_gb"] * 1024 * 1024 * 1024
            
            with LMDBStorage(lmdb_path, map_size=map_size) as storage:
                print(f"Saving {len(split_data)} samples to {lmdb_path}")
                
                # Store samples with progress bar
                for i, item in enumerate(tqdm(split_data, desc=f"Saving {split} to LMDB")):
                    # Create unique key for the sample
                    key = f"sample_{i:06d}"
                    
                    # Store the sample
                    storage.store_sample(key, item)
                
                # Create indexes if configured
                if self.config["output"]["lmdb"]["create_indexes"]:
                    print(f"Creating indexes for {split} split...")
                    self._create_lmdb_indexes(storage, split_data)
                
                # Save metadata separately for easy access
                metadata = {f"sample_{i:06d}": item["metadata"] for i, item in enumerate(split_data)}
                metadata_path = os.path.join(split_output_dir, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Print dataset info
                info = storage.get_dataset_info()
                print(f"LMDB info for {split}: {info['total_samples']} samples")
    
    def _create_lmdb_indexes(self, storage: LMDBStorage, data: List[Dict[str, Any]]):
        """Create indexes for efficient querying in LMDB."""
        index_fields = self.config["output"]["lmdb"]["index_fields"]
        
        for field in index_fields:
            if field == "label":
                # Create label index
                def label_index_func(sample):
                    return sample.get("label", "unknown")
                storage.create_index("label_index", label_index_func)
                
            elif field == "category":
                # Create category index
                def category_index_func(sample):
                    return sample.get("metadata", {}).get("category", "unknown")
                storage.create_index("category_index", category_index_func)
                
            elif field == "method":
                # Create method index
                def method_index_func(sample):
                    return sample.get("metadata", {}).get("method", "unknown")
                storage.create_index("method_index", method_index_func) 