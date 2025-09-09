import os
import subprocess
from typing import Dict, Any, List, Tuple, Optional

import cv2
import librosa
# import face_alignment
import numpy as np
from tqdm import tqdm

from src.data.shard_writer import ShardWriter


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

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames

    def get_video_fps(self, video_path: str) -> float:
        """Get frames-per-second (FPS) of the input video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        cap.release()
        # Fallback to a sensible default if FPS could not be determined
        return float(fps if fps and fps > 0 else 25.0)

    def _extract_audio_mel(self, video_path: str, sr: int, n_mels: int, n_fft: int, hop_length: int) -> Optional[np.ndarray]:
        """Extract mel spectrogram using ffmpeg (PCM s16le) + librosa.

        Returns a numpy array [n_mels, time] in dB scale, or None on failure.
        """
        try:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(sr),
                "-",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            raw_audio = proc.stdout
            if not raw_audio:
                return None
            y = np.frombuffer(raw_audio, np.int16).astype(np.float32) / 32768.0
            if y.size == 0:
                return None
            mel_kwargs = {"y": y, "sr": sr, "n_mels": n_mels, "n_fft": n_fft, "hop_length": hop_length}
            S = librosa.feature.melspectrogram(**mel_kwargs)
            S_dB = librosa.power_to_db(S, ref=np.max)
            return S_dB.astype(np.float32)
        except Exception:
            return None
    
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
        
        # Keep as uint8 for compact storage; defer normalization to loader
        return face
    
    def process_video(self, video_path: str, label: int) -> Dict[str, Any]:
        """Process a single video.
        
        Args:
            video_path: Path to the video file
            label: Label of the video (0 for real, 1 for fake)
            
        Returns:
            Dictionary containing processed data and metadata
        """
        # Extract frames and fps
        frames = self.extract_frames(video_path)
        fps = self.get_video_fps(video_path)
        
        # Process frames
        processed_frames = []
        for frame in frames:
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is not None:
                processed_frames.append(processed_frame)
        
        if not processed_frames:
            return None

        result: Dict[str, Any] = {
            "data": np.stack(processed_frames),
            "label": label,
            "metadata": {
                "num_frames": len(processed_frames),
                "video_path": video_path,
                "fps": float(fps),
            },
        }

        # Optionally extract audio and compute full mel spectrogram (for later clip slicing)
        audio_cfg = self.config.get("preprocessing", {}).get("audio_processing", {})
        if audio_cfg and audio_cfg.get("enabled", False):
            sr = int(audio_cfg.get("sample_rate", 16000))
            n_mels = int(audio_cfg.get("n_mels", 80))
            n_fft = int(audio_cfg.get("n_fft", 2048))
            hop = int(audio_cfg.get("hop_length", 512))
            # Use librosa default upper frequency (Nyquist)
            mel_db = self._extract_audio_mel(video_path, sr, n_mels, n_fft, hop)
            if mel_db is not None and mel_db.size > 0:
                # Store as numpy float16 for compactness
                result["audio_mel_full"] = mel_db.astype(np.float16)
                result["metadata"].update({
                    "audio_sample_rate": sr,
                    "mel_hop_length": hop,
                    "mel_n_mels": n_mels,
                })

        return result
    
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

        # Save processed data to shards (assumed webdataset)
        self._save_shards(processed_data, output_dir)

    def _save_shards(self, data: List[Dict[str, Any]], output_dir: str):
        """Save processed data to shard tar files (webdataset-compatible)."""
        wds_cfg = self.config["output"]["webdataset"]
        for split in ["train", "val", "test"]:
            split_data = [d for d in data if d["metadata"]["video_path"].split(os.sep)[-2] == split]
            if not split_data:
                continue
            split_output_dir = os.path.join(output_dir, split, "shards")
            os.makedirs(split_output_dir, exist_ok=True)

            writer = ShardWriter(
                output_dir=split_output_dir,
                shard_prefix=wds_cfg.get("prefix", "shard"),
                max_shard_size_bytes=int(wds_cfg.get("max_shard_size_mb", 1024)) * 1024 * 1024,
                image_codec=wds_cfg.get("image_codec", "webp"),
                image_quality=int(wds_cfg.get("image_quality", 90)),
                index_filename=wds_cfg.get("index_filename", "index.csv"),
            )
            print(f"Saving {len(split_data)} samples to shards in {split_output_dir}")
            try:
                for i, item in enumerate(tqdm(split_data, desc=f"Saving {split} to shards")):
                    sample_id = f"sample_{i:06d}"
                    frames = item.get("data")
                    label = int(item.get("label", 0))
                    meta = item.get("metadata", {})
                    writer.add_sample(sample_id, frames, label, meta)
            finally:
                writer.close()