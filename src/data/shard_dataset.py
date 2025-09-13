import io
import json
import os
import tarfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset


def _decode_image(buf: bytes) -> np.ndarray:
    nparr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode image from buffer")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


class ShardClipDataset(IterableDataset):
    """Streams tar shards of samples (directories) with frames and meta.json.

    Each yielded item is a dict: {"data": Tensor[T,H,W,3], "label": int, "metadata": dict, "audio_mel_frames": Tensor[T,n_mels,mel_frames_per_frame]}
    The audio_mel_frames key is only present if audio processing was enabled during preprocessing.
    Normalization can be applied downstream using existing LMDBDataTransform if desired.
    """

    def __init__(
        self,
        shards_dir: str,
        index_filename: str = "index.csv",
        target_device: str = "cpu",
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.shards_dir = shards_dir
        self.index_path = os.path.join(shards_dir, index_filename)
        self.target_device = target_device
        self.max_samples = max_samples

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        # Preload index (simple CSV)
        self._entries = self._load_index(self.index_path)

    def _load_index(self, path: str) -> List[Tuple[str, str, str, int, int, str]]:
        entries: List[Tuple[str, str, str, int, int, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            # skip header
            next(f)
            for line in f:
                sample_id, shard, sample_dir, num_frames, label, meta_json = line.rstrip("\n").split(",", 5)
                entries.append((sample_id, shard, sample_dir, int(num_frames), int(label), meta_json))
        return entries

    def _iter_samples(self):
        # Group by shard to minimize opens
        shard_to_entries: Dict[str, List[Tuple[str, str, str, int, int, str]]] = {}
        for e in self._entries[: self.max_samples] if self.max_samples else self._entries:
            shard_to_entries.setdefault(e[1], []).append(e)

        for shard_name, entries in shard_to_entries.items():
            shard_path = os.path.join(self.shards_dir, shard_name)
            with tarfile.open(shard_path, mode="r") as tar:
                # Index members for quick access
                members = {m.name: m for m in tar.getmembers()}
                for sample_id, _, sample_dir, num_frames, label, meta_json in entries:
                    # Read frames
                    frames: List[np.ndarray] = []
                    for i in range(num_frames):
                        # Try webp then jpg
                        for ext in (".webp", ".jpg"):
                            name = f"{sample_dir}/frame_{i:03d}{ext}"
                            mem = members.get(name)
                            if mem is not None:
                                buf = tar.extractfile(mem).read()
                                frames.append(_decode_image(buf))
                                break
                        else:
                            raise FileNotFoundError(f"Missing frame {i} for {sample_id}")

                    data = np.stack(frames).astype(np.uint8)
                    meta = json.loads(meta_json)
                    meta["id"] = sample_id
                    meta["num_frames"] = num_frames

                    # Optional audio mel features (frame-level)
                    mel_frames_tensor = None
                    mel_frames_shape_member = members.get(f"{sample_dir}/audio_mel_frames.json")
                    mel_frames_data_member = members.get(f"{sample_dir}/audio_mel_frames.f16")
                    if mel_frames_shape_member is not None and mel_frames_data_member is not None:
                        try:
                            shape_info = json.loads(tar.extractfile(mel_frames_shape_member).read().decode("utf-8"))
                            arr_bytes = tar.extractfile(mel_frames_data_member).read()
                            shape = shape_info.get("shape", [])
                            if isinstance(shape, list) and len(shape) == 3:
                                frames, mels, mel_frames_per_frame = int(shape[0]), int(shape[1]), int(shape[2])
                                arr = np.frombuffer(arr_bytes, dtype=np.float16)
                                if arr.size == frames * mels * mel_frames_per_frame:
                                    mel_frames_np = arr.reshape(frames, mels, mel_frames_per_frame)
                                    mel_frames_tensor = torch.from_numpy(mel_frames_np)
                        except Exception:
                            mel_frames_tensor = None

                    sample = {
                        "data": torch.from_numpy(data).to(self.target_device),
                        "label": label,
                        "metadata": meta,
                    }
                    if mel_frames_tensor is not None:
                        sample["audio_mel_frames"] = mel_frames_tensor

                    yield sample

    def __iter__(self):
        return self._iter_samples()


