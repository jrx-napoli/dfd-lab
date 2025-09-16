# Models Documentation

## BaseDetector (`base.py`)

The `BaseDetector` class is an abstract base class that defines the interface for all multi-modal (image and audio) deepfake detectors in the system. It inherits from `torch.nn.Module` and `ABC`, providing a standardized contract that all detector implementations must follow.

### Key Features:
- **Abstract Interface**: Defines mandatory methods that all detector implementations must provide
- **Multi-modal Support**: Designed to handle both video (image) and audio inputs
- **Standardized API**: Ensures consistent interface across different model architectures

### Abstract Methods:
- `forward()`: Core forward pass implementation
- `predict()`: Get class predictions from multi-modal input
- `get_confidence()`: Extract prediction confidence scores
- `get_modality_features()`: Extract features from both modalities
- `get_video_features()`: Extract features from video input only
- `get_audio_features()`: Extract features from audio input only
- `predict_single_modality()`: Single-modality prediction for ablation studies

## XceptionMaxFusionDetector (`xception.py`)

The `XceptionMaxFusionDetector` implements a multi-modal deepfake detection system using Xception networks from the `timm` library with a late fusion strategy.

### Architecture:
- **Video Branch**: Uses Xception model with 3 input channels for RGB video frames
- **Audio Branch**: Uses Xception model with 1 input channel (`in_chans=1`) for spectrogram data
- **Late Fusion**: Combines predictions from both modalities using element-wise maximum

### Fusion Logic:
1. **Parallel Processing**: Both video and audio inputs are processed independently through their respective Xception backbones
2. **Element-wise Maximum**: The logits from both modalities are combined using `torch.max(video_logits, audio_logits)`
3. **Temporal Pooling**: Final predictions are obtained by taking the maximum across the temporal dimension
4. **Output**: Returns fused logits representing the combined decision from both modalities

### Key Benefits:
- **Late Fusion**: Allows each modality to contribute its strongest evidence
- **Pretrained Backbones**: Leverages ImageNet-pretrained Xception weights
- **Flexible Input**: Supports variable-length video sequences
- **Single-modality Support**: Can perform predictions using only video or audio when needed
