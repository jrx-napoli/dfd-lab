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

## AVClassifier with AVFF_encoder (`AVFF.py` & `AVFF_encoder.py`)

The `AVClassifier` implements an Audio-Video Fusion Framework (AVFF) that uses self-supervised pretraining to learn joint representations across modalities before classification.

### Architecture Overview:
- **Patch-based Tokenization**: Converts video and audio inputs into patch tokens
- **Transformer Encoders**: Separate encoders for video and audio modalities
- **Cross-modal Mappers**: Adaptive transformers that map features between modalities
- **Temporal Slicing**: Uses slice-based positional embeddings for temporal alignment

### Key Components from `AVFF_encoder.py`:

#### PatchTokenizer
- **Video**: Uses 3D convolution for spatio-temporal patches (default: 2×16×16)
- **Audio**: Uses 2D convolution per frame (default: 16×16)
- Outputs token embeddings suitable for transformer processing

#### AdaptiveCrossModalMapper
- Maps features between modalities with different token counts
- Handles upsampling (linear interpolation) and downsampling (mean pooling)
- Ensures cross-modal compatibility regardless of input dimensions

#### EncoderPretrain
The pretraining system uses a self-supervised approach with:
- **Complementary Masking**: Randomly masks half the temporal slices in each modality
- **Cross-modal Reconstruction**: Recovers masked slices using information from the other modality
- **InfoNCE Contrastive Loss**: Aligns audio and video representations
- **Adversarial Training**: Uses discriminators to ensure realistic reconstructions
- **WGAN-GP**: Gradient penalty for stable discriminator training

### AVClassifier Details:
- **Fusion Strategy**: Concatenates four feature vectors:
  1. Audio unimodal features (self-encoded)
  2. Audio cross-modal features (from video)
  3. Video unimodal features (self-encoded)
  4. Video cross-modal features (from audio)
- **Classifier**: Two-layer MLP with dropout for final prediction
- **Transfer Learning**: Supports freezing pretrained encoders for fine-tuning

### Training Workflow:
1. **Pretraining** (EncoderPretrain): Self-supervised learning of cross-modal representations
2. **Fine-tuning** (AVClassifier): Task-specific classification with optional encoder freezing

### Key Benefits:
- **Self-supervised Pretraining**: Learns rich cross-modal features without labeled data
- **Temporal Alignment**: Slice-based positional embeddings maintain temporal structure
- **Transferable Representations**: Pretrained encoders can be frozen or fine-tuned
- **Full Cross-modal Integration**: Uses both unimodal and cross-modal features for robust predictions