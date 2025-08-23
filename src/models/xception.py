from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional
from base import BaseDetector

class XceptionMaxFusionDetector(BaseDetector):
    """
    Deepfake detector utilizing Xception from torchvision for both video (image) and audio (spectrogram)
    modalities, with a late fusion strategy based on taking the maximum of their
    individual predicted logits.
    """
    def __init__(self, num_classes: int = 2):
        """
        Initializes the XceptionMaxFusionDetector.

        Args:
            num_classes (int): The number of output classes for classification
                                (e.g., 2 for real/fake).
        """
        super().__init__(num_classes)
        
        # --- Video Predictor (Xception from timm) ---
        # Load pre-trained Xception for video processing
        # The 'xception' model in timm typically has a 'num_classes' argument for its head.
        self.video_model = timm.create_model('xception', pretrained=True, num_classes=num_classes)
        
        # --- Audio Predictor (Xception from timm) ---
        # Load another pre-trained Xception for audio processing.
        # IMPORTANT: This assumes your audio input (e.g., spectrograms) is preprocessed
        # into a 3-channel image-like format (e.g., 299x299 pixels) to be compatible
        # with Xception's input requirements.
        self.audio_model = timm.create_model('xception', pretrained=True, num_classes=num_classes)

    def get_video_features(self, image_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from the image input using the video Xception backbone.
        This method returns features before the final classification head.

        Args:
            image_input (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width) for video sequences.
            lengths (torch.Tensor): (B,) true lengths (# real frames per sample, before padding)

        Returns:
            torch.Tensor: Video feature tensor.
        """
        # Ensure input is a video sequence (5D tensor)
        if len(image_input.shape) != 5:
            raise ValueError("Input must be a video sequence with shape (batch_size, num_frames, channels, height, width)")
        
        batch_size, num_frames, channels, height, width = image_input.shape
        
        # Reshape to process all frames at once: (batch_size * num_frames, channels, height, width)
        video_reshaped = image_input.view(batch_size * num_frames, channels, height, width)
        
        # timm models have forward_features method
        features = self.video_model.forward_features(video_reshaped)
        
        # If the output is spatial (e.g., (B, C, H, W)), global average pool it
        if features.dim() == 4:  # Check if it's (B, C, H, W)
            features = features.mean(dim=[-2, -1])  # Global Average Pooling
        
        # Reshape back to (batch_size, num_frames, feature_dim)
        features = features.view(batch_size, num_frames, -1)
        
        # Create mask to ignore padded frames
        mask = torch.arange(num_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
        mask = mask.unsqueeze(-1).float()  # (B, T, 1)
        
        # Apply mask and aggregate across frames using mean pooling
        masked_features = features * mask  # Zero out padded frames
        final_features = masked_features.sum(dim=1) / lengths.unsqueeze(1).float()  # Mean only over valid frames
        
        return final_features

    def get_audio_features(self, audio_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from the audio input using the audio Xception backbone.
        This method returns features before the final classification head.

        Args:
            audio_input (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width) for audio sequences.
            lengths (torch.Tensor): (B,) true lengths (# real frames per sample, before padding)

        Returns:
            torch.Tensor: Audio feature tensor.
        """
        # Ensure input is a video sequence (5D tensor)
        if len(audio_input.shape) != 5:
            raise ValueError("Input must be an audio sequence with shape (batch_size, num_frames, channels, height, width)")
        
        batch_size, num_frames, channels, height, width = audio_input.shape
        
        # Reshape to process all frames at once: (batch_size * num_frames, channels, height, width)
        audio_reshaped = audio_input.view(batch_size * num_frames, channels, height, width)
        
        features = self.audio_model.forward_features(audio_reshaped)
        
        if features.dim() == 4:
            features = features.mean(dim=[-2, -1])
        
        # Reshape back to (batch_size, num_frames, feature_dim)
        features = features.view(batch_size, num_frames, -1)
        
        # Create mask to ignore padded frames
        mask = torch.arange(num_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
        mask = mask.unsqueeze(-1).float()  # (B, T, 1)
        
        # Apply mask and aggregate across frames using mean pooling
        masked_features = features * mask  # Zero out padded frames
        final_features = masked_features.sum(dim=1) / lengths.unsqueeze(1).float()  # Mean only over valid frames
        
        return final_features

    def forward(self, image_input: torch.Tensor, audio_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the XceptionMaxFusionDetector.

        Args:
            image_input (torch.Tensor): Input tensor of shape (batch_size, num_frames, C, H, W) for video sequences.
                                       Expected size for Xception is 299x299.
            audio_input (torch.Tensor): Input audio tensor of shape (batch_size, num_frames, C, H, W) for audio sequences.
                                       Expected size for Xception is 299x299.
            lengths (torch.Tensor): (B,) true lengths (# real frames per sample, before padding)

        Returns:
            torch.Tensor: Fused output logits of shape (batch_size, num_classes)
                          obtained by taking the element-wise maximum of video and audio logits and aggregating across frames using max pooling.
        """
        # Ensure inputs are video sequences (5D tensors)
        if len(image_input.shape) != 5 or len(audio_input.shape) != 5:
            raise ValueError("Inputs must be video sequences with shape (batch_size, num_frames, channels, height, width)")

        # Handle video sequence: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = image_input.shape
        
        # Reshape to process all frames at once: (batch_size * num_frames, channels, height, width)
        video_reshaped = image_input.view(batch_size * num_frames, channels, height, width)
        audio_reshaped = audio_input.view(batch_size * num_frames, channels, height, width)
        
        # Get logits for all frames
        video_logits = self.video_model(video_reshaped)
        audio_logits = self.audio_model(audio_reshaped)
        
        # Fuse per-frame logits
        fused_logits = torch.max(video_logits, audio_logits)  # (B*T, num_classes)

        # Reshape back to (batch_size, num_frames, num_classes)
        fused_logits = fused_logits.view(batch_size, num_frames, self.num_classes)

        # ---- Apply masking to ignore padded frames ----
        # Build mask: shape (batch_size, num_frames, 1)
        mask = torch.arange(num_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()  # (batch_size, num_frames, 1)

        # Set logits of padded frames to -inf so max-pool ignores them
        masked_logits = fused_logits.masked_fill(mask == 0, float('-inf'))

        # Temporal max-pooling across frames
        final_logits, _ = masked_logits.max(dim=1)  # (batch_size, num_classes)
            
        return final_logits


    def predict(self, image_input: torch.Tensor, audio_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Get model predictions (single class) for multi-modal input.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width) 
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Tensor of predicted classes (0 or 1) - one prediction per video in the batch
        """
        self.eval()
        with torch.no_grad():
            probabilities = torch.softmax(self.forward(image_input, audio_input, lengths), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            return predictions  # Return tensor directly
    
    
    def get_confidence(self, image_input: torch.Tensor, audio_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Get prediction confidence scores for multi-modal input.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width) 
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Confidence scores of shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(image_input, audio_input, lengths)
            probabilities = torch.softmax(logits, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            return confidence_scores
    
    def get_modality_features(self, image_input: torch.Tensor, audio_input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from both modalities.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width) 
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Tuple of (video_features, audio_features)
        """
        video_features = self.get_video_features(image_input, lengths)
        audio_features = self.get_audio_features(audio_input, lengths)
        return video_features, audio_features
    
    def predict_single_modality(self, image_input: Optional[torch.Tensor] = None, 
                                audio_input: Optional[torch.Tensor] = None, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get predictions using only one modality."""
        
        if image_input is not None and audio_input is not None:
            raise ValueError("Only one modality can be provided at a time")
        
        if image_input is None and audio_input is None:
            raise ValueError("At least one modality must be provided")
        
        self.eval()
        with torch.no_grad():
            if image_input is not None:
                batch_size, num_frames, C, H, W = image_input.shape
                frames = image_input.view(batch_size * num_frames, C, H, W)
                logits = self.video_model(frames)  # (B*T, num_classes)
                logits = logits.view(batch_size, num_frames, -1)

            elif audio_input is not None:
                batch_size, num_frames, C, H, W = audio_input.shape
                frames = audio_input.view(batch_size * num_frames, C, H, W)
                logits = self.audio_model(frames)  # (B*T, num_classes)
                logits = logits.view(batch_size, num_frames, -1)

            # If no lengths provided, assume all frames are valid
            if lengths is None:
                lengths = torch.full((batch_size), num_frames, device=logits.device)

            # ---- Apply masking ----
            mask = torch.arange(num_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
            mask = mask.unsqueeze(-1)  # (B, T, 1)

            # Mask out padded frames (set to -inf so max ignores them)
            masked_logits = logits.masked_fill(mask == 0, float("-inf"))

            # Temporal max pooling across frames
            final_logits, _ = masked_logits.max(dim=1)  # (B, num_classes)

            # Prediction
            probs = torch.softmax(final_logits, dim=1)
            predictions = torch.argmax(probs, dim=1)  # (B,)

            return predictions.cpu().tolist() if batch_size > 1 else predictions.item()