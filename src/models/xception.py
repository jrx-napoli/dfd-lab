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

    def get_video_features(self, image_input: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from the image input using the video Xception backbone.
        This method returns features before the final classification head.

        Args:
            image_input (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width) for video sequences.

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
        
        # Aggregate across frames using mean pooling
        final_features = features.mean(dim=1)  # Shape: (batch_size, feature_dim)
        
        return final_features

    def get_audio_features(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Extracts features from the audio input using the audio Xception backbone.
        This method returns features before the final classification head.

        Args:
            audio_input (torch.Tensor): Input tensor of shape (batch_size, num_frames, channels, height, width) for audio sequences.

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
        
        # Aggregate across frames using mean pooling
        final_features = features.mean(dim=1)  # Shape: (batch_size, feature_dim)
        
        return final_features

    def forward(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the XceptionMaxFusionDetector.

        Args:
            image_input (torch.Tensor): Input tensor of shape (batch_size, num_frames, C, H, W) for video sequences.
                                       Expected size for Xception is 299x299.
            audio_input (torch.Tensor): Input audio tensor of shape (batch_size, num_frames, C, H, W) for audio sequences.
                                       Expected size for Xception is 299x299.

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
        
        # Apply max fusion across modalities and frames in one operation
        fused_logits = torch.max(video_logits, audio_logits)
        final_logits = torch.max(fused_logits, dim=1)[0] 
            
        return final_logits


    def predict(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> int:
        """Get model predictions (single class) for multi-modal input.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width) 
            
        Returns:
            Predicted class (0 or 1) - the class with maximum probability
        """
        self.eval()
        with torch.no_grad():
            probabilities = torch.softmax(self.forward(image_input, audio_input), dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            return predictions.item()  # Convert tensor to Python int
    
    
    def get_confidence(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """Get prediction confidence scores for multi-modal input.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width) 
            
        Returns:
            Confidence scores of shape (batch_size,)
        """
        probs = self.predict(image_input, audio_input)
        return torch.max(probs, dim=1)[0]
    
    def get_modality_features(self, image_input: torch.Tensor, audio_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from both modalities.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width) 
            
        Returns:
            Tuple of (video_features, audio_features)
        """
        video_features = self.get_video_features(image_input)
        audio_features = self.get_audio_features(audio_input)
        return video_features, audio_features
    
def predict_single_modality(self, image_input: Optional[torch.Tensor] = None, 
                            audio_input: Optional[torch.Tensor] = None) -> int:
    """Get predictions using only one modality."""
    
    if image_input is not None and audio_input is not None:
        raise ValueError("Only one modality can be provided at a time")
    
    if image_input is None and audio_input is None:
        raise ValueError("At least one modality must be provided")
    
    self.eval()
    with torch.no_grad():
        if image_input is not None:
            # Use video model directly
            batch_size, num_frames, channels, height, width = image_input.shape
            video_reshaped = image_input.view(batch_size * num_frames, channels, height, width)
            logits = self.video_model(video_reshaped)
            # Reshape and apply max pooling across frames
            logits = logits.view(batch_size, num_frames, -1)
            final_logits = torch.max(logits, dim=1)[0]
            predictions = torch.argmax(torch.softmax(final_logits, dim=1), dim=1)
            return predictions.item()  # Convert tensor to Python int
            
        elif audio_input is not None:
            # Use audio model directly
            batch_size, num_frames, channels, height, width = audio_input.shape
            audio_reshaped = audio_input.view(batch_size * num_frames, channels, height, width)
            logits = self.audio_model(audio_reshaped)
            # Reshape and apply max pooling across frames
            logits = logits.view(batch_size, num_frames, -1)
            final_logits = torch.max(logits, dim=1)[0]
            predictions = torch.argmax(torch.softmax(final_logits, dim=1), dim=1)
            return predictions.item()  # Convert tensor to Python int