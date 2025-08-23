from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Optional

class BaseDetector(nn.Module, ABC):
    """Base class for all multi-modal deepfake detectors."""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        
    @abstractmethod
    def forward(self, image_input: torch.Tensor, audio_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    @abstractmethod
    def get_video_features(self, image_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Extract features from the image input.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Video feature tensor
        """
        pass

    @abstractmethod
    def get_audio_features(self, audio_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Extract features from the audio input.
        
        Args:
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Audio feature tensor
        """
        pass
    
    @abstractmethod
    def predict(self, image_input: torch.Tensor, audio_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Get model predictions for multi-modal input.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Tensor of predicted classes (0 or 1) - one prediction per video in the batch
        """
        pass
    

    
    @abstractmethod
    def get_confidence(self, image_input: torch.Tensor, audio_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Get prediction confidence scores for multi-modal input.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Confidence scores of shape (batch_size,)
        """
        pass
    
    @abstractmethod
    def get_modality_features(self, image_input: torch.Tensor, audio_input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from both modalities.
        
        Args:
            image_input: Input tensor of shape (batch_size, channels, height, width)
            audio_input: Input tensor of shape (batch_size, audio_features, time_steps) or 
                         (batch_size, channels, height, width) for spectrogram input
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Tuple of (video_features, audio_features)
        """
        pass
    
    @abstractmethod
    def predict_single_modality(self, image_input: Optional[torch.Tensor] = None, 
                                audio_input: Optional[torch.Tensor] = None, lengths: Optional[torch.Tensor] = None) -> int:
        """Get predictions using only one modality (for ablation studies or single-modality inference).
        
        Args:
            image_input: Optional image input tensor
            audio_input: Optional audio input tensor
            lengths: (B,) true lengths (# real frames per sample, before padding)
        Returns:
            Predicted class (0 or 1) - the class with maximum probability
        """
        pass