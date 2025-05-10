from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseDetector(nn.Module, ABC):
    """Base class for all deepfake detectors."""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        pass
    
    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the input.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor
        """
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get model predictions.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Predicted class probabilities of shape (batch_size, num_classes)
        """
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)
    
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class labels.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Predicted class labels of shape (batch_size,)
        """
        return torch.argmax(self.predict(x), dim=1)
    
    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction confidence scores.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Confidence scores of shape (batch_size,)
        """
        probs = self.predict(x)
        return torch.max(probs, dim=1)[0] 