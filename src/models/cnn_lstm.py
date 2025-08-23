import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Tuple, Optional
from base import BaseDetector


class FeatureExtractor(nn.Module):
    """Wrapper around torchvision models to extract features (no classifier)."""
    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            layers = list(model.children())[:-1]   # drop final FC, keep avgpool
            self.backbone = nn.Sequential(*layers)
            self.out_dim = model.fc.in_features    # feature size (512)
        elif backbone == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            layers = list(model.children())[:-1]
            self.backbone = nn.Sequential(*layers)
            self.out_dim = model.fc.in_features    # feature size (2048)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.backbone(x)          # (B, D, 1, 1)
        feats = torch.flatten(feats, 1)   # (B, D)
        return feats


class CNNLSTMDetector(BaseDetector):
    def __init__(self, hidden_size=512, num_classes=2, backbone_video="resnet18", backbone_audio="resnet18"):
        super().__init__(num_classes)
        
        # Use torchvision CNNs for both modalities
        self.video_cnn = FeatureExtractor(backbone=backbone_video, pretrained=True)
        self.audio_cnn = FeatureExtractor(backbone=backbone_audio, pretrained=True)

        fused_dim = self.video_cnn.out_dim + self.audio_cnn.out_dim

        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def get_video_features(self, image_input: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Extract features from the image input using the video CNN backbone.
        
        Args:
            image_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            lengths: (B,) true lengths (# real frames per sample, before padding)
            
        Returns:
            torch.Tensor: Video feature tensor of shape (batch_size, feature_dim)
        """
        batch_size, num_frames, channels, height, width = image_input.shape
        
        # Reshape to process all frames at once: (batch_size * num_frames, channels, height, width)
        video_reshaped = image_input.view(batch_size * num_frames, channels, height, width)
        
        # Extract features using CNN
        features = self.video_cnn(video_reshaped)  # (batch_size * num_frames, feature_dim)
        
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
        """Extract features from the audio input using the audio CNN backbone.
        
        Args:
            audio_input: Input tensor of shape (batch_size, num_frames, channels, height, width)
            lengths: (B,) true lengths (# real frames per sample, before padding)
            
        Returns:
            torch.Tensor: Audio feature tensor of shape (batch_size, feature_dim)
        """
        batch_size, num_frames, channels, height, width = audio_input.shape
        
        # Reshape to process all frames at once: (batch_size * num_frames, channels, height, width)
        audio_reshaped = audio_input.view(batch_size * num_frames, channels, height, width)
        
        # Extract features using CNN
        features = self.audio_cnn(audio_reshaped)  # (batch_size * num_frames, feature_dim)
        
        # Reshape back to (batch_size, num_frames, feature_dim)
        features = features.view(batch_size, num_frames, -1)
        
        # Create mask to ignore padded frames
        mask = torch.arange(num_frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
        mask = mask.unsqueeze(-1).float()  # (B, T, 1)
        
        # Apply mask and aggregate across frames using mean pooling
        masked_features = features * mask  # Zero out padded frames
        final_features = masked_features.sum(dim=1) / lengths.unsqueeze(1).float()  # Mean only over valid frames
        
        return final_features

    def forward(self, 
                image_input: Optional[torch.Tensor], 
                audio_input: Optional[torch.Tensor], 
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_input: (B, T, C, H, W) or None
            audio_input: (B, T, C, H, W) or None
            lengths: (B,) sequence lengths before padding
        """
        if image_input is None and audio_input is None:
            raise ValueError("At least one modality must be provided")

        B = lengths.shape[0]
        device = lengths.device

        # Prepare image features
        if image_input is not None:
            B_img, T, C, H, W = image_input.shape
            img_flat = image_input.view(B_img * T, C, H, W)
            img_feats = self.video_cnn(img_flat).view(B_img, T, -1)  # (B, T, D_img)
        else:
            # Zero padding if missing
            img_feats = torch.zeros(B, T, self.video_cnn.out_dim, device=device)

        # Prepare audio features
        if audio_input is not None:
            B_aud, T, C, H, W = audio_input.shape
            aud_flat = audio_input.view(B_aud * T, C, H, W)
            aud_feats = self.audio_cnn(aud_flat).view(B_aud, T, -1)  # (B, T, D_aud)
        else:
            # Zero padding if missing
            aud_feats = torch.zeros(B, T, self.audio_cnn.out_dim, device=device)

        # Fuse modalities (concatenation)
        fused_feats = torch.cat([img_feats, aud_feats], dim=-1)  # (B, T, D_img + D_aud)

        # Pack padded sequences
        packed = pack_padded_sequence(
            fused_feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        _, (h_n, _) = self.lstm(packed)
        last_hidden = h_n[-1]  # (B, H_lstm)

        # Classifier
        logits = self.fc(last_hidden)  # (B, num_classes)
        return logits

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
            logits = self.forward(image_input, audio_input, lengths)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            return predictions

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
        """Get predictions using only one modality (for ablation studies or single-modality inference).
        
        Args:
            image_input: Optional image input tensor
            audio_input: Optional audio input tensor
            lengths: (B,) true lengths (# real frames per sample, before padding)
            
        Returns:
            Predicted class (0 or 1) - the class with maximum probability
        """
        if image_input is not None and audio_input is not None:
            raise ValueError("Only one modality can be provided at a time")
        
        if image_input is None and audio_input is None:
            raise ValueError("At least one modality must be provided")
        
        self.eval()
        with torch.no_grad():
            if image_input is not None:
                batch_size, num_frames, C, H, W = image_input.shape
                device = image_input.device
                
                # Get video features
                img_flat = image_input.view(batch_size * num_frames, C, H, W)
                img_feats = self.video_cnn(img_flat).view(batch_size, num_frames, -1)  # (B, T, D_img)
                
                # Create zero audio features to match the expected input dimension
                aud_feats = torch.zeros(batch_size, num_frames, self.audio_cnn.out_dim, device=device)

            elif audio_input is not None:
                batch_size, num_frames, C, H, W = audio_input.shape
                device = audio_input.device
                
                # Get audio features
                aud_flat = audio_input.view(batch_size * num_frames, C, H, W)
                aud_feats = self.audio_cnn(aud_flat).view(batch_size, num_frames, -1)  # (B, T, D_aud)
                
                # Create zero video features to match the expected input dimension
                img_feats = torch.zeros(batch_size, num_frames, self.video_cnn.out_dim, device=device)

            # If no lengths provided, assume all frames are valid
            if lengths is None:
                lengths = torch.full((batch_size,), num_frames, device=device)

            # Fuse features (concatenation) - one modality will be zeros
            fused_feats = torch.cat([img_feats, aud_feats], dim=-1)  # (B, T, D_img + D_aud)

            # Pack padded sequences
            packed = pack_padded_sequence(
                fused_feats, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # LSTM
            _, (h_n, _) = self.lstm(packed)
            last_hidden = h_n[-1]  # (B, H_lstm)
            
            # Classifier
            logits = self.fc(last_hidden)  # (B, num_classes)
            
            # Prediction
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)  # (B,)
            
            return predictions
