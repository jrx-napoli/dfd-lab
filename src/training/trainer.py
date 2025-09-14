from typing import Dict, Any, Optional, List
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class Trainer:
    """Class for training deepfake detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_criterion()
        
        # Initialize metrics
        self.metrics = self._initialize_metrics()
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config["checkpointing"]["dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training state
        self.current_epoch = 0
        self.best_metric = float("-inf") if config["checkpointing"]["mode"] == "max" else float("inf")
        self.early_stopping_counter = 0
        
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on configuration."""
        optimizer_config = self.config["training"]["optimizer"]
        if optimizer_config["name"] == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config["learning_rate"],
                weight_decay=optimizer_config["weight_decay"]
            )
        elif optimizer_config["name"] == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=optimizer_config["learning_rate"],
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_config = self.config["training"]["scheduler"]
        if scheduler_config["name"] == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["epochs"],
                eta_min=scheduler_config["min_lr"]
            )
        elif scheduler_config["name"] == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_config["name"] == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.1,
                patience=5,
                min_lr=scheduler_config["min_lr"]
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_config = self.config["loss"]
        if loss_config["name"] == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_config["name"] == "focal":
            return FocalLoss(gamma=loss_config["focal_gamma"])
        else:
            raise ValueError(f"Unsupported loss function: {loss_config['name']}")
    
    def _initialize_metrics(self) -> Dict[str, float]:
        """Initialize metrics dictionary."""
        return {metric: 0.0 for metric in self.config["logging"]["metrics"]}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        epoch_metrics = self._initialize_metrics()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                pred = output.argmax(dim=1)
                self._update_metrics(epoch_metrics, pred, target)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{epoch_metrics['accuracy']:.4f}"
            })
        
        # Average metrics
        for metric in epoch_metrics:
            epoch_metrics[metric] /= len(self.train_loader)
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        val_metrics = self._initialize_metrics()
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                self._update_metrics(val_metrics, pred, target)
        
        # Average metrics
        for metric in val_metrics:
            val_metrics[metric] /= len(self.val_loader)
        
        return val_metrics
    
    def _update_metrics(self, metrics: Dict[str, float], pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with current batch predictions.
        
        Args:
            metrics: Dictionary of metrics to update
            pred: Predicted labels
            target: Ground truth labels
        """
        # Convert to numpy for metric calculation
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        Update each metric
        for metric in metrics:
            if metric == "accuracy":
                metrics[metric] += (pred_np == target_np).mean()
            elif metric == "precision":
                metrics[metric] += precision_score(target_np, pred_np, average="binary")
            elif metric == "recall":
                metrics[metric] += recall_score(target_np, pred_np, average="binary")
            elif metric == "f1_score":
                metrics[metric] += f1_score(target_np, pred_np, average="binary")
            elif metric == "auc_roc":
                metrics[metric] += roc_auc_score(target_np, pred_np)
    
    def train(self):
        """Train the model."""
        for epoch in range(self.config["training"]["epochs"]):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.config["checkpointing"]["monitor"]])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            self._save_checkpoint(val_metrics)
            
            # Early stopping
            if self._should_stop_early(val_metrics):
                print("Early stopping triggered")
                break
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log training and validation metrics.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Log to console
        print(f"\nEpoch {self.current_epoch + 1}:")
        print("Training metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("Validation metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Log to file
        log_dir = Path(self.config["logging"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "training_log.json"
        log_data = {
            "epoch": self.current_epoch + 1,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_data)
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)
    
    def _save_checkpoint(self, val_metrics: Dict[str, float]):
        """Save model checkpoint.
        
        Args:
            val_metrics: Validation metrics
        """
        current_metric = val_metrics[self.config["checkpointing"]["monitor"]]
        is_best = (
            current_metric > self.best_metric
            if self.config["checkpointing"]["mode"] == "max"
            else current_metric < self.best_metric
        )
        
        if is_best:
            self.best_metric = current_metric
            self.early_stopping_counter = 0
            
            # Save best model
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": self.current_epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_metric": self.best_metric,
                "config": self.config
            }, checkpoint_path)
        else:
            self.early_stopping_counter += 1
        
        # Save regular checkpoint if configured
        if (
            not self.config["checkpointing"]["save_best_only"]
            and (self.current_epoch + 1) % self.config["logging"]["save_frequency"] == 0
        ):
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"
            torch.save({
                "epoch": self.current_epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_metric": self.best_metric,
                "config": self.config
            }, checkpoint_path)
    
    def _should_stop_early(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should be stopped early.
        
        Args:
            val_metrics: Validation metrics
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.early_stopping_counter >= self.config["training"]["early_stopping"]["patience"]:
            return True
        return False

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0):
        """Initialize focal loss.
        
        Args:
            gamma: Focusing parameter
        """
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            input: Model predictions
            target: Ground truth labels
            
        Returns:
            Focal loss value
        """
        ce_loss = self.ce(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss 