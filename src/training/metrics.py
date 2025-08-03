from typing import Dict, Any, List, Tuple
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class MetricsCalculator:
    """Class for calculating and visualizing evaluation metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the metrics calculator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metrics = {}
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor, probs: torch.Tensor):
        """Update metrics with new batch of predictions.
        
        Args:
            pred: Predicted labels
            target: Ground truth labels
            probs: Predicted probabilities
        """
        self.predictions.extend(pred.cpu().numpy())
        self.targets.extend(target.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary containing computed metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        metrics = {}
        for metric in self.config["evaluation"]["metrics"]:
            if metric == "accuracy":
                metrics[metric] = accuracy_score(targets, predictions)
            elif metric == "precision":
                metrics[metric] = precision_score(targets, predictions, average="binary")
            elif metric == "recall":
                metrics[metric] = recall_score(targets, predictions, average="binary")
            elif metric == "f1_score":
                metrics[metric] = f1_score(targets, predictions, average="binary")
            elif metric == "auc_roc":
                metrics[metric] = roc_auc_score(targets, probabilities[:, 1])
            elif metric == "confusion_matrix":
                metrics[metric] = confusion_matrix(targets, predictions).tolist()
            elif metric == "specificity":
                # True Negative Rate
                tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
                metrics[metric] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            elif metric == "balanced_accuracy":
                metrics[metric] = (recall_score(targets, predictions, average="binary") + 
                                 (tn / (tn + fp) if (tn + fp) > 0 else 0.0)) / 2
            elif metric == "matthews_correlation":
                metrics[metric] = matthews_corrcoef(targets, predictions)
            elif metric == "cohen_kappa":
                metrics[metric] = cohen_kappa_score(targets, predictions)
            elif metric == "average_precision":
                metrics[metric] = average_precision_score(targets, probabilities[:, 1])
        

        
        # Add classification report
        metrics["classification_report"] = classification_report(
            targets, predictions, target_names=["Real", "Fake"], output_dict=True
        )
        
        self.metrics = metrics
        return metrics
    
    def visualize(self, save_dir: str):
        """Generate and save visualization plots.
        
        Args:
            save_dir: Directory to save plots
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot confusion matrix
        if "confusion_matrix" in self.metrics:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                self.metrics["confusion_matrix"],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"]
            )
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.savefig(save_dir / "confusion_matrix.png")
            plt.close()
        
        # Plot ROC curve
        if "auc_roc" in self.metrics:
            fpr, tpr, _ = roc_curve(self.targets, self.probabilities[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {self.metrics['auc_roc']:.3f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.savefig(save_dir / "roc_curve.png")
            plt.close()
        
        # Plot precision-recall curve
        precision, recall, _ = precision_recall_curve(self.targets, self.probabilities[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.savefig(save_dir / "precision_recall_curve.png")
        plt.close()
        

        
        # Save metrics to file
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def analyze_errors(self, data_loader, model, device: str):
        """Analyze misclassified samples with comprehensive error analysis.
        
        Args:
            data_loader: Data loader containing test data
            model: Trained model
            device: Device to run inference on
        """
        if not self.config["evaluation"]["error_analysis"]["enabled"]:
            return
        
        misclassified = []
        error_patterns = {
            "false_positives": [],  # Real classified as Fake
            "false_negatives": [],  # Fake classified as Real
            "high_confidence_errors": [],
            "low_confidence_errors": []
        }
        
        confidence_threshold = self.config["evaluation"]["error_analysis"]["confidence_threshold"]
        low_confidence_threshold = 0.5  # Threshold for low confidence errors
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                # Find misclassified samples
                mask = pred != target
                if mask.any():
                    for i in range(len(mask)):
                        if mask[i]:
                            confidence = probs[i, pred[i]].item()
                            sample_idx = batch_idx * data_loader.batch_size + i
                            true_label = target[i].item()
                            predicted_label = pred[i].item()
                            
                            error_info = {
                                "sample_idx": sample_idx,
                                "true_label": true_label,
                                "predicted_label": predicted_label,
                                "confidence": confidence,
                                "true_prob": probs[i, true_label].item(),
                                "pred_prob": confidence,
                                "confidence_gap": abs(confidence - probs[i, true_label].item())
                            }
                            
                            misclassified.append(error_info)
                            
                            # Categorize errors
                            if true_label == 0 and predicted_label == 1:  # Real -> Fake
                                error_patterns["false_positives"].append(error_info)
                            elif true_label == 1 and predicted_label == 0:  # Fake -> Real
                                error_patterns["false_negatives"].append(error_info)
                            
                            # Categorize by confidence
                            if confidence >= confidence_threshold:
                                error_patterns["high_confidence_errors"].append(error_info)
                            elif confidence <= low_confidence_threshold:
                                error_patterns["low_confidence_errors"].append(error_info)
        
        # Generate error analysis summary
        error_summary = {
            "total_errors": len(misclassified),
            "false_positives_count": len(error_patterns["false_positives"]),
            "false_negatives_count": len(error_patterns["false_negatives"]),
            "high_confidence_errors_count": len(error_patterns["high_confidence_errors"]),
            "low_confidence_errors_count": len(error_patterns["low_confidence_errors"]),
            "avg_confidence_errors": np.mean([e["confidence"] for e in misclassified]) if misclassified else 0,
            "avg_confidence_gap": np.mean([e["confidence_gap"] for e in misclassified]) if misclassified else 0,
            "most_confident_error": max(misclassified, key=lambda x: x["confidence"]) if misclassified else None,
            "least_confident_error": min(misclassified, key=lambda x: x["confidence"]) if misclassified else None
        }
        
        # Add per-category statistics
        if error_patterns["false_positives"]:
            error_summary["fp_avg_confidence"] = np.mean([e["confidence"] for e in error_patterns["false_positives"]])
        if error_patterns["false_negatives"]:
            error_summary["fn_avg_confidence"] = np.mean([e["confidence"] for e in error_patterns["false_negatives"]])
        
        # Save comprehensive error analysis results
        if self.config["evaluation"]["error_analysis"]["save_misclassified"]:
            save_dir = Path(self.config["evaluation"]["visualization"]["save_dir"])
            
            # Save detailed error analysis
            with open(save_dir / "error_analysis_detailed.json", "w") as f:
                json.dump({
                    "error_summary": error_summary,
                    "error_patterns": error_patterns,
                    "all_misclassified": misclassified
                }, f, indent=2)
            
            # Save error patterns separately for easier analysis
            with open(save_dir / "error_patterns.json", "w") as f:
                json.dump(error_patterns, f, indent=2)
            
            # Generate error analysis visualization
            self._visualize_error_analysis(error_summary, error_patterns, save_dir)
        
        return misclassified, error_summary
    
    def _visualize_error_analysis(self, error_summary, error_patterns, save_dir):
        """Generate visualizations for error analysis."""
        
        # Error distribution by type
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error type distribution
        error_types = ["False Positives", "False Negatives"]
        error_counts = [error_summary["false_positives_count"], error_summary["false_negatives_count"]]
        ax1.bar(error_types, error_counts, color=['red', 'blue'], alpha=0.7)
        ax1.set_title("Error Distribution by Type")
        ax1.set_ylabel("Count")
        for i, v in enumerate(error_counts):
            ax1.text(i, v + 0.1, str(v), ha='center')
        
        # Confidence distribution of errors
        if error_patterns["false_positives"] and error_patterns["false_negatives"]:
            fp_confidences = [e["confidence"] for e in error_patterns["false_positives"]]
            fn_confidences = [e["confidence"] for e in error_patterns["false_negatives"]]
            
            ax2.hist(fp_confidences, bins=20, alpha=0.7, label="False Positives", color='red')
            ax2.hist(fn_confidences, bins=20, alpha=0.7, label="False Negatives", color='blue')
            ax2.set_title("Confidence Distribution of Errors")
            ax2.set_xlabel("Confidence")
            ax2.set_ylabel("Frequency")
            ax2.legend()
        
        # Error confidence vs true probability
        if misclassified:
            confidences = [e["confidence"] for e in misclassified]
            true_probs = [e["true_prob"] for e in misclassified]
            ax3.scatter(confidences, true_probs, alpha=0.6)
            ax3.plot([0, 1], [0, 1], 'r--', alpha=0.5)
            ax3.set_title("Error Confidence vs True Probability")
            ax3.set_xlabel("Predicted Confidence")
            ax3.set_ylabel("True Class Probability")
        
        # Confidence gap distribution
        if misclassified:
            confidence_gaps = [e["confidence_gap"] for e in misclassified]
            ax4.hist(confidence_gaps, bins=20, alpha=0.7, color='green')
            ax4.set_title("Distribution of Confidence Gaps")
            ax4.set_xlabel("Confidence Gap")
            ax4.set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(save_dir / "error_analysis_visualization.png")
        plt.close() 