"""
Method-level evaluation utilities for deepfake detection.

Provides functions to analyze model performance broken down by deepfake generation method.
"""

from typing import Dict, List, Any
import numpy as np


def calculate_method_statistics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata_list: List[Dict[str, Any]],
    method_key: str = "method"
) -> Dict[str, Dict[str, float]]:
    """
    Calculate classification metrics per deepfake generation method.
    
    Args:
        predictions: Array of predicted class labels (shape: [N])
        targets: Array of true class labels (shape: [N])
        metadata_list: List of metadata dicts, one per sample
        method_key: Key in metadata dict for the method name
        
    Returns:
        Dictionary mapping method names to their metrics:
        {
            "method_name": {
                "accuracy": float,
                "precision": float,
                "recall": float,
                "f1": float,
                "count": int,
                "correct": int
            }
        }
    """
    # Group samples by method
    method_groups: Dict[str, Dict[str, List]] = {}
    
    for i, meta in enumerate(metadata_list):
        method = meta.get(method_key, "unknown")
        if method not in method_groups:
            method_groups[method] = {"preds": [], "targets": []}
        method_groups[method]["preds"].append(predictions[i])
        method_groups[method]["targets"].append(targets[i])
    
    # Calculate metrics per method
    results = {}
    for method, data in method_groups.items():
        preds = np.array(data["preds"])
        tgts = np.array(data["targets"])
        
        results[method] = _calculate_binary_metrics(preds, tgts)
    
    return results


def _calculate_binary_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """Calculate binary classification metrics for a set of predictions."""
    count = len(predictions)
    correct = int((predictions == targets).sum())
    accuracy = correct / count if count > 0 else 0.0
    
    # For binary classification: positive class = 1 (fake)
    tp = int(((predictions == 1) & (targets == 1)).sum())
    fp = int(((predictions == 1) & (targets == 0)).sum())
    fn = int(((predictions == 0) & (targets == 1)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "count": count,
        "correct": correct
    }


def format_method_statistics(stats: Dict[str, Dict[str, float]]) -> str:
    """Format method statistics as a readable table string."""
    if not stats:
        return "No method statistics available."
    
    # Sort by count descending
    sorted_methods = sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)
    
    lines = [
        f"{'Method':<15} {'Count':>7} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}",
        "-" * 60
    ]
    
    for method, m in sorted_methods:
        lines.append(
            f"{method:<15} {m['count']:>7} {m['accuracy']:>8.4f} "
            f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}"
        )
    
    return "\n".join(lines)
