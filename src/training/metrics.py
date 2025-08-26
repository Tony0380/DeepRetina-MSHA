"""
Specific metrics for medical evaluation - FIXED AUC calculation
Includes Quadratic Kappa, AUC-ROC, Sensitivity, Specificity
"""

import torch
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, 
    roc_auc_score, 
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MedicalMetrics:
    """
    Class for calculating specific medical metrics
    """
    
    def __init__(self, num_classes: int = 5, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Grade {i}" for i in range(num_classes)]
        
        # Specific grade names for diabetic retinopathy
        self.dr_grades = [
            "No DR",           # Grade 0
            "Mild",            # Grade 1  
            "Moderate",        # Grade 2
            "Severe",          # Grade 3
            "Proliferative"    # Grade 4
        ]
    
    def quadratic_weighted_kappa(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Quadratic Weighted Kappa
        Main metric for diabetic retinopathy
        """
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    def linear_weighted_kappa(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate Linear Weighted Kappa
        """
        return cohen_kappa_score(y_true, y_pred, weights='linear')
    
    def multi_class_auc(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate AUC-ROC for each class (one-vs-rest) - FIXED VERSION
        """
        auc_scores = {}
        
        try:
            # Ensure we have all classes represented
            unique_classes = np.unique(y_true)
            
            # If we don't have all classes in this batch, skip detailed AUC calculation
            if len(unique_classes) < 2:
                # Not enough classes for meaningful AUC calculation
                auc_scores = {f'class_{i}': 0.0 for i in range(self.num_classes)}
                auc_scores['macro'] = 0.0
                return auc_scores
            
            # Try multiclass AUC only if we have multiple classes
            if len(unique_classes) >= 2:
                # Use label_binarize for proper multiclass handling
                from sklearn.preprocessing import label_binarize
                
                # Binarize labels for all possible classes (not just those in batch)
                y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))
                
                # Handle binary case (only 2 classes total)
                if self.num_classes == 2:
                    auc_macro = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multiclass case - only calculate if we have enough samples
                    if y_true_binarized.shape[1] > 1:
                        auc_macro = roc_auc_score(
                            y_true_binarized, 
                            y_prob, 
                            multi_class='ovr', 
                            average='macro'
                        )
                    else:
                        # Fallback for edge cases
                        auc_macro = 0.0
                
                auc_scores['macro'] = auc_macro
                
                # AUC for each class (only for classes present in batch)
                for i in range(self.num_classes):
                    if i in unique_classes:
                        try:
                            y_true_binary = (y_true == i).astype(int)
                            if len(np.unique(y_true_binary)) == 2:  # Both positive and negative samples
                                auc_class = roc_auc_score(y_true_binary, y_prob[:, i])
                                auc_scores[f'class_{i}_{self.dr_grades[i]}'] = auc_class
                            else:
                                auc_scores[f'class_{i}_{self.dr_grades[i]}'] = 0.0
                        except Exception:
                            auc_scores[f'class_{i}_{self.dr_grades[i]}'] = 0.0
                    else:
                        auc_scores[f'class_{i}_{self.dr_grades[i]}'] = 0.0
            
        except Exception as e:
            print(f"Warning: Error in AUC calculation: {e}")
            # Return default values on error
            auc_scores = {f'class_{i}': 0.0 for i in range(self.num_classes)}
            auc_scores['macro'] = 0.0
        
        return auc_scores
    
    def sensitivity_specificity(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate Sensitivity and Specificity for each class
        """
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        metrics = {}
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            # Sensitivity (Recall) = TP / (TP + FN)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Specificity = TN / (TN + FP)  
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            metrics[f'class_{i}_{self.dr_grades[i]}'] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }
        
        return metrics
    
    def clinical_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all relevant clinical metrics - FIXED VERSION
        """
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Quadratic Weighted Kappa (main metric)
        metrics['quadratic_kappa'] = self.quadratic_weighted_kappa(y_true, y_pred)
        metrics['linear_kappa'] = self.linear_weighted_kappa(y_true, y_pred)
        
        # Precision, Recall, F1 macro
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['precision_macro'] = precision
        metrics['recall_macro'] = recall
        metrics['f1_macro'] = f1
        
        # Precision, Recall, F1 weighted
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = precision_w
        metrics['recall_weighted'] = recall_w
        metrics['f1_weighted'] = f1_w
        
        # AUC scores if probabilities are available - FIXED
        if y_prob is not None:
            try:
                auc_scores = self.multi_class_auc(y_true, y_prob)
                metrics.update(auc_scores)
            except Exception as e:
                print(f"Warning: Skipping AUC calculation due to error: {e}")
                # Continue without AUC metrics
        
        # Sensitivity e Specificity
        try:
            sens_spec = self.sensitivity_specificity(y_true, y_pred)
            for class_name, class_metrics in sens_spec.items():
                metrics[f'{class_name}_sensitivity'] = class_metrics['sensitivity']
                metrics[f'{class_name}_specificity'] = class_metrics['specificity']
        except Exception as e:
            print(f"Warning: Error calculating sensitivity/specificity: {e}")
        
        return metrics
    
    def per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Detailed metrics for each class
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        per_class = {}
        for i in range(self.num_classes):
            per_class[f'class_{i}_{self.dr_grades[i]}'] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        
        return per_class
    
    def off_by_one_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Accuracy considering ±1 class errors as correct
        Important for ordered classes like severity levels
        """
        diff = np.abs(y_true - y_pred)
        correct = (diff <= 1).sum()
        return correct / len(y_true)
    
    def create_confusion_matrix_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create confusion matrix plot
        """
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.dr_grades,
            yticklabels=self.dr_grades,
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def clinical_interpretation(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Clinical interpretation of metrics
        """
        interpretation = {}
        
        # Quadratic Kappa interpretation
        kappa = metrics.get('quadratic_kappa', 0)
        if kappa >= 0.81:
            kappa_interp = "Excellent agreement"
        elif kappa >= 0.61:
            kappa_interp = "Substantial agreement"
        elif kappa >= 0.41:
            kappa_interp = "Moderate agreement"
        elif kappa >= 0.21:
            kappa_interp = "Fair agreement"
        else:
            kappa_interp = "Poor agreement"
        
        interpretation['kappa_interpretation'] = kappa_interp
        
        # Accuracy interpretation
        accuracy = metrics.get('accuracy', 0)
        if accuracy >= 0.90:
            acc_interp = "Excellent performance"
        elif accuracy >= 0.80:
            acc_interp = "Good performance"
        elif accuracy >= 0.70:
            acc_interp = "Acceptable performance"
        else:
            acc_interp = "Poor performance"
        
        interpretation['accuracy_interpretation'] = acc_interp
        
        # Sensitivity/Specificity for critical classes
        for grade in ['Severe', 'Proliferative']:
            sens_key = f'class_{self.dr_grades.index(grade)}_{grade}_sensitivity'
            spec_key = f'class_{self.dr_grades.index(grade)}_{grade}_specificity'
            
            if sens_key in metrics and spec_key in metrics:
                sens = metrics[sens_key]
                spec = metrics[spec_key]
                
                if sens >= 0.90 and spec >= 0.90:
                    clinical_value = "Excellent clinical utility"
                elif sens >= 0.80 and spec >= 0.80:
                    clinical_value = "Good clinical utility"
                elif sens >= 0.70 or spec >= 0.70:
                    clinical_value = "Limited clinical utility"
                else:
                    clinical_value = "Insufficient clinical utility"
                
                interpretation[f'{grade.lower()}_clinical_utility'] = clinical_value
        
        return interpretation


class MetricsTracker:
    """
    Tracker to accumulate metrics during training/validation
    """
    
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.metrics_calculator = MedicalMetrics(num_classes)
        self.reset()
    
    def reset(self):
        """Reset the tracker"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None,
        loss: Optional[float] = None
    ):
        """
        Update tracker with new data
        """
        # Convert to numpy
        pred_numpy = predictions.detach().cpu().numpy()
        target_numpy = targets.detach().cpu().numpy()
        
        self.predictions.extend(pred_numpy)
        self.targets.extend(target_numpy)
        
        if probabilities is not None:
            prob_numpy = probabilities.detach().cpu().numpy()
            self.probabilities.extend(prob_numpy)
        
        if loss is not None:
            self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Calculate all accumulated metrics - FIXED VERSION
        """
        if not self.predictions:
            return {}
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities) if self.probabilities else None
        
        # Use the fixed clinical_metrics method
        metrics = self.metrics_calculator.clinical_metrics(y_true, y_pred, y_prob)
        
        # Add average loss if available
        if self.losses:
            metrics['avg_loss'] = np.mean(self.losses)
        
        # Add off-by-one accuracy
        metrics['off_by_one_accuracy'] = self.metrics_calculator.off_by_one_accuracy(
            y_true, y_pred
        )
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Return confusion matrix"""
        if not self.predictions:
            return np.zeros((self.num_classes, self.num_classes))
        
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        return confusion_matrix(y_true, y_pred, labels=range(self.num_classes))


class MetricsCalculator:
    """
    Main metrics calculator - convenience wrapper around MedicalMetrics
    Provides the interface expected by evaluation code
    """
    
    def __init__(self, num_classes: int = 5, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.medical_metrics = MedicalMetrics(num_classes, class_names)
        self.class_names = class_names or [f"Grade {i}" for i in range(num_classes)]
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for model evaluation - FIXED VERSION
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_prob: Predicted probabilities (optional)
        
        Returns:
            Dictionary of calculated metrics
        """
        # Get clinical metrics from MedicalMetrics (now with fixed AUC)
        metrics = self.medical_metrics.clinical_metrics(y_true, y_pred, y_prob)
        
        # Add standard metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['quadratic_kappa'] = self.medical_metrics.quadratic_weighted_kappa(y_true, y_pred)
        metrics['off_by_one_accuracy'] = self.medical_metrics.off_by_one_accuracy(y_true, y_pred)
        
        # Add per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['per_class_precision'] = precision.tolist()
        metrics['per_class_recall'] = recall.tolist()
        metrics['per_class_f1'] = f1.tolist()
        metrics['per_class_support'] = support.tolist()
        
        # Add macro AUC if probabilities available - SAFER VERSION
        if y_prob is not None:
            try:
                unique_classes = np.unique(y_true)
                if len(unique_classes) >= 2:
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                    if y_true_bin.shape[1] == 1:  # Binary case
                        if y_prob.shape[1] >= 2:
                            auc_macro = roc_auc_score(y_true, y_prob[:, 1])
                        else:
                            auc_macro = 0.0
                    else:  # Multiclass case
                        auc_macro = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
                    metrics['macro'] = auc_macro
            except Exception as e:
                print(f"Warning: Skipping macro AUC calculation: {e}")
                # Continue without this metric
        
        return metrics