"""
Clinical metrics calculator specific for diabetic retinopathy
"""

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, cohen_kappa_score,
    roc_auc_score, precision_recall_fscore_support
)
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


class ClinicalMetricsCalculator:
    """
    Specialized calculator for clinical metrics
    """
    
    def __init__(self, class_names: List[str]):
        """
        Args:
            class_names: Names of DR classes
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate complete set of clinical metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_prob: Prediction probabilities
        
        Returns:
            metrics: Complete dictionary of metrics
        """
        metrics = {}
        
        # 1. Basic metrics
        metrics.update(self._calculate_basic_metrics(y_true, y_pred))
        
        # 2. Severity-weighted metrics
        metrics.update(self._calculate_severity_weighted_metrics(y_true, y_pred))
        
        # 3. Per-class metrics
        metrics.update(self._calculate_per_class_metrics(y_true, y_pred))
        
        # 4. Agreement metrics
        metrics.update(self._calculate_agreement_metrics(y_true, y_pred))
        
        # 5. AUC metrics if probabilities available
        if y_prob is not None:
            metrics.update(self._calculate_auc_metrics(y_true, y_prob))
        
        # 6. Clinically relevant error analysis
        metrics.update(self._analyze_clinical_errors(y_true, y_pred))
        
        # 7. Clinical interpretation
        metrics['clinical_interpretation'] = self._clinical_interpretation(metrics)
        
        return metrics
    
    def _calculate_basic_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Basic metrics"""
        from sklearn.metrics import accuracy_score
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 (macro e weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics.update({
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        })
        
        return metrics
    
    def _calculate_severity_weighted_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Severity-weighted metrics for clinical assessment"""
        metrics = {}
        
        # Quadratic Weighted Kappa (main metric for DR)
        metrics['quadratic_kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        metrics['linear_kappa'] = cohen_kappa_score(y_true, y_pred, weights='linear')
        
        # Off-by-one accuracy (±1 class errors acceptable)
        off_by_one_correct = np.abs(y_true - y_pred) <= 1
        metrics['off_by_one_accuracy'] = off_by_one_correct.mean()
        
        # Mean Absolute Error (treats classes as ordinal)
        metrics['mean_absolute_error'] = np.abs(y_true - y_pred).mean()
        
        # Severity-weighted accuracy (errors on severe classes weigh more)
        severity_weights = np.array([1, 2, 3, 4, 5])  # Increasing weights for severity
        weighted_errors = np.abs(y_true - y_pred) * severity_weights[y_true]
        metrics['severity_weighted_mae'] = weighted_errors.mean()
        
        return metrics
    
    def _calculate_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Metrics for each class"""
        per_class = {}
        
        # Precision, Recall, F1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Sensitivity and Specificity per class
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            per_class[self.class_names[i]] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i]),
                'sensitivity': sensitivity,
                'specificity': specificity,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
        
        return {'per_class_metrics': per_class}
    
    def _calculate_agreement_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Inter-rater agreement metrics"""
        metrics = {}
        
        # Cohen's Kappa (unweighted)
        metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Percent agreement
        metrics['percent_agreement'] = (y_true == y_pred).mean()
        
        # Expected agreement by chance
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        n = cm.sum()
        marginal_true = cm.sum(axis=1) / n
        marginal_pred = cm.sum(axis=0) / n
        expected_agreement = (marginal_true * marginal_pred).sum()
        metrics['expected_agreement'] = expected_agreement
        
        return metrics
    
    def _calculate_auc_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """AUC-ROC metrics"""
        metrics = {}
        
        try:
            # AUC macro (one-vs-rest)
            auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            metrics['auc_macro'] = auc_macro
            
            # AUC per class
            for i in range(self.num_classes):
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) == 2:
                    auc_class = roc_auc_score(y_true_binary, y_prob[:, i])
                    metrics[f'auc_{self.class_names[i]}'] = auc_class
            
            # AUC for pathology detection (0 vs 1-4)
            y_pathology = (y_true > 0).astype(int)
            y_prob_pathology = 1 - y_prob[:, 0]  # Probability of having pathology
            if len(np.unique(y_pathology)) == 2:
                metrics['auc_pathology_detection'] = roc_auc_score(y_pathology, y_prob_pathology)
            
            # AUC for severe DR detection (0-2 vs 3-4)
            y_severe = (y_true >= 3).astype(int)
            y_prob_severe = y_prob[:, 3:].sum(axis=1)  # Prob of severe/proliferative
            if len(np.unique(y_severe)) == 2:
                metrics['auc_severe_detection'] = roc_auc_score(y_severe, y_prob_severe)
                
        except Exception as e:
            print(f"Warning: Error in AUC calculation: {e}")
        
        return metrics
    
    def _analyze_clinical_errors(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze errors from clinical perspective"""
        analysis = {}
        
        # Error matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        
        # Under-diagnosis errors (miss severe cases)
        severe_cases = y_true >= 3  # Severe and Proliferative
        under_diagnosed = severe_cases & (y_pred < 3)
        analysis['under_diagnosis_rate'] = under_diagnosed.mean()
        analysis['severe_cases_missed'] = under_diagnosed.sum()
        
        # Over-diagnosis errors (false alarms)
        non_severe_cases = y_true < 3
        over_diagnosed = non_severe_cases & (y_pred >= 3)
        analysis['over_diagnosis_rate'] = over_diagnosed.mean()
        analysis['false_severe_alarms'] = over_diagnosed.sum()
        
        # Major errors (difference > 2 grades)
        major_errors = np.abs(y_true - y_pred) > 2
        analysis['major_error_rate'] = major_errors.mean()
        analysis['major_errors_count'] = major_errors.sum()
        
        # Analysis for critical transitions
        # No DR -> Severe/Proliferative (very serious)
        critical_miss = (y_true >= 3) & (y_pred == 0)
        analysis['critical_miss_rate'] = critical_miss.mean()
        analysis['critical_miss_count'] = critical_miss.sum()
        
        # Normalized confusion matrix analysis
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        analysis['confusion_matrix'] = cm.tolist()
        analysis['confusion_matrix_normalized'] = cm_norm.tolist()
        
        return {'clinical_error_analysis': analysis}
    
    def _clinical_interpretation(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Clinical interpretation of metrics"""
        interpretation = {}
        
        # Quadratic Kappa interpretation
        kappa = metrics.get('quadratic_kappa', 0)
        if kappa >= 0.85:
            kappa_interp = "Excellent agreement - Suitable for screening"
        elif kappa >= 0.75:
            kappa_interp = "Good agreement - Acceptable for screening with supervision"
        elif kappa >= 0.60:
            kappa_interp = "Moderate agreement - Requires improvements"
        else:
            kappa_interp = "Insufficient agreement - Not suitable for clinical use"
        
        interpretation['quadratic_kappa'] = kappa_interp
        
        # Accuracy interpretation
        accuracy = metrics.get('accuracy', 0)
        if accuracy >= 0.90:
            acc_interp = "Excellent performance"
        elif accuracy >= 0.85:
            acc_interp = "Good performance"
        elif accuracy >= 0.75:
            acc_interp = "Acceptable performance"
        else:
            acc_interp = "Insufficient performance"
        
        interpretation['accuracy'] = acc_interp
        
        # Clinical errors interpretation
        under_diag = metrics.get('clinical_error_analysis', {}).get('under_diagnosis_rate', 0)
        if under_diag <= 0.05:
            under_diag_interp = "Low risk of missed severe diagnoses"
        elif under_diag <= 0.10:
            under_diag_interp = "Moderate risk of missed severe diagnoses"
        else:
            under_diag_interp = "High risk of missed severe diagnoses - CRITICAL"
        
        interpretation['under_diagnosis'] = under_diag_interp
        
        # Clinical recommendations
        recommendations = []
        
        if kappa < 0.75:
            recommendations.append("Improve agreement with expert diagnosis")
        
        if under_diag > 0.10:
            recommendations.append("PRIORITY: Reduce false negatives for severe cases")
        
        if accuracy < 0.85:
            recommendations.append("Improve overall accuracy")
        
        # AUC severe detection
        auc_severe = metrics.get('auc_severe_detection', 0)
        if auc_severe < 0.90:
            recommendations.append("Improve detection of severe/proliferative retinopathy")
        
        interpretation['clinical_recommendations'] = recommendations
        
        # Overall clinical readiness
        clinical_ready = (
            kappa >= 0.75 and 
            accuracy >= 0.85 and 
            under_diag <= 0.10
        )
        
        interpretation['clinical_readiness'] = (
            "READY for clinical evaluation" if clinical_ready 
            else "NOT READY - Requires improvements"
        )
        
        return interpretation
    
    def generate_clinical_report(
        self,
        metrics: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate complete clinical report
        
        Args:
            metrics: Calculated metrics
            save_path: Path to save report
        
        Returns:
            report: Report in text format
        """
        report_lines = []
        
        # Header
        report_lines.extend([
            "=" * 80,
            "CLINICAL EVALUATION REPORT",
            "DeepRetina-MSHA Model for Diabetic Retinopathy Classification",
            "=" * 80,
            ""
        ])
        
        # Summary metrics
        report_lines.extend([
            "SUMMARY METRICS:",
            "-" * 40,
            f"Overall Accuracy: {metrics.get('accuracy', 0):.4f}",
            f"Quadratic Weighted Kappa: {metrics.get('quadratic_kappa', 0):.4f}",
            f"Off-by-One Accuracy: {metrics.get('off_by_one_accuracy', 0):.4f}",
            f"Mean Absolute Error: {metrics.get('mean_absolute_error', 0):.4f}",
            ""
        ])
        
        # Clinical error analysis
        if 'clinical_error_analysis' in metrics:
            error_analysis = metrics['clinical_error_analysis']
            report_lines.extend([
                "CLINICAL ERROR ANALYSIS:",
                "-" * 40,
                f"Under-diagnosis Rate (Severe cases missed): {error_analysis.get('under_diagnosis_rate', 0):.4f}",
                f"Over-diagnosis Rate (False severe alarms): {error_analysis.get('over_diagnosis_rate', 0):.4f}",
                f"Major Error Rate (>2 grades difference): {error_analysis.get('major_error_rate', 0):.4f}",
                f"Critical Miss Rate (Severe->No DR): {error_analysis.get('critical_miss_rate', 0):.4f}",
                ""
            ])
        
        # Per-class performance
        if 'per_class_metrics' in metrics:
            report_lines.extend([
                "PER-CLASS PERFORMANCE:",
                "-" * 40
            ])
            
            for class_name, class_metrics in metrics['per_class_metrics'].items():
                report_lines.extend([
                    f"{class_name}:",
                    f"  Sensitivity: {class_metrics['sensitivity']:.4f}",
                    f"  Specificity: {class_metrics['specificity']:.4f}",
                    f"  F1-Score: {class_metrics['f1_score']:.4f}",
                    f"  Support: {class_metrics['support']}",
                    ""
                ])
        
        # Clinical interpretation
        if 'clinical_interpretation' in metrics:
            interp = metrics['clinical_interpretation']
            report_lines.extend([
                "CLINICAL INTERPRETATION:",
                "-" * 40,
                f"Quadratic Kappa: {interp.get('quadratic_kappa', 'N/A')}",
                f"Accuracy: {interp.get('accuracy', 'N/A')}",
                f"Under-diagnosis Risk: {interp.get('under_diagnosis', 'N/A')}",
                f"Clinical Readiness: {interp.get('clinical_readiness', 'N/A')}",
                ""
            ])
            
            if 'clinical_recommendations' in interp:
                report_lines.extend([
                    "CLINICAL RECOMMENDATIONS:",
                    "-" * 40
                ])
                for i, rec in enumerate(interp['clinical_recommendations'], 1):
                    report_lines.append(f"{i}. {rec}")
                report_lines.append("")
        
        # Medical disclaimer
        report_lines.extend([
            "MEDICAL DISCLAIMER:",
            "-" * 40,
            "This model is for RESEARCH PURPOSES ONLY.",
            "NOT intended for clinical diagnosis or patient care.",
            "Always consult qualified healthcare professionals.",
            "=" * 80
        ])
        
        # Join report
        report = "\n".join(report_lines)
        
        # Save if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
