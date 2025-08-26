"""
Analisi dell'incertezza per modelli MSHA
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


class UncertaintyAnalyzer:
    """
    Analizzatore per incertezza predittiva
    """
    
    def __init__(self):
        """Inizializza analyzer"""
        pass
    
    def analyze_uncertainty(
        self,
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analisi completa dell'incertezza
        
        Args:
            uncertainties: Valori di incertezza [N]
            predictions: Predizioni [N]
            targets: Target veri [N] 
            confidences: Confidenze [N] (opzionale)
        
        Returns:
            analysis: Risultati dell'analisi
        """
        analysis = {}
        
        # 1. Statistiche base dell'incertezza
        analysis.update(self._uncertainty_statistics(uncertainties))
        
        # 2. Correlazione uncertainty-accuracy
        analysis.update(self._uncertainty_accuracy_correlation(
            uncertainties, predictions, targets
        ))
        
        # 3. Distribuzione per correttezza predizione
        analysis.update(self._uncertainty_by_correctness(
            uncertainties, predictions, targets
        ))
        
        # 4. Analisi per classe
        analysis.update(self._uncertainty_by_class(
            uncertainties, predictions, targets
        ))
        
        # 5. Soglie di rejection
        analysis.update(self._rejection_analysis(
            uncertainties, predictions, targets
        ))
        
        # 6. Calibrazione se disponibili confidenze
        if confidences is not None:
            analysis.update(self._calibration_analysis(
                confidences, predictions, targets
            ))
        
        return analysis
    
    def _uncertainty_statistics(self, uncertainties: np.ndarray) -> Dict[str, float]:
        """Statistiche descrittive dell'incertezza"""
        return {
            'uncertainty_mean': float(np.mean(uncertainties)),
            'uncertainty_std': float(np.std(uncertainties)),
            'uncertainty_median': float(np.median(uncertainties)),
            'uncertainty_min': float(np.min(uncertainties)),
            'uncertainty_max': float(np.max(uncertainties)),
            'uncertainty_q25': float(np.percentile(uncertainties, 25)),
            'uncertainty_q75': float(np.percentile(uncertainties, 75)),
            'uncertainty_iqr': float(np.percentile(uncertainties, 75) - np.percentile(uncertainties, 25))
        }
    
    def _uncertainty_accuracy_correlation(
        self,
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Correlazione tra incertezza e accuratezza"""
        # Calcola correttezza
        correct = (predictions == targets).astype(float)
        
        # Correlazioni
        pearson_corr, pearson_p = stats.pearsonr(uncertainties, correct)
        spearman_corr, spearman_p = stats.spearmanr(uncertainties, correct)
        
        return {
            'uncertainty_accuracy_pearson': pearson_corr,
            'uncertainty_accuracy_pearson_p': pearson_p,
            'uncertainty_accuracy_spearman': spearman_corr,
            'uncertainty_accuracy_spearman_p': spearman_p
        }
    
    def _uncertainty_by_correctness(
        self,
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Distribuzione incertezza per predizioni corrette/errate"""
        correct_mask = predictions == targets
        
        correct_uncertainties = uncertainties[correct_mask]
        wrong_uncertainties = uncertainties[~correct_mask]
        
        # Test statistico
        if len(correct_uncertainties) > 0 and len(wrong_uncertainties) > 0:
            statistic, p_value = stats.mannwhitneyu(
                wrong_uncertainties, correct_uncertainties, 
                alternative='greater'
            )
        else:
            statistic, p_value = np.nan, np.nan
        
        return {
            'uncertainty_by_correctness': {
                'correct_mean': float(np.mean(correct_uncertainties)) if len(correct_uncertainties) > 0 else np.nan,
                'correct_std': float(np.std(correct_uncertainties)) if len(correct_uncertainties) > 0 else np.nan,
                'wrong_mean': float(np.mean(wrong_uncertainties)) if len(wrong_uncertainties) > 0 else np.nan,
                'wrong_std': float(np.std(wrong_uncertainties)) if len(wrong_uncertainties) > 0 else np.nan,
                'mann_whitney_statistic': float(statistic),
                'mann_whitney_p_value': float(p_value),
                'effect_size': float(np.mean(wrong_uncertainties) - np.mean(correct_uncertainties)) if len(correct_uncertainties) > 0 and len(wrong_uncertainties) > 0 else np.nan
            }
        }
    
    def _uncertainty_by_class(
        self,
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analisi incertezza per classe"""
        unique_classes = np.unique(targets)
        class_analysis = {}
        
        for class_idx in unique_classes:
            class_mask = targets == class_idx
            class_uncertainties = uncertainties[class_mask]
            class_predictions = predictions[class_mask]
            class_targets = targets[class_mask]
            
            if len(class_uncertainties) > 0:
                correct_mask = class_predictions == class_targets
                
                class_analysis[f'class_{int(class_idx)}'] = {
                    'uncertainty_mean': float(np.mean(class_uncertainties)),
                    'uncertainty_std': float(np.std(class_uncertainties)),
                    'accuracy': float(correct_mask.mean()),
                    'sample_count': int(len(class_uncertainties)),
                    'correct_uncertainty_mean': float(np.mean(class_uncertainties[correct_mask])) if correct_mask.sum() > 0 else np.nan,
                    'wrong_uncertainty_mean': float(np.mean(class_uncertainties[~correct_mask])) if (~correct_mask).sum() > 0 else np.nan
                }
        
        return {'uncertainty_by_class': class_analysis}
    
    def _rejection_analysis(
        self,
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analisi per rejection based su uncertainty threshold"""
        # Range di threshold da testare
        thresholds = np.percentile(uncertainties, [50, 60, 70, 80, 90, 95, 99])
        
        rejection_results = {}
        
        for threshold in thresholds:
            # Campioni sopra threshold (da rejettare)
            reject_mask = uncertainties >= threshold
            keep_mask = ~reject_mask
            
            if keep_mask.sum() > 0:
                # Accuracy sui campioni mantenuti
                kept_accuracy = (predictions[keep_mask] == targets[keep_mask]).mean()
                
                # Percentuale rejettata
                rejection_rate = reject_mask.mean()
                
                # Accuracy sui campioni rejettati
                rejected_accuracy = (predictions[reject_mask] == targets[reject_mask]).mean() if reject_mask.sum() > 0 else np.nan
                
                rejection_results[f'threshold_{threshold:.4f}'] = {
                    'rejection_rate': float(rejection_rate),
                    'kept_accuracy': float(kept_accuracy),
                    'rejected_accuracy': float(rejected_accuracy),
                    'kept_samples': int(keep_mask.sum()),
                    'rejected_samples': int(reject_mask.sum())
                }
        
        return {'rejection_analysis': rejection_results}
    
    def _calibration_analysis(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, Any]:
        """Analisi della calibrazione delle confidenze"""
        # Calcola correttezza
        correct = (predictions == targets).astype(float)
        
        # Calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                correct, confidences, n_bins=10, strategy='uniform'
            )
            
            # Expected Calibration Error (ECE)
            ece = self._calculate_ece(confidences, correct, n_bins=10)
            
            # Brier Score
            brier_score = brier_score_loss(correct, confidences)
            
            # Reliability (quanto le confidenze predicono l'accuracy)
            reliability_bins = []
            for i in range(len(fraction_of_positives)):
                if not np.isnan(fraction_of_positives[i]) and not np.isnan(mean_predicted_value[i]):
                    reliability_bins.append({
                        'predicted_confidence': float(mean_predicted_value[i]),
                        'actual_accuracy': float(fraction_of_positives[i]),
                        'calibration_error': float(abs(fraction_of_positives[i] - mean_predicted_value[i]))
                    })
            
            return {
                'calibration_analysis': {
                    'expected_calibration_error': float(ece),
                    'brier_score': float(brier_score),
                    'reliability_bins': reliability_bins,
                    'mean_confidence': float(np.mean(confidences)),
                    'confidence_std': float(np.std(confidences))
                }
            }
            
        except Exception as e:
            return {
                'calibration_analysis': {
                    'error': f"Errore nel calcolo calibrazione: {str(e)}"
                }
            }
    
    def _calculate_ece(
        self, 
        confidences: np.ndarray, 
        accuracies: np.ndarray, 
        n_bins: int = 10
    ) -> float:
        """Calcola Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Campioni in questo bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def plot_uncertainty_analysis(
        self,
        uncertainties: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Crea plot per analisi uncertainty
        
        Args:
            uncertainties: Valori di incertezza
            predictions: Predizioni
            targets: Target veri
            class_names: Nomi delle classi
            save_path: Path per salvare il plot
        
        Returns:
            fig: Figura matplotlib
        """
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(max(max(predictions), max(targets)) + 1)]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribuzione uncertainty per correttezza
        correct_mask = predictions == targets
        
        axes[0, 0].hist(uncertainties[correct_mask], bins=30, alpha=0.7, 
                       label='Correct', color='green', density=True)
        axes[0, 0].hist(uncertainties[~correct_mask], bins=30, alpha=0.7, 
                       label='Wrong', color='red', density=True)
        axes[0, 0].set_xlabel('Uncertainty')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Uncertainty Distribution by Correctness')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Uncertainty vs Accuracy scatter
        axes[0, 1].scatter(uncertainties, correct_mask, alpha=0.6)
        axes[0, 1].set_xlabel('Uncertainty')
        axes[0, 1].set_ylabel('Correct Prediction')
        axes[0, 1].set_title('Uncertainty vs Correctness')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Uncertainty per classe (target)
        unique_classes = np.unique(targets)
        for class_idx in unique_classes:
            class_mask = targets == class_idx
            if class_mask.sum() > 0:
                axes[0, 2].hist(uncertainties[class_mask], bins=20, alpha=0.6,
                              label=class_names[class_idx], density=True)
        
        axes[0, 2].set_xlabel('Uncertainty')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Uncertainty Distribution by True Class')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Box plot uncertainty per classe
        uncertainty_by_class = [uncertainties[targets == i] for i in unique_classes]
        axes[1, 0].boxplot(uncertainty_by_class, labels=[class_names[i] for i in unique_classes])
        axes[1, 0].set_xlabel('True Class')
        axes[1, 0].set_ylabel('Uncertainty')
        axes[1, 0].set_title('Uncertainty Distribution by Class')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Rejection curve
        thresholds = np.percentile(uncertainties, np.linspace(0, 100, 21))
        rejection_rates = []
        accuracies = []
        
        for threshold in thresholds:
            keep_mask = uncertainties <= threshold
            if keep_mask.sum() > 0:
                rejection_rate = 1 - keep_mask.mean()
                accuracy = (predictions[keep_mask] == targets[keep_mask]).mean()
                rejection_rates.append(rejection_rate)
                accuracies.append(accuracy)
        
        axes[1, 1].plot(rejection_rates, accuracies, 'b-o', linewidth=2)
        axes[1, 1].set_xlabel('Rejection Rate')
        axes[1, 1].set_ylabel('Accuracy on Remaining Samples')
        axes[1, 1].set_title('Accuracy vs Rejection Rate')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Uncertainty histogram
        axes[1, 2].hist(uncertainties, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(np.mean(uncertainties), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(uncertainties):.4f}')
        axes[1, 2].axvline(np.median(uncertainties), color='orange', linestyle='--',
                          label=f'Median: {np.median(uncertainties):.4f}')
        axes[1, 2].set_xlabel('Uncertainty')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Overall Uncertainty Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Uncertainty Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
