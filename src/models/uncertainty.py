"""
Uncertainty Quantification per MSHA Network
Implementa Bayesian Dropout e stima dell'incertezza
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BayesianDropout(nn.Module):
    """
    Bayesian Dropout che rimane attivo anche durante inference
    per stima dell'incertezza
    """
    
    def __init__(self, dropout_rate=0.1):
        super(BayesianDropout, self).__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        """
        Applica dropout anche durante inference se uncertainty=True
        """
        if self.training or hasattr(self, 'uncertainty_mode'):
            return F.dropout(x, p=self.dropout_rate, training=True)
        else:
            return x


class UncertaintyQuantification(nn.Module):
    """
    Modulo per quantificare l'incertezza delle predizioni
    Utilizza Monte Carlo Dropout
    """
    
    def __init__(self, input_dim, hidden_dim=512, num_classes=5, dropout_rate=0.3):
        super(UncertaintyQuantification, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Layers con Bayesian Dropout
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            BayesianDropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            BayesianDropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(inplace=True),
            BayesianDropout(dropout_rate),
        )
        
        # Output layers
        self.mean_head = nn.Linear(hidden_dim // 4, num_classes)
        self.variance_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, num_classes),
            nn.Softplus()  # Assicura valori positivi per varianza
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Inizializzazione dei pesi"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass standard
        """
        features = self.layers(x)
        mean = self.mean_head(features)
        variance = self.variance_head(features)
        
        return mean, variance
    
    def monte_carlo_inference(self, x, num_samples=100):
        """
        Inference con Monte Carlo Dropout per stima incertezza
        
        Args:
            x: Input tensor [B, input_dim]
            num_samples: Numero di campioni MC
            
        Returns:
            mean_prediction: Media delle predizioni [B, num_classes]
            uncertainty: Incertezza epistemic [B, num_classes]
            aleatoric_uncertainty: Incertezza aleatoric [B, num_classes]
        """
        # Abilita uncertainty mode
        for module in self.modules():
            if isinstance(module, BayesianDropout):
                module.uncertainty_mode = True
        
        predictions = []
        variances = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                mean, variance = self.forward(x)
                predictions.append(torch.softmax(mean, dim=-1))
                variances.append(variance)
        
        # Disabilita uncertainty mode
        for module in self.modules():
            if isinstance(module, BayesianDropout):
                if hasattr(module, 'uncertainty_mode'):
                    delattr(module, 'uncertainty_mode')
        
        # Calcola statistiche
        predictions = torch.stack(predictions, dim=0)  # [num_samples, B, num_classes]
        variances = torch.stack(variances, dim=0)      # [num_samples, B, num_classes]
        
        # Media delle predizioni
        mean_prediction = torch.mean(predictions, dim=0)
        
        # Incertezza epistemics (varianza tra predizioni)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        # Incertezza aleatoric (media delle varianze)
        aleatoric_uncertainty = torch.mean(variances, dim=0)
        
        # Incertezza totale
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'mean_prediction': mean_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'confidence': 1.0 - total_uncertainty.mean(dim=-1, keepdim=True)
        }


class UncertaintyLoss(nn.Module):
    """
    Loss function che considera l'incertezza aleatoric
    """
    
    def __init__(self, reduction='mean'):
        super(UncertaintyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, mean_pred, variance_pred, targets):
        """
        Args:
            mean_pred: Predizioni medie [B, num_classes]
            variance_pred: Varianze predette [B, num_classes] 
            targets: Target labels [B] (long tensor)
        """
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=mean_pred.size(-1)).float()
        
        # Gaussian likelihood loss
        precision = 1.0 / (variance_pred + 1e-8)  # Evita divisione per zero
        
        # Log likelihood Gaussiano
        log_likelihood = -0.5 * precision * (mean_pred - targets_one_hot) ** 2 \
                        - 0.5 * torch.log(variance_pred + 1e-8) \
                        - 0.5 * np.log(2 * np.pi)
        
        # Negative log likelihood
        nll = -log_likelihood.sum(dim=-1)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:
            return nll


class ConfidenceCalibration(nn.Module):
    """
    Modulo per calibrazione della confidenza
    """
    
    def __init__(self, num_bins=15):
        super(ConfidenceCalibration, self).__init__()
        self.num_bins = num_bins
        
    def forward(self, confidences, accuracies):
        """
        Calcola Expected Calibration Error (ECE)
        
        Args:
            confidences: Confidenze predette [N]
            accuracies: Accuratezze effettive [N] (0 o 1)
        """
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Samples in bin
            in_bin = (confidences > bin_lower.item()) & (confidences <= bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


class EnsembleUncertainty(nn.Module):
    """
    Uncertainty estimation tramite ensemble di modelli
    """
    
    def __init__(self, models):
        super(EnsembleUncertainty, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
    
    def forward(self, x):
        """
        Forward pass con ensemble
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]  # Prendi solo le predizioni
                predictions.append(torch.softmax(pred, dim=-1))
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, B, num_classes]
        
        # Calcola statistiche ensemble
        mean_prediction = torch.mean(predictions, dim=0)
        variance_prediction = torch.var(predictions, dim=0)
        confidence = 1.0 - variance_prediction.mean(dim=-1, keepdim=True)
        
        return {
            'mean_prediction': mean_prediction,
            'uncertainty': variance_prediction,
            'confidence': confidence,
            'individual_predictions': predictions
        }
