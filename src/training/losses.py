"""
Loss functions for training MSHA Network
Includes Focal Loss and Uncertainty Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Paper: "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: Class weights [C] or None for uniform weights
            gamma: Focusing parameter (default: 2.0)
            reduction: 'mean', 'sum' o 'none'
            label_smoothing: Label smoothing factor
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        if alpha is not None:
            self.alpha = torch.FloatTensor(alpha)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [N, C]
            targets: Ground truth [N] (class indices)
        """
        device = inputs.device
        if self.alpha is not None:
            self.alpha = self.alpha.to(device)
        
        # Calcola cross entropy
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        
        # Calcola probabilità predette
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Applica reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with automatic weights based on class frequency
    """
    
    def __init__(
        self,
        class_frequencies: List[int],
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super(WeightedFocalLoss, self).__init__()
        
        # Calcola pesi automatici (inversamente proporzionali alla frequenza)
        total_samples = sum(class_frequencies)
        num_classes = len(class_frequencies)
        
        alpha = []
        for freq in class_frequencies:
            weight = total_samples / (num_classes * freq)
            alpha.append(weight)
        
        # Normalizza i pesi
        alpha = np.array(alpha)
        alpha = alpha / alpha.sum() * num_classes
        
        self.focal_loss = FocalLoss(
            alpha=alpha.tolist(),
            gamma=gamma,
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(inputs, targets)


class OrdinalLoss(nn.Module):
    """
    Ordinal Loss per classi ordinate (severity levels)
    Considera l'ordinamento naturale delle classi di retinopatia
    """
    
    def __init__(self, num_classes: int = 5, reduction: str = 'mean'):
        super(OrdinalLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [N, C]
            targets: True classes [N]
        """
        batch_size = inputs.size(0)
        
        # Crea distance matrix
        device = inputs.device
        class_indices = torch.arange(self.num_classes, device=device).float()
        target_expanded = targets.unsqueeze(1).float()  # [N, 1]
        class_expanded = class_indices.unsqueeze(0)     # [1, C]
        
        # Distanza tra classe predetta e vera
        distances = torch.abs(class_expanded - target_expanded)  # [N, C]
        
        # Pesi basati sulla distanza (più lontano = penalità maggiore)
        weights = 1.0 + distances
        
        # Cross entropy pesato
        log_probs = F.log_softmax(inputs, dim=-1)
        weighted_nll = -weights * log_probs
        
        # Seleziona loss per classe vera
        ordinal_loss = weighted_nll.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        if self.reduction == 'mean':
            return ordinal_loss.mean()
        elif self.reduction == 'sum':
            return ordinal_loss.sum()
        else:
            return ordinal_loss


class UncertaintyLoss(nn.Module):
    """
    Loss che combina predizione e incertezza aleatoric
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(UncertaintyLoss, self).__init__()
        self.reduction = reduction
    
    def forward(
        self, 
        mean_pred: torch.Tensor, 
        variance_pred: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mean_pred: Predizioni medie [N, C]
            variance_pred: Varianze predette [N, C]
            targets: Target classes [N]
        """
        # Converti targets a one-hot
        targets_one_hot = F.one_hot(targets, num_classes=mean_pred.size(-1)).float()
        
        # Calcola precision (inverso della varianza)
        precision = 1.0 / (variance_pred + 1e-8)
        
        # Gaussian negative log-likelihood
        squared_error = (mean_pred - targets_one_hot) ** 2
        nll = 0.5 * precision * squared_error + 0.5 * torch.log(variance_pred + 1e-8)
        
        # Sum over classes
        loss = nll.sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Loss combinata che include Focal Loss, Uncertainty Loss e regolarizzazione
    """
    
    def __init__(
        self,
        focal_alpha: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        uncertainty_weight: float = 0.1,
        ordinal_weight: float = 0.2,
        label_smoothing: float = 0.1,
        use_uncertainty: bool = True,
        use_ordinal: bool = True
    ):
        super(CombinedLoss, self).__init__()
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
        
        self.uncertainty_loss = UncertaintyLoss() if use_uncertainty else None
        self.ordinal_loss = OrdinalLoss() if use_ordinal else None
        
        self.uncertainty_weight = uncertainty_weight
        self.ordinal_weight = ordinal_weight
        self.use_uncertainty = use_uncertainty
        self.use_ordinal = use_ordinal
    
    def forward(
        self, 
        predictions: dict, 
        targets: torch.Tensor
    ) -> dict:
        """
        Args:
            predictions: Dict con 'logits' e opzionalmente 'variance'
            targets: Target classes [N]
        
        Returns:
            loss_dict: Dictionary con loss components
        """
        logits = predictions['logits']
        
        # Focal loss principale
        focal_loss = self.focal_loss(logits, targets)
        total_loss = focal_loss
        
        loss_dict = {
            'focal_loss': focal_loss,
            'total_loss': focal_loss
        }
        
        # Uncertainty loss
        if self.use_uncertainty and 'variance' in predictions:
            variance = predictions['variance']
            uncertainty_loss = self.uncertainty_loss(logits, variance, targets)
            total_loss = total_loss + self.uncertainty_weight * uncertainty_loss
            
            loss_dict['uncertainty_loss'] = uncertainty_loss
            loss_dict['total_loss'] = total_loss
        
        # Ordinal loss
        if self.use_ordinal:
            ordinal_loss = self.ordinal_loss(logits, targets)
            total_loss = total_loss + self.ordinal_weight * ordinal_loss
            
            loss_dict['ordinal_loss'] = ordinal_loss
            loss_dict['total_loss'] = total_loss
        
        return loss_dict


class KappaLoss(nn.Module):
    """
    Loss basata su Quadratic Weighted Kappa
    Ottimizza direttamente la metrica di valutazione
    """
    
    def __init__(self, num_classes: int = 5, reduction: str = 'mean'):
        super(KappaLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        
        # Matrice dei pesi per kappa quadratico
        self.weight_matrix = self._create_weight_matrix()
    
    def _create_weight_matrix(self) -> torch.Tensor:
        """Crea matrice dei pesi per kappa quadratico"""
        weights = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                weights[i, j] = ((i - j) ** 2) / ((self.num_classes - 1) ** 2)
        return weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Differentiable approximation di quadratic weighted kappa
        """
        device = inputs.device
        self.weight_matrix = self.weight_matrix.to(device)
        
        # Soft predictions
        pred_probs = F.softmax(inputs, dim=-1)
        
        # True distribution (one-hot)
        true_dist = F.one_hot(targets, self.num_classes).float()
        
        # Confusion matrix (soft)
        confusion_matrix = torch.mm(true_dist.t(), pred_probs)
        
        # Marginal distributions
        pred_marginal = pred_probs.mean(0)
        true_marginal = true_dist.mean(0)
        
        # Expected matrix under independence
        expected = torch.outer(true_marginal, pred_marginal) * len(targets)
        
        # Weighted sums
        weighted_confusion = (confusion_matrix * self.weight_matrix).sum()
        weighted_expected = (expected * self.weight_matrix).sum()
        
        # Kappa computation
        kappa = 1.0 - weighted_confusion / (weighted_expected + 1e-8)
        
        # Convert to loss (maximize kappa = minimize negative kappa)
        loss = -kappa
        
        return loss


class ConsistencyLoss(nn.Module):
    """
    Loss di consistenza per Test Time Augmentation
    """
    
    def __init__(self, temperature: float = 3.0):
        super(ConsistencyLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, predictions_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            predictions_list: Lista di predizioni da diverse augmentations [List[N, C]]
        """
        if len(predictions_list) < 2:
            return torch.tensor(0.0, device=predictions_list[0].device)
        
        # Calcola predizione media
        mean_pred = torch.stack(predictions_list).mean(0)
        
        # KL divergence tra ogni predizione e la media
        consistency_loss = 0.0
        for pred in predictions_list:
            soft_pred = F.log_softmax(pred / self.temperature, dim=-1)
            soft_mean = F.softmax(mean_pred / self.temperature, dim=-1)
            
            kl_div = F.kl_div(soft_pred, soft_mean, reduction='batchmean')
            consistency_loss += kl_div
        
        return consistency_loss / len(predictions_list)


# Factory function per creare loss appropriate
def create_loss_function(config: dict, class_frequencies: Optional[List[int]] = None):
    """
    Factory per creare loss function basata su config
    """
    loss_type = config.get('type', 'combined')
    
    if loss_type == 'focal':
        if class_frequencies:
            return WeightedFocalLoss(
                class_frequencies=class_frequencies,
                gamma=config.get('focal_gamma', 2.0),
                label_smoothing=config.get('label_smoothing', 0.0)
            )
        else:
            return FocalLoss(
                alpha=config.get('focal_alpha'),
                gamma=config.get('focal_gamma', 2.0),
                label_smoothing=config.get('label_smoothing', 0.0)
            )
    
    elif loss_type == 'combined':
        return CombinedLoss(
            focal_alpha=config.get('focal_alpha'),
            focal_gamma=config.get('focal_gamma', 2.0),
            uncertainty_weight=config.get('uncertainty_weight', 0.1),
            ordinal_weight=config.get('ordinal_weight', 0.2),
            label_smoothing=config.get('label_smoothing', 0.1),
            use_uncertainty=config.get('use_uncertainty', True),
            use_ordinal=config.get('use_ordinal', True)
        )
    
    elif loss_type == 'kappa':
        return KappaLoss(num_classes=config.get('num_classes', 5))
    
    else:
        raise ValueError(f"Loss type '{loss_type}' non supportato")
