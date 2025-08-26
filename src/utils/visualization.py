"""
Utilities per visualizzazione e interpretazione del modello
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import cv2
from pathlib import Path


def visualize_attention_maps(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Visualizza attention maps del modello MSHA
    
    Args:
        model: Modello MSHA trainato
        images: Batch di immagini [B, C, H, W]
        labels: Labels corrispondenti [B]
        class_names: Nomi delle classi
        save_path: Path per salvare la figura
        figsize: Dimensioni della figura
    
    Returns:
        fig: Figura matplotlib
    """
    device = next(model.parameters()).device
    images = images.to(device)
    
    with torch.no_grad():
        # Ottieni attention weights
        attention_weights = model.get_attention_weights(images)
        
        # Predizioni
        outputs = model(images)
        predictions = torch.argmax(outputs['logits'], dim=1)
        probabilities = torch.softmax(outputs['logits'], dim=1)
    
    # Numero di campioni da visualizzare
    num_samples = min(4, len(images))
    num_attentions = len(attention_weights)
    
    # Crea subplot grid
    fig, axes = plt.subplots(
        num_attentions + 1, 
        num_samples, 
        figsize=figsize
    )
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    # Denormalizza immagini per visualizzazione
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    denorm_images = images * std + mean
    denorm_images = torch.clamp(denorm_images, 0, 1)
    
    for j in range(num_samples):
        # Immagine originale
        ax = axes[0, j]
        img = denorm_images[j].cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        
        # Info predizione
        true_label = labels[j].item()
        pred_label = predictions[j].item()
        confidence = probabilities[j, pred_label].item()
        
        title = f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}\nConf: {confidence:.3f}'
        color = 'green' if true_label == pred_label else 'red'
        
        ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
        
        # Attention maps
        for i, (att_name, att_maps) in enumerate(attention_weights.items()):
            ax = axes[i + 1, j]
            
            # Normalizza attention map
            att_map = att_maps[j, 0].cpu().numpy()
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
            
            # Overlay con immagine
            ax.imshow(img, alpha=0.6)
            im = ax.imshow(att_map, cmap='hot', alpha=0.7)
            
            ax.set_title(att_name.replace('_', ' ').title(), fontsize=9)
            ax.axis('off')
    
    plt.suptitle('MSHA Attention Maps Visualization', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plotta curve di training
    
    Args:
        train_metrics: Metriche di training per epoca
        val_metrics: Metriche di validation per epoca
        save_path: Path per salvare la figura
        figsize: Dimensioni della figura
    
    Returns:
        fig: Figura matplotlib
    """
    # Metriche da plottare
    metrics_to_plot = ['loss', 'accuracy', 'kappa', 'f1_score']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    epochs = range(1, len(list(train_metrics.values())[0]) + 1)
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        if metric in train_metrics:
            ax.plot(epochs, train_metrics[metric], 'b-', label=f'Train {metric}', linewidth=2)
        
        if metric in val_metrics:
            ax.plot(epochs, val_metrics[metric], 'r-', label=f'Val {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()} Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Curves', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_confusion_matrix_comparison(
    y_true_list: List[np.ndarray],
    y_pred_list: List[np.ndarray],
    model_names: List[str],
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Confronta confusion matrix di diversi modelli
    """
    from sklearn.metrics import confusion_matrix
    
    num_models = len(y_true_list)
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5))
    
    if num_models == 1:
        axes = [axes]
    
    for i, (y_true, y_pred, model_name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[i]
        )
        
        axes[i].set_title(f'{model_name}', fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_uncertainty_distribution(
    uncertainties: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualizza distribuzione dell'incertezza per classe
    """
    num_classes = len(class_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Uncertainty per classe
    for i in range(num_classes):
        mask = targets == i
        if mask.sum() > 0:
            class_uncertainties = uncertainties[mask]
            correct_mask = predictions[mask] == targets[mask]
            
            # Histogram
            axes[i].hist(class_uncertainties[correct_mask], bins=20, alpha=0.7, 
                        label='Correct', color='green', density=True)
            axes[i].hist(class_uncertainties[~correct_mask], bins=20, alpha=0.7, 
                        label='Wrong', color='red', density=True)
            
            axes[i].set_title(f'{class_names[i]}')
            axes[i].set_xlabel('Uncertainty')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Overall uncertainty vs accuracy
    axes[5].scatter(uncertainties, predictions == targets, alpha=0.6)
    axes[5].set_xlabel('Uncertainty')
    axes[5].set_ylabel('Correct Prediction')
    axes[5].set_title('Uncertainty vs Correctness')
    axes[5].grid(True, alpha=0.3)
    
    plt.suptitle('Uncertainty Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
