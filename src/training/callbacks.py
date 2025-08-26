import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateMonitor,
    TQDMProgressBar
)
import torch
import numpy as np
from typing import List, Optional, Dict, Any
# Optional wandb import - not required
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsLoggingCallback(pl.Callback):
    """
    Callback for advanced metrics logging
    """
    
    def __init__(self, log_frequency: int = 10):
        super().__init__()
        self.log_frequency = log_frequency
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log only essential metrics during training"""
        # Don't log confusion matrix during training - only acc and loss
        # Confusion matrix will be shown only in final evaluation
        pass
    
    def generate_confusion_matrix(self, trainer, pl_module):
        """Generate confusion matrix for final evaluation"""
        return self._log_confusion_matrix(trainer, pl_module)
    
    def _log_confusion_matrix(self, trainer, pl_module):
        """Log confusion matrix - called only from final evaluation"""
        if hasattr(pl_module, 'val_metrics_tracker'):
            cm = pl_module.val_metrics_tracker.get_confusion_matrix()
            
            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                xticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                yticklabels=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                ax=ax
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'Confusion Matrix - Epoch {trainer.current_epoch}')
            
            # Log confusion matrix to appropriate logger
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                try:
                    # Check if using Weights & Biases
                    if hasattr(trainer.logger.experiment, 'log') and 'wandb' in str(type(trainer.logger.experiment)):
                        if WANDB_AVAILABLE:
                            trainer.logger.experiment.log({
                                "confusion_matrix": wandb.Image(fig),
                                "epoch": trainer.current_epoch
                            })
                    # TensorBoard logging
                    elif hasattr(trainer.logger.experiment, 'add_figure'):
                        trainer.logger.experiment.add_figure(
                            'confusion_matrix',
                            fig,
                            global_step=trainer.current_epoch
                        )
                except Exception as e:
                    print(f"Warning: Could not log confusion matrix: {e}")
                    pass
            
            plt.close(fig)
    
    def _log_per_class_metrics(self, trainer, pl_module):
        """Log per-class metrics"""
        if hasattr(pl_module, 'val_metrics_tracker'):
            y_true = np.array(pl_module.val_metrics_tracker.targets)
            y_pred = np.array(pl_module.val_metrics_tracker.predictions)
            
            # Calculate per-class metrics
            per_class = pl_module.metrics_calculator.per_class_metrics(y_true, y_pred)
            
            # Log to trainer
            for class_name, metrics in per_class.items():
                for metric_name, metric_value in metrics.items():
                    trainer.logger.log_metrics({
                        f"val/{class_name}_{metric_name}": metric_value
                    }, step=trainer.global_step)


class UncertaintyLoggingCallback(pl.Callback):
    """
    Callback per logging dell'incertezza
    """
    
    def __init__(self, log_frequency: int = 20):
        super().__init__()
        self.log_frequency = log_frequency
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log uncertainty statistics"""
        if (trainer.current_epoch % self.log_frequency == 0 and 
            pl_module.model_config.get('uncertainty_estimation', False)):
            
            self._log_uncertainty_stats(trainer, pl_module)
    
    def _log_uncertainty_stats(self, trainer, pl_module):
        """Calcola e logga statistiche di incertezza"""
        try:
            # Safe access to validation dataloader
            val_dataloaders = trainer.val_dataloaders
            if isinstance(val_dataloaders, list):
                val_loader = val_dataloaders[0]
            else:
                val_loader = val_dataloaders
            
            batch = next(iter(val_loader))
            
            # Sposta su device
            images = batch['image'].to(pl_module.device)
            labels = batch['label'].to(pl_module.device)
            
            with torch.no_grad():
                # Monte Carlo predictions
                mc_results = pl_module.model.monte_carlo_predict(images, num_samples=50)
                
                # Statistiche incertezza
                uncertainty = mc_results['total_uncertainty']
                confidence = mc_results['confidence']
                
                # Per classe
                uncertainty_by_class = {}
                confidence_by_class = {}
                
                for class_idx in range(5):
                    mask = labels == class_idx
                    if mask.sum() > 0:
                        uncertainty_by_class[f'uncertainty_class_{class_idx}'] = uncertainty[mask].mean().item()
                        confidence_by_class[f'confidence_class_{class_idx}'] = confidence[mask].mean().item()
                
                # Log metriche
                metrics = {
                    'uncertainty/mean': uncertainty.mean().item(),
                    'uncertainty/std': uncertainty.std().item(),
                    'confidence/mean': confidence.mean().item(),
                    'confidence/std': confidence.std().item(),
                    **uncertainty_by_class,
                    **confidence_by_class
                }
                
                trainer.logger.log_metrics(metrics, step=trainer.global_step)
                
        except Exception as e:
            print(f"Warning: Could not log uncertainty stats: {e}")


class AttentionVisualizationCallback(pl.Callback):
    """
    Callback for visualizing attention maps - FIXED VERSION
    """
    
    def __init__(self, log_frequency: int = 50, num_samples: int = 4):
        super().__init__()
        self.log_frequency = log_frequency
        self.num_samples = num_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Visualize attention maps only occasionally"""
        # Reduced frequency to avoid cluttering log during training
        if trainer.current_epoch % (self.log_frequency * 2) == 0:
            self._visualize_attention(trainer, pl_module)
    
    def _visualize_attention(self, trainer, pl_module):
        """Create attention map visualizations - FIXED VERSION"""
        try:
            # FIXED: Safe access to validation dataloader
            val_dataloaders = trainer.val_dataloaders
            if isinstance(val_dataloaders, list):
                val_loader = val_dataloaders[0]
            else:
                val_loader = val_dataloaders
            
            # Check if val_loader is None or empty
            if val_loader is None:
                print("Warning: No validation dataloader available for attention visualization")
                return
            
            batch = next(iter(val_loader))
            
            # Take only a few samples
            images = batch['image'][:self.num_samples].to(pl_module.device)
            labels = batch['label'][:self.num_samples]
            
            with torch.no_grad():
                attention_weights = pl_module.model.get_attention_weights(images)
            
            # Create visualizations
            fig, axes = plt.subplots(len(attention_weights), self.num_samples, 
                                    figsize=(4*self.num_samples, 4*len(attention_weights)))
            
            # Handle single attention type case
            if len(attention_weights) == 1:
                axes = axes.reshape(1, -1)
            
            for i, (attention_name, attention_maps) in enumerate(attention_weights.items()):
                for j in range(self.num_samples):
                    ax = axes[i, j] if len(attention_weights) > 1 else axes[j]
                    
                    # Normalize attention map
                    att_map = attention_maps[j, 0].cpu().numpy()
                    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
                    
                    # Plot
                    im = ax.imshow(att_map, cmap='hot', alpha=0.8)
                    ax.set_title(f'{attention_name} - Label {labels[j].item()}')
                    ax.axis('off')
                    
                    # Colorbar
                    plt.colorbar(im, ax=ax, shrink=0.8)
            
            plt.tight_layout()
            
            # Log attention maps to appropriate logger
            if trainer.logger and hasattr(trainer.logger, 'experiment'):
                try:
                    # Check if using Weights & Biases
                    if hasattr(trainer.logger.experiment, 'log') and 'wandb' in str(type(trainer.logger.experiment)):
                        if WANDB_AVAILABLE:
                            trainer.logger.experiment.log({
                                f"attention_maps_epoch_{trainer.current_epoch}": wandb.Image(fig)
                            })
                    # TensorBoard logging
                    elif hasattr(trainer.logger.experiment, 'add_figure'):
                        trainer.logger.experiment.add_figure(
                            f'attention_maps_epoch_{trainer.current_epoch}',
                            fig,
                            global_step=trainer.current_epoch
                        )
                except Exception as e:
                    print(f"Warning: Could not log attention maps: {e}")
            
            plt.close(fig)
            
        except Exception as e:
            print(f"Warning: Could not visualize attention maps: {e}")
            return


class GradualUnfreezingCallback(pl.Callback):
    """
    Callback per unfreezing graduale del backbone
    """
    
    def __init__(self, unfreeze_epoch: int = 10):
        super().__init__()
        self.unfreeze_epoch = unfreeze_epoch
        self.unfrozen = False
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Unfreeze backbone dopo N epoche"""
        if trainer.current_epoch == self.unfreeze_epoch and not self.unfrozen:
            pl_module.model.freeze_backbone(False)
            self.unfrozen = True
            
            # Riduci learning rate per fine-tuning
            for param_group in trainer.optimizers[0].param_groups:
                param_group['lr'] *= 0.1
            
            print(f"Epoch {trainer.current_epoch}: Backbone unfrozen, LR ridotto")


class ModelStatisticsCallback(pl.Callback):
    """
    Callback per logging statistiche del modello
    """
    
    def __init__(self, log_frequency: int = 100):
        super().__init__()
        self.log_frequency = log_frequency
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Log gradient norms e weight statistics"""
        if trainer.global_step % self.log_frequency == 0:
            self._log_gradient_stats(trainer, pl_module)
            self._log_weight_stats(trainer, pl_module)
    
    def _log_gradient_stats(self, trainer, pl_module):
        """Log gradient norms"""
        try:
            total_norm = 0
            param_count = 0
            
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                
                trainer.logger.log_metrics({
                    'gradients/total_norm': total_norm,
                    'gradients/num_params_with_grad': param_count
                }, step=trainer.global_step)
        except Exception as e:
            print(f"Warning: Could not log gradient stats: {e}")
    
    def _log_weight_stats(self, trainer, pl_module):
        """Log weight statistics"""
        try:
            for name, param in pl_module.named_parameters():
                if 'weight' in name and param.numel() > 0:
                    trainer.logger.log_metrics({
                        f'weights/{name}_mean': param.data.mean().item(),
                        f'weights/{name}_std': param.data.std().item(),
                        f'weights/{name}_max': param.data.max().item(),
                        f'weights/{name}_min': param.data.min().item()
                    }, step=trainer.global_step)
        except Exception as e:
            print(f"Warning: Could not log weight stats: {e}")


def create_callbacks(
    config: Dict[str, Any],
    monitor_metric: str = 'val/kappa'
) -> List[pl.Callback]:
    """
    Create list of callbacks based on configuration
    """
    callbacks = []
    
    # Early Stopping
    early_stopping = EarlyStopping(
        monitor=monitor_metric,
        patience=config['training'].get('early_stopping_patience', 15),
        mode='max',  # Higher Kappa is better
        verbose=True,
        strict=False
    )
    callbacks.append(early_stopping)
    
    # Model Checkpoint
    checkpoint = ModelCheckpoint(
        dirpath=config['paths'].get('checkpoints_dir', 'checkpoints'),
        filename='msha-{epoch:02d}-{val/kappa:.4f}',
        monitor=monitor_metric,
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint)
    
    # Learning Rate Monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Progress Bar
    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks.append(progress_bar)
    
    # Callbacks personalizzati
    callbacks.append(MetricsLoggingCallback(log_frequency=10))
    callbacks.append(UncertaintyLoggingCallback(log_frequency=20))
    callbacks.append(AttentionVisualizationCallback(log_frequency=50))
    callbacks.append(ModelStatisticsCallback(log_frequency=100))
    
    # Gradual Unfreezing se abilitato
    if config['training'].get('gradual_unfreezing', False):
        unfreeze_epoch = config['training'].get('unfreeze_epoch', 10)
        callbacks.append(GradualUnfreezingCallback(unfreeze_epoch))
    
    return callbacks