"""
Main trainer for MSHA Network using PyTorch Lightning
Handles parameter validation and robust configuration management
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import wandb

from ..models.msha_network import MSHANetwork
from .losses import CombinedLoss, create_loss_function
from .metrics import MedicalMetrics, MetricsTracker
# Import will be done locally to avoid circular imports


class MSHATrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for MSHA Network
    Includes robust parameter validation and error handling
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any],
        loss_config: Dict[str, Any],
        class_frequencies: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Configurations
        self.model_config = model_config
        self.training_config = training_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.loss_config = loss_config
        
        # Modello - validate and filter parameters
        # Filter only valid MSHANetwork parameters to avoid unexpected keyword arguments
        valid_keys = {'num_classes', 'backbone', 'attention_dim', 'dropout_rate', 'uncertainty_estimation', 'pretrained'}
        valid_model_params = {k: v for k, v in model_config.items() if k in valid_keys}
        
        # Add defaults for missing keys
        defaults = {
            'num_classes': 5,
            'backbone': 'efficientnet_b4', 
            'attention_dim': 256,
            'dropout_rate': 0.3,
            'uncertainty_estimation': True,
            'pretrained': True
        }
        for key, default_val in defaults.items():
            if key not in valid_model_params:
                valid_model_params[key] = default_val
                
        self.model = MSHANetwork(**valid_model_params)
        
        # Ensure model is on correct device (fix for device mismatch)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Loss function
        self.criterion = create_loss_function(loss_config, class_frequencies)
        
        # Metrics
        self.num_classes = model_config.get('num_classes', 5)
        self.metrics_calculator = MedicalMetrics(self.num_classes)
        
        # Trackers for train/val
        self.train_metrics_tracker = MetricsTracker(self.num_classes)
        self.val_metrics_tracker = MetricsTracker(self.num_classes)
        
        # For best model tracking
        self.best_val_kappa = -1.0
        self.best_val_accuracy = 0.0
        
        # Learning rate for logging
        self.automatic_optimization = True
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        return self.model(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self.forward(images)
        
        # Calculate loss
        loss_dict = self.criterion(outputs, labels)
        total_loss = loss_dict['total_loss']
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'train/{loss_name}', loss_value, 
                    on_step=True, on_epoch=True, prog_bar=True)
        
        # Predictions for metrics
        predictions = torch.argmax(outputs['logits'], dim=1)
        probabilities = outputs['predictions']
        
        # Update tracker
        self.train_metrics_tracker.update(
            predictions=predictions,
            targets=labels,
            probabilities=probabilities,
            loss=total_loss.item()
        )
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        outputs = self.forward(images)
        
        # Calculate loss
        loss_dict = self.criterion(outputs, labels)
        total_loss = loss_dict['total_loss']
        
        # Log losses
        for loss_name, loss_value in loss_dict.items():
            self.log(f'val/{loss_name}', loss_value,
                    on_step=False, on_epoch=True, prog_bar=True)
        
        # Predictions for metrics
        predictions = torch.argmax(outputs['logits'], dim=1)
        probabilities = outputs['predictions']
        
        # Update tracker
        self.val_metrics_tracker.update(
            predictions=predictions,
            targets=labels,
            probabilities=probabilities,
            loss=total_loss.item()
        )
        
        return total_loss
    
    def on_train_epoch_start(self):
        """Reset trackers at the beginning of epoch"""
        self.train_metrics_tracker.reset()
    
    def on_validation_epoch_start(self):
        """Reset trackers at the beginning of validation"""
        self.val_metrics_tracker.reset()
    
    def on_train_epoch_end(self):
        """Calculate and log training metrics"""
        metrics = self.train_metrics_tracker.compute_metrics()
        
        for metric_name, metric_value in metrics.items():
            self.log(f'train/{metric_name}', metric_value, 
                    on_epoch=True, prog_bar=False)
        
        # Log main metrics
        if 'quadratic_kappa' in metrics:
            self.log('train/kappa', metrics['quadratic_kappa'], 
                    on_epoch=True, prog_bar=True)
        if 'accuracy' in metrics:
            self.log('train/acc', metrics['accuracy'], 
                    on_epoch=True, prog_bar=True)
    
    def on_validation_epoch_end(self):
        """Calculate and log validation metrics"""
        metrics = self.val_metrics_tracker.compute_metrics()
        
        for metric_name, metric_value in metrics.items():
            self.log(f'val/{metric_name}', metric_value,
                    on_epoch=True, prog_bar=False)
        
        # Log main metrics
        current_kappa = metrics.get('quadratic_kappa', -1.0)
        current_accuracy = metrics.get('accuracy', 0.0)
        
        self.log('val/kappa', current_kappa, on_epoch=True, prog_bar=True)
        self.log('val/acc', current_accuracy, on_epoch=True, prog_bar=True)
        
        # Update best metrics
        if current_kappa > self.best_val_kappa:
            self.best_val_kappa = current_kappa
            self.log('val/best_kappa', self.best_val_kappa, on_epoch=True)
        
        if current_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_accuracy
            self.log('val/best_accuracy', self.best_val_accuracy, on_epoch=True)
        
        # Log confusion matrix every 10 epochs
        if self.current_epoch % 10 == 0:
            self._log_confusion_matrix()
    
    def _log_confusion_matrix(self):
        """Log confusion matrix to W&B"""
        if self.logger and hasattr(self.logger, 'experiment'):
            cm = self.val_metrics_tracker.get_confusion_matrix()
            
            # Create plot
            fig = self.metrics_calculator.create_confusion_matrix_plot(
                y_true=np.array(self.val_metrics_tracker.targets),
                y_pred=np.array(self.val_metrics_tracker.predictions),
                normalize=True
            )
            
            # Log to W&B if available
            if hasattr(self.logger.experiment, 'log'):
                self.logger.experiment.log({
                    f"confusion_matrix_epoch_{self.current_epoch}": wandb.Image(fig)
                })
            
            # Close plot to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # Optimizer
        optimizer_name = self.optimizer_config.get('name', 'AdamW')
        
        # Ensure learning rate is float
        learning_rate = float(self.training_config['learning_rate'])
        weight_decay = float(self.training_config.get('weight_decay', 1e-5))
        
        if optimizer_name == 'AdamW':
            optimizer = AdamW(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=self.optimizer_config.get('betas', [0.9, 0.999]),
                eps=float(self.optimizer_config.get('eps', 1e-8))
            )
        else:
            raise ValueError(f"Optimizer {optimizer_name} not supported")
        
        # Scheduler
        scheduler_name = self.scheduler_config.get('name', 'CosineAnnealingWarmRestarts')
        
        if scheduler_name == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.scheduler_config.get('T_0', 10)),
                T_mult=int(self.scheduler_config.get('T_mult', 2)),
                eta_min=float(self.scheduler_config.get('eta_min', 1e-6))
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        elif scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',  # Monitor kappa (higher is better)
                factor=0.5,
                patience=self.training_config.get('reduce_lr_patience', 8),
                verbose=True,
                min_lr=1e-7
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/kappa',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        else:
            return optimizer
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step for inference"""
        images = batch['image']
        
        # Forward pass
        outputs = self.forward(images)
        
        # If model supports uncertainty
        if self.model_config.get('uncertainty_estimation', False):
            # Monte Carlo predictions
            mc_results = self.model.monte_carlo_predict(images, num_samples=50)
            
            return {
                'predictions': outputs['predictions'],
                'logits': outputs['logits'],
                'mc_predictions': mc_results['mean_prediction'],
                'uncertainty': mc_results['total_uncertainty'],
                'confidence': mc_results['confidence'],
                'image_names': batch.get('image_name', [])
            }
        else:
            return {
                'predictions': outputs['predictions'],
                'logits': outputs['logits'],
                'image_names': batch.get('image_name', [])
            }
    
    def get_attention_maps(self, batch):
        """Extract attention maps for visualization"""
        images = batch['image']
        
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(images)
        
        return attention_weights
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone for fine-tuning"""
        self.model.freeze_backbone(freeze)
        
        if freeze:
            print("Backbone frozen for fine-tuning")
        else:
            print("Backbone unfrozen for complete training")


class MSHACallback(pl.Callback):
    """
    Custom callback for MSHA training
    """
    
    def __init__(self, save_attention_maps: bool = False):
        super().__init__()
        self.save_attention_maps = save_attention_maps
    
    def on_validation_end(self, trainer, pl_module):
        """Callback at end of validation"""
        # Log model size
        if trainer.current_epoch == 0:
            model_info = pl_module.model.get_model_size()
            trainer.logger.log_metrics({
                'model/total_parameters': model_info['total_parameters'],
                'model/trainable_parameters': model_info['trainable_parameters'],
                'model/size_mb': model_info['model_size_mb']
            }, step=trainer.global_step)
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Callback at start of each epoch"""
        # Log learning rate
        current_lr = trainer.optimizers[0].param_groups[0]['lr']
        trainer.logger.log_metrics({'lr': current_lr}, step=trainer.global_step)


def create_trainer(
    config: Dict[str, Any],
    class_frequencies: Optional[List[int]] = None,
    logger=None,
    callbacks: Optional[List[pl.Callback]] = None
) -> MSHATrainer:
    """
    Factory function to create the trainer
    """
    trainer = MSHATrainer(
        model_config=config['model'],
        training_config=config['training'],
        optimizer_config=config['optimizer'],
        scheduler_config=config['scheduler'],
        loss_config=config['loss'],
        class_frequencies=class_frequencies
    )
    
    return trainer


# Utility functions for training
def train_model(
    trainer_module: MSHATrainer,
    train_loader,
    val_loader,
    max_epochs: int = 100,
    accelerator: str = 'auto',
    devices: str = 'auto',
    precision: str = '16-mixed',
    callbacks: Optional[List[pl.Callback]] = None,
    logger=None
) -> pl.Trainer:
    """
    Complete training function
    """
    # Default callbacks
    if callbacks is None:
        callbacks = []
    
    callbacks.append(MSHACallback())
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        deterministic=True,
        log_every_n_steps=50
    )
    
    # Training
    trainer.fit(trainer_module, train_loader, val_loader)
    
    return trainer


def evaluate_model(
    trainer_module: MSHATrainer,
    test_loader,
    ckpt_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete model evaluation
    """
    trainer = pl.Trainer(
        logger=False,
        devices=1,
        accelerator='auto'
    )
    
    # Test
    test_results = trainer.test(trainer_module, test_loader, ckpt_path=ckpt_path)
    
    return test_results[0] if test_results else {}
