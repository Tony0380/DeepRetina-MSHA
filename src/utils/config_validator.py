"""
Configuration validation utilities
Ensures robust parameter handling across the project
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and filters model configuration parameters
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Filtered configuration with only valid MSHANetwork parameters
    """
    valid_params = {
        'num_classes', 'backbone', 'attention_dim', 
        'dropout_rate', 'uncertainty_estimation', 'pretrained'
    }
    
    # Default values
    defaults = {
        'num_classes': 5,
        'backbone': 'efficientnet_b4',
        'attention_dim': 256,
        'dropout_rate': 0.3,
        'uncertainty_estimation': True,
        'pretrained': True
    }
    
    # Filter and validate
    validated_config = {}
    for key, default_value in defaults.items():
        if key in config:
            validated_config[key] = config[key]
        else:
            validated_config[key] = default_value
            logger.warning(f"Missing config key '{key}', using default: {default_value}")
    
    # Remove invalid keys
    invalid_keys = set(config.keys()) - valid_params
    if invalid_keys:
        logger.warning(f"Ignoring invalid model config keys: {invalid_keys}")
    
    return validated_config


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates training configuration parameters
    """
    defaults = {
        'batch_size': 24,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'gradient_clip_val': 1.0,
        'early_stopping_patience': 15,
        'reduce_lr_patience': 8
    }
    
    validated_config = {}
    for key, default_value in defaults.items():
        validated_config[key] = config.get(key, default_value)
    
    return validated_config


def validate_optimizer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates optimizer configuration parameters
    """
    defaults = {
        'name': 'AdamW',
        'betas': [0.9, 0.999],
        'eps': 1e-8
    }
    
    validated_config = {}
    for key, default_value in defaults.items():
        validated_config[key] = config.get(key, default_value)
    
    return validated_config


def validate_scheduler_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates scheduler configuration parameters
    """
    defaults = {
        'name': 'CosineAnnealingWarmRestarts',
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 1e-6
    }
    
    validated_config = {}
    for key, default_value in defaults.items():
        validated_config[key] = config.get(key, default_value)
    
    return validated_config


def validate_loss_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates loss configuration parameters
    """
    defaults = {
        'focal_alpha': [1.0, 10.5, 5.9, 37.2, 40.6],
        'focal_gamma': 2.0,
        'uncertainty_weight': 0.1,
        'label_smoothing': 0.1
    }
    
    validated_config = {}
    for key, default_value in defaults.items():
        validated_config[key] = config.get(key, default_value)
    
    return validated_config


def create_robust_trainer_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a robust trainer configuration with all validations applied
    """
    return {
        'model': validate_model_config(config.get('model', {})),
        'training': validate_training_config(config.get('training', {})),
        'optimizer': validate_optimizer_config(config.get('optimizer', {})),
        'scheduler': validate_scheduler_config(config.get('scheduler', {})),
        'loss': validate_loss_config(config.get('loss', {}))
    }
