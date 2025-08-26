"""
Logging utilities for DeepRetina-MSHA
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
        format_string: Custom format for logs
    
    Returns:
        logger: Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Setup root logger
    logger = logging.getLogger('DeepRetina-MSHA')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_model_info(logger: logging.Logger, model, config: Dict[str, Any]):
    """
    Logga informazioni dettagliate sul modello
    
    Args:
        logger: Logger instance
        model: Modello PyTorch
        config: Model configuration
    """
    logger.info("=" * 60)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 60)
    
    # Model configuration
    logger.info("Model Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Dimensioni modello
    if hasattr(model, 'get_model_size'):
        model_info = model.get_model_size()
        logger.info("\nModel Size Information:")
        for key, value in model_info.items():
            if isinstance(value, (int, float)):
                if 'parameters' in key.lower():
                    logger.info(f"  {key}: {value:,}")
                else:
                    logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Conteggio parametri per layer type
    logger.info("\nParameters by Layer Type:")
    layer_params = {}
    for name, param in model.named_parameters():
        layer_type = name.split('.')[0]
        if layer_type not in layer_params:
            layer_params[layer_type] = 0
        layer_params[layer_type] += param.numel()
    
    for layer_type, count in sorted(layer_params.items()):
        logger.info(f"  {layer_type}: {count:,} parameters")
    
    logger.info("=" * 60)


def log_training_start(
    logger: logging.Logger, 
    train_size: int, 
    val_size: int, 
    config: Dict[str, Any]
):
    """
    Logga informazioni all'inizio del training
    """
    logger.info("=" * 60)
    logger.info("TRAINING START")
    logger.info("=" * 60)
    
    logger.info(f"Training samples: {train_size:,}")
    logger.info(f"Validation samples: {val_size:,}")
    logger.info(f"Total samples: {train_size + val_size:,}")
    
    logger.info("\nTraining Configuration:")
    training_config = config.get('training', {})
    for key, value in training_config.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nOptimizer Configuration:")
    optimizer_config = config.get('optimizer', {})
    for key, value in optimizer_config.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


def log_epoch_results(
    logger: logging.Logger,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float]
):
    """
    Logga risultati per epoca
    """
    logger.info(f"\nEpoch {epoch} Results:")
    logger.info("-" * 40)
    
    logger.info("Training Metrics:")
    for metric, value in train_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    
    logger.info("Validation Metrics:")
    for metric, value in val_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


def log_final_results(
    logger: logging.Logger,
    final_metrics: Dict[str, float],
    best_epoch: int,
    total_training_time: float
):
    """
    Logga risultati finali del training
    """
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 60)
    
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    logger.info(f"Training time: {total_training_time/3600:.2f} hours")
    
    logger.info("\nFinal Metrics:")
    logger.info("-" * 30)
    for metric, value in final_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.6f}")
        else:
            logger.info(f"  {metric}: {value}")
    
    # Interpretazione clinica
    kappa = final_metrics.get('quadratic_kappa', 0)
    accuracy = final_metrics.get('accuracy', 0)
    
    logger.info("\nClinical Interpretation:")
    logger.info("-" * 30)
    
    if kappa >= 0.85:
        logger.info("  Quadratic Kappa: EXCELLENT (≥0.85)")
    elif kappa >= 0.75:
        logger.info("  Quadratic Kappa: GOOD (≥0.75)")
    elif kappa >= 0.60:
        logger.info("  Quadratic Kappa: MODERATE (≥0.60)")
    else:
        logger.info("  Quadratic Kappa: NEEDS IMPROVEMENT (<0.60)")
    
    if accuracy >= 0.90:
        logger.info("  Accuracy: EXCELLENT (≥90%)")
    elif accuracy >= 0.80:
        logger.info("  Accuracy: GOOD (≥80%)")
    elif accuracy >= 0.70:
        logger.info("  Accuracy: ACCEPTABLE (≥70%)")
    else:
        logger.info("  Accuracy: NEEDS IMPROVEMENT (<70%)")
    
    logger.info("=" * 60)


def save_experiment_log(
    experiment_name: str,
    config: Dict[str, Any],
    final_metrics: Dict[str, float],
    log_dir: str = "logs"
):
    """
    Salva log completo dell'esperimento
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_log = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'config': config,
        'final_metrics': final_metrics,
        'model_info': {
            'architecture': config.get('model', {}).get('name', 'Unknown'),
            'backbone': config.get('model', {}).get('backbone', 'Unknown')
        }
    }
    
    log_file = log_path / f"{experiment_name}_{timestamp}.json"
    with open(log_file, 'w') as f:
        json.dump(experiment_log, f, indent=2, default=str)
    
    return str(log_file)
