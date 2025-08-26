"""
Model management utilities
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from ..models.msha_network import MSHANetwork
from ..training.trainer import MSHATrainer


def load_model_from_checkpoint(
    checkpoint_path: str,
    map_location: Optional[str] = None
) -> Tuple[MSHANetwork, Dict[str, Any]]:
    """
    Load model from PyTorch checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint
        map_location: Device mapping for loading
    
    Returns:
        model: Loaded model
        info: Additional information from checkpoint
    """
    logger = logging.getLogger('DeepRetina-MSHA')
    
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Extract model configuration
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = MSHANetwork(**model_config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        # Lightning checkpoint
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise KeyError("Checkpoint does not contain model state_dict")
    
    # Additional information
    info = {
        'model_config': model_config,
        'training_config': checkpoint.get('training_config', {}),
        'final_metrics': checkpoint.get('final_metrics', {}),
        'class_names': checkpoint.get('class_names', [])
    }
    
    logger.info("Model loaded successfully")
    return model, info


def load_lightning_checkpoint(
    checkpoint_path: str,
    map_location: Optional[str] = None
) -> MSHATrainer:
    """
    Load PyTorch Lightning trainer from checkpoint
    
    Args:
        checkpoint_path: Path to Lightning checkpoint
        map_location: Device mapping
    
    Returns:
        trainer: Loaded Lightning trainer
    """
    logger = logging.getLogger('DeepRetina-MSHA')
    
    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading Lightning checkpoint: {checkpoint_path}")
    
    trainer = MSHATrainer.load_from_checkpoint(
        checkpoint_path,
        map_location=map_location
    )
    
    logger.info("Lightning trainer loaded successfully")
    return trainer


def save_model_state(
    model: MSHANetwork,
    config: Dict[str, Any],
    metrics: Dict[str, float],
    save_path: str,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save model state with complete information
    
    Args:
        model: Model to save
        config: Complete configuration
        metrics: Final metrics
        save_path: Destination path
        additional_info: Additional information
    """
    logger = logging.getLogger('DeepRetina-MSHA')
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # State to save
    state = {
        'model_state_dict': model.state_dict(),
        'model_config': config.get('model', {}),
        'training_config': config.get('training', {}),
        'optimizer_config': config.get('optimizer', {}),
        'scheduler_config': config.get('scheduler', {}),
        'loss_config': config.get('loss', {}),
        'final_metrics': metrics,
        'class_names': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
        'model_info': model.get_model_size()
    }
    
    # Add additional information
    if additional_info:
        state.update(additional_info)
    
    # Save
    torch.save(state, save_path)
    
    logger.info(f"Model saved to: {save_path}")
    logger.info(f"File size: {save_path.stat().st_size / (1024**2):.1f} MB")


def compare_model_sizes(*models) -> Dict[str, Dict[str, Any]]:
    """
    Compare sizes of multiple models
    
    Args:
        *models: Models to compare
    
    Returns:
        comparison: Dictionary with comparison
    """
    comparison = {}
    
    for i, model in enumerate(models):
        model_name = f"Model_{i+1}"
        if hasattr(model, 'get_model_size'):
            comparison[model_name] = model.get_model_size()
        else:
            # Manual calculation
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
            
            comparison[model_name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb
            }
    
    return comparison


def export_model_for_inference(
    model: MSHANetwork,
    config: Dict[str, Any],
    export_path: str,
    example_input: Optional[torch.Tensor] = None
):
    """
    Export optimized model for inference
    
    Args:
        model: Model to export
        config: Configuration
        export_path: Export path
        example_input: Example input for tracing
    """
    logger = logging.getLogger('DeepRetina-MSHA')
    
    model.eval()
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Example input if not provided
    if example_input is None:
        image_size = config.get('data', {}).get('image_size', 512)
        example_input = torch.randn(1, 3, image_size, image_size)
    
    try:
        # TorchScript tracing
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
        
        # Save traced model
        traced_path = export_path.with_suffix('.traced.pt')
        traced_model.save(str(traced_path))
        
        logger.info(f"Traced model saved to: {traced_path}")
        
        # Test traced model
        with torch.no_grad():
            original_output = model(example_input)
            traced_output = traced_model(example_input)
            
            # Verify outputs are similar
            if isinstance(original_output, dict) and isinstance(traced_output, dict):
                logits_diff = torch.abs(original_output['logits'] - traced_output['logits']).max()
                logger.info(f"Maximum logits difference: {logits_diff:.6f}")
            else:
                diff = torch.abs(original_output - traced_output).max()
                logger.info(f"Maximum output difference: {diff:.6f}")
        
        return str(traced_path)
        
    except Exception as e:
        logger.error(f"Error during tracing: {e}")
        logger.info("Saving only state_dict...")
        
        # Fallback: save only state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': config.get('model', {}),
            'class_names': ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        }, export_path)
        
        return str(export_path)


def optimize_model_for_inference(model: MSHANetwork) -> MSHANetwork:
    """
    Optimize model for inference
    
    Args:
        model: Model to optimize
    
    Returns:
        optimized_model: Optimized model
    """
    # Set to eval mode
    model.eval()
    
    # Disable gradient computation
    for param in model.parameters():
        param.requires_grad = False
    
    # Fuse conv-bn layers if possible
    try:
        model = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
    except:
        pass  # If not possible, continue
    
    return model
