#!/usr/bin/env python3
"""
Main script for training and evaluation of MSHA model for diabetic retinopathy classification
"""

import argparse
import yaml
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.dataset import create_data_loaders
from src.models.msha_network import create_msha_model
from src.training.trainer import create_trainer, train_model
from src.training.callbacks import create_callbacks
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logging_utils import setup_logging, log_model_info, log_training_start
from src.utils.model_utils import save_model_state
from src.utils.visualization import plot_training_curves
import pytorch_lightning as pl


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="DeepRetina-MSHA: Training and Evaluation"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/training_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'inference'],
        default='train',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint for evaluation or inference'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image for inference'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (override config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (override config)'
    )
    
    parser.add_argument(
        '--gpu',
        type=str,
        default='auto',
        help='GPU to use (auto, 0, 1, etc.)'
    )
    
    parser.add_argument(
        '--fast-dev-run',
        action='store_true',
        help='Run only a few iterations for testing'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_environment(args, config):
    """Setup environment and logging"""
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger = setup_logging(
        log_file=str(output_dir / 'training.log'),
        log_level=log_level
    )
    
    # Log configurations
    logger.info("=" * 60)
    logger.info("DEEPRETINA-MSHA EXECUTION")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"GPU: {args.gpu}")
    
    # Override configurations from command line
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
        logger.info(f"Override epochs: {args.epochs}")
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
        logger.info(f"Override batch size: {args.batch_size}")
    
    if args.fast_dev_run:
        config['training']['num_epochs'] = 2
        logger.info("Fast dev run: limited to 2 epochs")
    
    return logger


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration  
    config = load_config(args.config)
    
    # Setup environment
    logger = setup_environment(args, config)
    
    try:
        if args.mode == 'train':
            train_mode(args, config, logger)
        elif args.mode == 'evaluate':
            evaluate_mode(args, config, logger)
        elif args.mode == 'inference':
            inference_mode(args, config, logger)
            
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise


def train_mode(args, config, logger):
    """Training mode"""
    logger.info("Starting training mode...")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, data_info = create_data_loaders(
        labels_file=config['paths']['labels_file'],
        images_dir=config['paths']['images_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        train_ratio=config['data']['train_split'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Log training start
    log_training_start(logger, data_info['train_size'], data_info['val_size'], config)
    
    # Create model
    logger.info("Creating MSHA model...")
    
    # Calculate class frequencies
    class_frequencies = [data_info['train_distribution'][i] for i in range(5)]
    
    # Create trainer
    trainer_module = create_trainer(config, class_frequencies)
    log_model_info(logger, trainer_module.model, config['model'])
    
    # Setup PyTorch Lightning trainer
    callbacks = create_callbacks(config)
    
    # Logger for Lightning
    if args.verbose:
        from pytorch_lightning.loggers import TensorBoardLogger
        pl_logger = TensorBoardLogger(
            save_dir=config['paths']['logs_dir'],
            name="msha_training"
        )
    else:
        pl_logger = None
    
    # Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        accelerator='auto' if args.gpu == 'auto' else 'gpu',
        devices='auto' if args.gpu == 'auto' else [int(args.gpu)] if args.gpu.isdigit() else 1,
        precision='16-mixed',
        callbacks=callbacks,
        logger=pl_logger,
        gradient_clip_val=config['training']['gradient_clip_val'],
        fast_dev_run=args.fast_dev_run,
        deterministic=True
    )
    
    # Training
    logger.info("Starting training...")
    trainer.fit(trainer_module, train_loader, val_loader)
    
    # Save final model
    output_dir = Path(args.output_dir)
    save_model_state(
        model=trainer_module.model,
        config=config,
        metrics=trainer_module.val_metrics_tracker.compute_metrics(),
        save_path=str(output_dir / 'final_model.pth')
    )
    
    logger.info("Training completed successfully!")


def evaluate_mode(args, config, logger):
    """Evaluation mode"""
    logger.info("Starting evaluation mode...")
    
    if not args.checkpoint:
        logger.error("Checkpoint path required for evaluation mode")
        sys.exit(1)
    
    # Load model
    from src.utils.model_utils import load_model_from_checkpoint
    model, model_info = load_model_from_checkpoint(args.checkpoint)
    
    # Create data loader for validation
    _, val_loader, data_info = create_data_loaders(
        labels_file=config['paths']['labels_file'],
        images_dir=config['paths']['images_dir'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        train_ratio=config['data']['train_split'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=args.gpu if args.gpu != 'auto' else 'auto',
        class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    )
    
    # Evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate_dataset(
        dataloader=val_loader,
        use_uncertainty=True,
        save_predictions=True,
        output_dir=args.output_dir
    )
    
    # Log main results
    logger.info("Evaluation Results:")
    logger.info(f"Accuracy: {results['standard_metrics']['accuracy']:.4f}")
    logger.info(f"Quadratic Kappa: {results['standard_metrics']['quadratic_kappa']:.4f}")
    
    logger.info("Evaluation completed successfully!")


def inference_mode(args, config, logger):
    """Inference mode"""
    logger.info("Starting inference mode...")
    
    if not args.checkpoint:
        logger.error("Checkpoint path required for inference mode")
        sys.exit(1)
    
    if not args.image:
        logger.error("Image path required for inference mode")
        sys.exit(1)
    
    # Load model
    from src.utils.model_utils import load_model_from_checkpoint
    from src.data.augmentation import get_val_transforms
    
    model, model_info = load_model_from_checkpoint(args.checkpoint)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=args.gpu if args.gpu != 'auto' else 'auto',
        class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    )
    
    # Transforms
    transforms = get_val_transforms(config['data']['image_size'])
    
    # Inference
    logger.info(f"Running inference on: {args.image}")
    result = evaluator.evaluate_single_image(
        image_path=args.image,
        transforms=transforms,
        use_uncertainty=True
    )
    
    # Log results
    logger.info("Inference Results:")
    logger.info(f"Predicted Class: {result['predicted_class']}")
    logger.info(f"Predicted Grade: {result['predicted_grade']}")
    logger.info(f"Confidence: {result['confidence']:.4f}")
    
    if 'uncertainty' in result:
        logger.info(f"Total Uncertainty: {result['uncertainty']['total']:.4f}")
    
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
