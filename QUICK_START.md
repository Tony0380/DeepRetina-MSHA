# 🚀 Quick Start Guide - DeepRetina-MSHA

## ⚠️ Research Disclaimer
**This software is for research purposes only and NOT for clinical use. Always consult qualified healthcare professionals.**

## 📋 Prerequisites

1. **GPU**: Modern GPU with 8GB+ VRAM (RTX 3060, RTX 4060, or higher)
2. **Python**: 3.8+
3. **Dataset** (Optional for exploration): EyePACS dataset
   
   **Note**: The notebook works without the dataset for code exploration and includes demo functionality. To use the full dataset:
   
   - Download EyePACS dataset (88,700+ retinal images)
   - Extract to `dataset/Images/` directory
   - Place labels as `dataset/all_labels.csv`
   
   Expected structure:
   ```
   dataset/
   ├── Images/        # 88,700+ PNG images
   └── all_labels.csv # Labels with columns: image, level
   ```

## ⚡ Quick Setup

### 1. Installation
```bash
# Clone repository
git clone <repository-url>
cd DeepRetina-MSHA

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
The `config/training_config.yaml` file is pre-optimized for 8GB VRAM GPUs:
- **Batch size**: 12 (conservative setting for stability)
- **Mixed precision**: FP16 training enabled
- **Memory-efficient settings**: Optimized for 8GB VRAM

### 3. Quick Training (Test)
```bash
# Fast training for testing (2 epochs)
python main.py --mode train --fast-dev-run --epochs 2

# Full training
python main.py --mode train --epochs 100
```

### 4. Interactive Notebook
```bash
# Launch Jupyter
jupyter notebook

# Open: notebooks/DeepRetina_MSHA_Complete.ipynb
```

## 🎯 Main Commands

### Training
```bash
# Standard training
python main.py --mode train

# Training with custom configuration
python main.py --mode train --epochs 50 --batch-size 16

# Training with verbose logging
python main.py --mode train --verbose
```

### Evaluation
```bash
# Evaluate trained model
python main.py --mode evaluate --checkpoint checkpoints/best_model.ckpt

# Evaluation with custom output directory
python main.py --mode evaluate --checkpoint path/to/model.ckpt --output-dir results/
```

### Single Image Inference
```bash
# Predict single image
python main.py --mode inference --checkpoint checkpoints/best_model.ckpt --image path/to/image.png
```

## 📊 Expected Output

### During Training
- **Metrics**: Accuracy, Quadratic Kappa, AUC-ROC
- **Checkpoints**: Automatically saved in `checkpoints/`
- **Logs**: Training logs in `logs/`

### Output Files
```
exported_models/
├── msha_model_state.pth     # PyTorch model
├── msha_best_checkpoint.ckpt # Lightning checkpoint
├── model_report.txt         # Detailed report
└── final_metrics.json       # Final metrics
```

## 🔧 Troubleshooting

### Insufficient GPU Memory
```yaml
# In config/training_config.yaml
training:
  batch_size: 16  # Reduce from 24
  
data:
  num_workers: 2  # Reduce from 4
```

### Dataset Not Found
The dataset is not included in the repository. To set it up:

1. **Download EyePACS dataset** from Kaggle or official source
2. **Extract images** to `dataset/Images/` directory
3. **Place labels file** as `dataset/all_labels.csv`
4. **Verify structure**:
   ```
   dataset/
   ├── Images/
   │   ├── 1_left.png
   │   ├── 1_right.png
   │   └── ...
   └── all_labels.csv
   ```
5. **Re-run notebook cells** after dataset setup

**Note**: The notebook includes demo functionality that works without the dataset for code exploration.

### CUDA Errors
```bash
# Force CPU
python main.py --mode train --gpu cpu

# Specify GPU
python main.py --mode train --gpu 0
```

## 📱 Monitoring

### TensorBoard
```bash
tensorboard --logdir logs/
```

### Weights & Biases (Optional)
```python
# In notebook, change:
use_wandb = True  # to enable W&B logging
```

## 🎯 Typical Workflow

1. **Setup**: `pip install -r requirements.txt`
2. **Quick Test**: `python main.py --fast-dev-run`
3. **Training**: `python main.py --mode train`
4. **Evaluation**: `python main.py --mode evaluate --checkpoint checkpoints/best_model.ckpt`
5. **Notebook**: Detailed analysis in `notebooks/DeepRetina_MSHA_Complete.ipynb`

## 📚 Main Files

- **`main.py`**: Main script
- **`notebooks/DeepRetina_MSHA_Complete.ipynb`**: Complete tutorial
- **`config/training_config.yaml`**: Training configuration
- **`src/models/msha_network.py`**: MSHA architecture
- **`requirements.txt`**: Python dependencies

## 🆘 Support

1. **Logs**: Check `logs/training.log`
2. **Issues**: Verify GPU has >8GB VRAM
3. **Dataset**: 88,700 EyePACS samples required
4. **Memory**: Use `--batch-size 16` if memory issues

---

**Estimated first training time**: 2-4 hours (RTX 2060 Super)  
**Final model size**: ~45MB  
**Required VRAM**: ~6GB during training
