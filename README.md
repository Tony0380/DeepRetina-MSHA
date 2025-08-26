# DeepRetina-MSHA: Multi-Scale Hierarchical Attention Networks for Diabetic Retinopathy

A deep learning framework for automated diabetic retinopathy classification using the EyePACS dataset with a novel Multi-Scale Hierarchical Attention (MSHA) architecture.

## ⚠️ Research Disclaimer

This software is for **research purposes only** and is **not intended for clinical use**. 
It has not been approved by any regulatory authority for medical diagnosis. 
Always consult qualified healthcare professionals for medical decisions.

- 🚫 Not FDA approved
- 🚫 Not CE marked  
- 🚫 Not intended for patient care
- ✅ Research and educational use only

## 🏗️ MSHA Architecture

### Core Components:

1. **Backbone**: Pre-trained EfficientNet-B4 for robust feature extraction
2. **Multi-Scale Feature Pyramid**: Feature extraction at 4 scales (1x, 2x, 4x, 8x)
3. **Hierarchical Attention**:
   - **Global Attention**: Attention over entire image
   - **Regional Attention**: Attention over 4 quadrants  
   - **Local Attention**: Attention over local patches
4. **Feature Fusion**: Learnable weighted concatenation
5. **Uncertainty Quantification**: Bayesian dropout layers for uncertainty estimation
6. **Classification Head**: 5-class output (DR grades 0-4)

## 📊 Dataset

**EyePACS Dataset**: 88,700 retinal fundus images
- **Grade 0** (No DR): 73.67% (65,342 samples)
- **Grade 1** (Mild): 7.00% (6,205 samples)  
- **Grade 2** (Moderate): 14.83% (13,152 samples)
- **Grade 3** (Severe): 2.35% (2,087 samples)
- **Grade 4** (Proliferative): 2.16% (1,914 samples)

### Dataset Setup

The dataset is not included in this repository. To use the full functionality:

1. **Download the EyePACS dataset**
2. **Extract images to `dataset/Images/`**
3. **Place labels file as `dataset/all_labels.csv`**

Expected structure:
```
dataset/
├── Images/
│   ├── 1_left.png
│   ├── 1_right.png
│   └── ...
└── all_labels.csv
```

The notebook includes demo functionality that works without the dataset for code exploration.

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)
```bash
# Linux/macOS
bash install.sh

# Windows
install.bat
```

### Option 2: Manual Installation
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running the Framework
```bash
# 1. Jupyter notebook (recommended)
jupyter lab notebooks/DeepRetina_MSHA_Complete.ipynb

# 2. Command-line training
python main.py --mode train                    # Full training (100 epochs)
python main.py --mode train --epochs 2 --fast-dev-run  # Quick test
python main.py --mode train --batch-size 16 --epochs 50  # Custom config

# 3. Evaluation and inference
python main.py --mode evaluate --checkpoint path/to/model.ckpt
python main.py --mode inference --checkpoint path/to/model.ckpt --image path/to/image.png
```

## 📁 Project Structure

```
DeepRetina-MSHA/
├── src/
│   ├── models/              # MSHA architecture and components
│   ├── data/                # DataLoaders and preprocessing
│   ├── training/            # Training loop and utilities
│   ├── utils/               # Utility functions
│   └── evaluation/          # Metrics and evaluation
├── config/                  # Configuration files
├── notebooks/               # Jupyter notebooks
├── dataset/                 # EyePACS dataset
├── checkpoints/             # Saved models
├── logs/                    # Training logs
└── requirements.txt         # Python dependencies
```

## 🔧 Hardware Configuration

Optimized for modern GPUs with 8GB+ VRAM:
- **Batch size**: 12 (configurable, increase for larger GPUs)
- **Image size**: 256x256 (configurable in training config)
- **Mixed precision**: FP16 training enabled
- **Gradient accumulation**: Automatic based on batch size
- **Memory optimization**: Efficient attention mechanisms

## 📈 Key Features

- **Medical-Specific Augmentation**: Specialized transformations for retinal images
- **Focal Loss**: Automatic class imbalance handling with configurable weights
- **Uncertainty Quantification**: Bayesian dropout for predictive uncertainty
- **Multi-Scale Learning**: Hierarchical attention at multiple scales
- **Advanced Optimization**: AdamW optimizer with CosineAnnealingWarmRestarts
- **Clinical Metrics**: Quadratic Kappa, AUC-ROC, Sensitivity, Specificity
- **Robust Training**: Early stopping, learning rate scheduling, gradient clipping

## 📚 Usage

### Jupyter Notebook (Recommended)
Refer to the notebook `notebooks/DeepRetina_MSHA_Complete.ipynb` for a complete tutorial covering:
- Exploratory data analysis
- Preprocessing and augmentation
- MSHA model training
- Evaluation and metrics
- Results visualization
- Inference on new images

### Command Line Interface
```bash
# Training modes
python main.py --mode train                    # Full training
python main.py --mode evaluate --checkpoint path/to/model.ckpt
python main.py --mode inference --checkpoint path/to/model.ckpt --image path/to/image.png

# Configuration options
python main.py --config custom_config.yaml    # Custom configuration
python main.py --gpu 0                        # Specific GPU
python main.py --verbose                      # Detailed logging
```

### Model Configuration
Key parameters in `config/training_config.yaml`:
- **Model**: MSHANetwork with EfficientNet-B4 backbone
- **Optimizer**: AdamW with β=(0.9, 0.999), lr=1e-4
- **Scheduler**: CosineAnnealingWarmRestarts (T₀=10, T_mult=2)
- **Loss**: Focal Loss + Uncertainty Loss + Ordinal Loss
- **Data**: 256×256 images, 80/20 train/val split

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Author

**Antonio Colamartino** - Research in Deep Learning for Medical Imaging