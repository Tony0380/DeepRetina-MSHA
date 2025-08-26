#!/bin/bash
# Installation script for DeepRetina-MSHA

set -e

echo "🚀 DeepRetina-MSHA Installation Script"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in a virtual environment. Consider creating one:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Virtual environment: $VIRTUAL_ENV"
fi

# Install PyTorch first (with CUDA support if available)
echo ""
echo "📦 Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected, installing PyTorch with CUDA support"
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "💻 No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo ""
echo "📦 Installing other dependencies..."
pip3 install -r requirements.txt

# Install package in development mode
echo ""
echo "📦 Installing DeepRetina-MSHA package..."
pip3 install -e .

echo ""
echo "✅ Installation completed!"
echo ""
echo "🔧 Quick setup check:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "🚀 Ready to use DeepRetina-MSHA!"
echo "   • Run notebook: jupyter lab notebooks/DeepRetina_MSHA_Complete.ipynb"
echo "   • Train model: python main.py --mode train"
echo "   • See README.md for more options"