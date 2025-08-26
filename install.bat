@echo off
REM Installation script for DeepRetina-MSHA (Windows)

echo 🚀 DeepRetina-MSHA Installation Script
echo ======================================

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version') do set python_version=%%i
echo ✅ Python version: %python_version%

REM Check if we're in a virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo ⚠️  Warning: Not in a virtual environment. Consider creating one:
    echo    python -m venv venv
    echo    venv\Scripts\activate
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    echo ✅ Virtual environment: %VIRTUAL_ENV%
)

REM Check for NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo 💻 No NVIDIA GPU detected, installing CPU-only PyTorch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo 🎮 NVIDIA GPU detected, installing PyTorch with CUDA support
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)

echo.
echo 📦 Installing other dependencies...
pip install -r requirements.txt

echo.
echo 📦 Installing DeepRetina-MSHA package...
pip install -e .

echo.
echo ✅ Installation completed!
echo.
echo 🔧 Quick setup check:
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo.
echo 🚀 Ready to use DeepRetina-MSHA!
echo    • Run notebook: jupyter lab notebooks/DeepRetina_MSHA_Complete.ipynb
echo    • Train model: python main.py --mode train
echo    • See README.md for more options
pause