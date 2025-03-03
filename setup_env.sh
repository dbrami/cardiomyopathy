#!/bin/bash

# Exit immediately if any command fails.
set -e

echo "Setting up project environment..."

# Function to create directory if it doesn't exist
create_dir_if_missing() {
    if [ ! -d "$1" ]; then
        echo "Creating directory: $1"
        mkdir -p "$1"
    else
        echo "Directory already exists: $1"
    fi
}

# Create project directories for external data files
echo "Setting up project directories..."
create_dir_if_missing "data"
create_dir_if_missing "models"
create_dir_if_missing "notebooks"
create_dir_if_missing "results"
create_dir_if_missing "src"

# Check if yq is installed, install if missing
if ! command -v yq &> /dev/null; then
    echo "Installing yq for YAML processing..."
    if [[ "$(uname)" == "Darwin" ]]; then
        brew install yq
    else
        echo "Please install yq manually:"
        echo "Ubuntu/Debian: sudo snap install yq"
        echo "Other: See https://github.com/mikefarah/yq#install"
        exit 1
    fi
fi

echo "Creating conda environment for DNABERT_2 with Python 3.11..."
# Create a new conda environment with Python 3.11 if it doesn't exist
if ! conda info --envs | grep -q "^dnabert_env"; then
    conda create -n dnabert_env python=3.11 -y
fi

# Activate the environment
echo "Activating the dnabert_env environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dnabert_env

echo "Installing Python dependencies..."
# Upgrade pip and install essential packages
pip install --upgrade pip

# Install PyTorch 2.0.0 (known compatible version) with appropriate backend
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Installing PyTorch with MPS support for Mac..."
    pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0
else
    echo "Installing PyTorch (CPU version)..."
    pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies including PyYAML
# Pin numpy to 1.x to avoid compatibility issues
pip install "numpy<2.0.0" ipywidgets
pip install transformers datasets pandas scikit-learn jupyterlab tqdm 
pip install biopython regex sentencepiece matplotlib pyyaml pyarrow geoparse

# Enable Jupyter widgets
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Check if DNABERT_2 is already cloned
if [ ! -d "src/DNABERT_2" ]; then
    echo "Cloning DNABERT_2 repository into src/ directory..."
    cd src
    git clone https://github.com/MAGICS-LAB/DNABERT_2.git
    cd DNABERT_2
else
    echo "DNABERT_2 repository already exists, updating..."
    cd src/DNABERT_2
    git pull origin main
fi

echo "Installing DNABERT_2 requirements (excluding PyTorch)..."
# Install requirements excluding torch (since we installed it separately)
grep -v "torch==" requirements.txt | pip install -r /dev/stdin

# Note: DNABERT_2 will be used directly from the src directory
# Adding src to PYTHONPATH for imports
REPO_ROOT=$(pwd)/../..
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_ROOT" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$REPO_ROOT" >> ~/.zshrc

# Optional: Install DNABERT-S, the foundation model for generating DNA embeddings.
if [ -d "DNABERT-S" ]; then
    echo "Installing DNABERT-S for DNA embeddings..."
    cd DNABERT-S
    if [ -f "requirements.txt" ]; then
        grep -v "torch==" requirements.txt | pip install -r /dev/stdin
    fi
    if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
        pip install --editable .
    fi
    cd ..
else
    echo "DNABERT-S directory not found. If DNABERT-S is integrated into DNABERT_2, no separate installation is needed."
fi

echo
echo "Environment setup complete!"
echo "Project directories verified: data, models, notebooks, results, src"
echo "Dependencies installed:"
echo "- yq (system-level YAML processor)"
echo "- PyYAML (Python YAML library)"
echo "Note: The src directory has been added to PYTHONPATH for DNABERT_2 imports"
echo "To activate the environment: conda activate dnabert_env"
