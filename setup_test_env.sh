#!/bin/bash

# Exit on error
set -e

echo "Setting up test environment for siRNA analysis pipeline..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Function to check if we're in the correct directory
check_project_root() {
    if [ ! -f "environment.yml" ] || [ ! -f "setup_env.sh" ]; then
        echo "Error: Please run this script from the project root directory"
        exit 1
    fi
}

# Function to handle conda environment setup
setup_conda_env() {
    echo "Setting up conda environment..."
    
    # Remove existing environment if it exists
    if conda env list | grep -q "^dnabert_env "; then
        echo "Removing existing dnabert_env environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n dnabert_env -y
    fi
    
    echo "Creating new dnabert_env environment..."
    conda env create -f environment.yml
    
    echo "Activating dnabert_env environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate dnabert_env
    
    # Verify environment activation
    if [ "$CONDA_DEFAULT_ENV" != "dnabert_env" ]; then
        echo "Error: Failed to activate conda environment"
        exit 1
    fi
}

# Function to initialize test data
init_test_data() {
    echo "Initializing test data..."
    python tests/init_test_data.py
    
    # Verify test data creation
    required_files=(
        "data/geo/GSE55296_count_data.txt.gz"
        "data/reference/test_genome.fa.gz"
        "data/gtex/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
        "data/encode/sample_peaks_1.bed.gz"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            echo "Error: Required test file not created: $file"
            exit 1
        fi
    done
    
    echo "Test data initialization successful"
}

# Function to verify Jupyter installation
verify_jupyter() {
    echo "Verifying Jupyter installation..."
    if ! jupyter --version > /dev/null 2>&1; then
        echo "Error: Jupyter installation failed"
        exit 1
    fi
    
    echo "Jupyter installation verified successfully"
}

# Function to run main setup script
run_main_setup() {
    echo "Running main setup script..."
    if [ -x "setup_env.sh" ]; then
        ./setup_env.sh
    else
        chmod +x setup_env.sh
        ./setup_env.sh
    fi
}

# Main setup process
main() {
    echo "Starting setup process..."
    
    # Verify we're in the project root
    check_project_root
    
    # Setup conda environment
    setup_conda_env
    
    # Initialize test data
    init_test_data
    
    # Verify Jupyter
    verify_jupyter
    
    # Run main setup script
    run_main_setup
    
    echo "Installing test dependencies..."
    pip install -r requirements-dev.txt
    
    echo
    echo "Test environment setup complete!"
    echo
    echo "To activate the environment:"
    echo "conda activate dnabert_env"
    echo
    echo "To run tests:"
    echo "pytest"
    echo "or"
    echo "pytest -m \"not slow\""
    echo
    echo "Note: Make sure you're in the dnabert_env environment when running tests."
}

# Run main function
main