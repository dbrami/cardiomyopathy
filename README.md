# Cardiomyopathy Project: DNABERT for Cardiac Myopathy & RNA Therapeutics

## 1. Project Overview

- **Goal:**  
  Leverage **DNABERT**, a transformer model for DNA sequences, to analyze genomic and transcriptomic data related to **cardiac myopathy**. The project aims to identify sequence motifs and nucleotide patterns that affect siRNA target specificity and off-target binding in the context of RNA therapeutics.

- **Hypothesis:**  
  Specific nucleotide patterns in genes associated with cardiac myopathy—especially in the regions targeted by siRNAs—correlate with the efficacy and safety of RNA-based therapeutic interventions. DNABERT can be fine-tuned to distinguish between on-target and off-target binding sites by identifying these patterns.

---

## 2. Project Setup

### Hardware and Software
- **Hardware:**  
  Apple **MacBook Air M3** (Apple Silicon). The model is chosen with the awareness of limited resources, so we will use the base DNABERT model and keep batch sizes small.

- **Required Software:**
  - **Python 3.11**
  - **Conda** (for environment management)
  - **PyTorch** (with MPS support on Mac)
  - **Jupyter Notebook**
  - **DNABERT_2** (cloned from its GitHub repository)
  - Additional libraries: Hugging Face Transformers, Pandas, NumPy, scikit-learn, Biopython, etc.

### Environment Setup
A script (see below) can be used to set up a conda environment, install dependencies, and create the required directory structure:
- **Directories:**
  - `data/` – for all downloaded raw data files.
  - `models/` – for saving pretrained and fine-tuned models.
  - `notebooks/` – for Jupyter Notebooks.
  - `results/` – for output files and figures.
  - `src/` – for source code, including the DNABERT repository.

---

## 3. Data Acquisition

Identify and download datasets relevant to both genomic and transcriptomic analyses:

### Required Datasets
- **GEO RNA-seq Dataset (GSE55296):**
  Contains gene expression profiles from heart tissue (heart failure patients with cardiomyopathy vs. healthy controls).
  - **Access:** Automatically downloaded by setup script
  - **Location:** `data/geo/GSE55296_series_matrix.txt`
  - **Size:** Series matrix data (~2 MB)

- **ENCODE Heart Tissue Data:**
  ChIP-seq peaks or regulatory region annotations from human heart tissue (e.g., left ventricle).
  - **Access:** Manual download required
  - **Steps:**
    1. Visit [ENCODE Portal](https://www.encodeproject.org/)
    2. Search for "heart tissue ChIP-seq"
    3. Download relevant BED files to `data/encode/`
  - **Size:** BED files typically range from a few MBs

- **GTEx Heart Tissue Expression Data:**
  Expression levels of genes across heart tissue samples (Heart - Left Ventricle and Heart - Atrial Appendage).
  - **Access:** Direct download via script
  - **Files:**
    1. RNA-seq Gene TPM Data (v8) (~1-2 GB):
       - Contains transcript-per-million values for all genes
       - Located at: `data/gtex/GTEx_Analysis_v8_RNA-seq_RNA-SeQCv1.1.9_gene_tpm.gct.gz`
    2. Sample Attributes:
       - Metadata for filtering heart-specific samples
       - Located at: `data/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt`
    3. Subject Phenotypes:
       - Donor clinical and demographic data
       - Located at: `data/gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt`

- **Human Genome Reference (GRCh38):**
  Primary assembly from Ensembl (release 109) for sequence retrieval.
  - **Access:** Automatically downloaded by setup script
  - **Location:** `data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz`
  - **Size:** Several hundred MBs compressed

---

## 4. Data Processing

### Steps to Prepare Data
- **Quality Control and Cleaning:**
  - For raw sequencing data: Use tools like **FastQC** and **Trimmomatic** (if needed).
  - For processed count matrices: Use Pandas to load and inspect data for outliers and normalization.
  
- **Sequence Retrieval and Formatting:**
  - Identify key cardiomyopathy-related genes (e.g., *TTN, MYH7, TNNT2*).
  - Use Ensembl BioMart or UCSC Genome Browser to fetch nucleotide sequences (mRNA, 3’ UTRs, or promoter regions) for these genes.
  - Convert the sequences to k-mer tokens (recommended: 6-mers) using DNABERT’s provided tokenization utilities.  
    Example:
    ```python
    from dnabert import seq2kmer
    kmer_tokens = seq2kmer("ATGCGATC...", k=6)
    ```

- **Labeling for Model Training:**
  - **Positive examples:** Sequences known to be on-target binding sites (intended siRNA binding regions).
  - **Negative examples:** Potential off-target sequences identified through sequence similarity (e.g., via BLAST or simple string matching for the siRNA seed region).

- **Annotation:**
  - Annotate sequences with additional features like GC content, expression levels from GTEx/GEO, and genomic coordinates to aid later analysis.

---

## 5. Model Application

### Fine-Tuning DNABERT
- **Pre-trained Model:**  
  Download the pre-trained DNABERT checkpoint for 6-mers from the DNABERT GitHub repository.

- **Fine-Tuning Task:**  
  Create a binary classification task where the model learns to distinguish between on-target (effective) and off-target (potentially problematic) sequences.

- **Training Pipeline:**
  - Prepare data in the required format (e.g., TSV with columns for `sequence` and `label`).
  - Use the DNABERT fine-tuning script or Hugging Face’s Trainer API. For example:
    ```bash
    python run_finetune.py \
      --model_type dna \
      --tokenizer_name dna6 \
      --model_name_or_path <path_to_pretrained_model> \
      --task_name siRNA_classification \
      --do_train --do_eval \
      --data_dir <path_to_prepared_data> \
      --max_seq_length 100 \
      --per_device_train_batch_size 16 \
      --learning_rate 2e-4 \
      --num_train_epochs 3 \
      --output_dir <finetuned_model_dir>
    ```
  - Configure the training to use Apple’s MPS (if available) by setting the device accordingly in your training script.

- **Evaluation:**
  - Evaluate model performance using metrics such as accuracy, precision, recall, and the confusion matrix.
  - Assess whether the model correctly distinguishes on-target from off-target sequences.

---

## 6. Analysis and Insights

### Key Analyses
- **Sequence Motif Discovery:**
  - Utilize DNABERT’s attention mechanisms to highlight key nucleotide regions (e.g., the siRNA seed match) that drive predictions.
  - Generate sequence logos for identified motifs.

- **Integration with Expression Data:**
  - Cross-reference model predictions with gene expression data (GEO and GTEx) to see if predicted off-target genes are expressed in heart tissue.
  - Prioritize candidate siRNA targets based on both sequence prediction and expression relevance.

- **Biological Implications:**
  - Analyze whether the identified motifs correlate with known issues in siRNA therapeutics (e.g., off-target effects leading to adverse events).
  - Discuss potential improvements for RNA therapeutic design based on the model’s insights.

---

## 7. Visualization and Reporting

### Reporting in Jupyter Notebook
- **Notebook Structure:**
  - **Introduction:** Explain the project’s objectives and hypothesis.
  - **Environment Setup:** Provide code and output verifying the environment (e.g., MPS availability).
  - **Data Acquisition & Processing:** Display sample data tables and tokenized sequence examples.
  - **Model Training:** Show training curves and performance metrics.
  - **Prediction & Interpretation:** Include attention heatmaps, sequence logos, and embedding plots (e.g., t-SNE/PCA).
  - **Conclusions:** Summarize findings and propose next steps.

### Visualization Tools
- Use **matplotlib** for plotting training curves, confusion matrices, and attention heatmaps.
- Generate **sequence logos** using available tools (e.g., WebLogo) to visually represent discovered motifs.
- Plot multi-dimensional scaling of sequence embeddings to demonstrate class separation.

---

## 8. Conclusion

This weekend project outlines an end-to-end pipeline using DNABERT to analyze nucleotide sequences related to cardiac myopathy, with the goal of improving siRNA target selection for RNA therapeutics. By integrating genomic data from GEO, ENCODE, GTEx, and reference sequences from Ensembl, and by fine-tuning DNABERT on a custom dataset of on-target and off-target sites, the project aims to provide actionable insights into RNA therapeutic design. The final deliverable is a comprehensive Jupyter Notebook that documents the entire process—from environment setup and data processing to model training and biological interpretation.

---

## Appendix: Example Scripts

### Bash Script for Environment Setup
```bash
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

# Install other dependencies
pip install transformers datasets pandas numpy scikit-learn jupyterlab tqdm biopython regex sentencepiece

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
echo "To activate the environment: conda activate dnabert_env"
