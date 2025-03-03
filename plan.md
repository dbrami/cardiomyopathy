# Analysis of Cardiac Myopathy and RNA Therapeutics using DNABERT-2

## Project Goal
Generate optimal siRNA sequences as therapeutics for cardiomyopathies by fine-tuning DNABERT-2 on domain-specific datasets, ensuring that generated sequences are optimized for efficacy, specificity, and safety.

## Environment Setup Requirements
- **Conda Environment**: Dedicated environment using Python 3.11.
- **Package Versions**:
  - PyTorch 2.0.0 (MPS support for Mac; CPU version for other platforms)
  - numpy (<2.0.0)
  - transformers, biopython, pandas, scikit-learn, jupyterlab, geoparse, pyarrow
- **Reproducibility**: All packages are pinned to ensure reproducibility.
- **Containerization**: Consider Docker for encapsulating dependencies and deployment across environments.

## Data Sources and Input Files
1. **GEO RNA-seq Data (GSE55296)**
   - **URL**: [GSE55296_series_matrix.txt.gz](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55296/matrix/GSE55296_series_matrix.txt.gz)
   - **Purpose**: Identify candidate genes associated with cardiac myopathy.
   - **Contents**: Expression matrix, sample metadata (26 cardiomyopathy samples [13 ischemic + 13 dilated], 10 healthy controls).
   - **Enhancements**: Incorporate additional quality control and metadata parsing.

2. **GTEx Data (v10)**
   - **TPM Data URL**: [GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz](https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz)
   - **Sample Attributes URL**: [GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt](https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt)
   - **Purpose**: Validate heart-specific gene expression.
   - **Focus**: Left Ventricle and Atrial Appendage tissues.

3. **ENCODE ChIP-seq Data**
   - **Source**: [ENCODE Project](https://www.encodeproject.org/)
   - **Search Terms**: "heart tissue ChIP-seq"
   - **File Type**: 188 BED narrowPeak files.
   - **Purpose**: Provide regulatory region information.
   - **Processing**: Aggregate BED files, analyze peak characteristics (length distribution, chromosome-wise counts).

4. **Reference Genome**
   - **URL**: [Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz](https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz)
   - **Version**: GRCh38
   - **Purpose**: Extract promoter sequences (1000bp upstream of TSS).

5. **siRNA Reference Data (For Model Fine-Tuning)**
   - **Purpose**: A curated dataset of known effective siRNA sequences for cardiomyopathies or similar targets.
   - **Enhancements**: May include synthetic data and experimentally validated sequences to bolster training.

## Analysis Pipeline Steps

### 1. GEO RNA-seq Analysis
- **Input**: GSE55296 series matrix and count data.
- **Processes**:
  - Parse and validate metadata.
  - Extract expression matrix.
  - Perform differential expression analysis with appropriate statistical tests.
  - Implement quality control and logging.
- **Output**: Candidate gene list (up-regulated in cardiomyopathy).

### 2. GTEx Data Processing
- **Input**: GTEx TPM data and sample attributes.
- **Processes**:
  - Filter for heart-specific samples.
  - Validate candidate gene expression in heart tissue.
- **Output**: Validated candidate gene list with heart tissue expression profiles.

### 3. ENCODE ChIP-seq Integration
- **Input**: 188 ENCODE BED files.
- **Processes**:
  - Aggregate and process BED files.
  - Analyze peak characteristics (length distributions, chromosome-wise counts).
  - Generate visualizations for regulatory region annotations.
- **Output**: Detailed regulatory region statistics and plots.

### 4. Reference Genome Processing
- **Input**: GRCh38 primary assembly.
- **Processes**:
  - Extract promoter regions (1000bp upstream of TSS).
  - Generate sequences for candidate genes.
  - Ensure robustness with error handling.
- **Output**: Promoter sequences for further analysis.

### 5. DNABERT-2 Fine-Tuning and siRNA Sequence Generation
- **Input**: 
  - Promoter sequences extracted from candidate genes.
  - Curated siRNA reference data for fine-tuning.
- **Processes**:
  - Fine-tune DNABERT-2 on the siRNA dataset to capture sequence motifs and regulatory patterns.
  - Generate DNA embeddings and use these to design siRNA sequences with optimal targeting properties.
  - Integrate evaluation metrics (e.g., off-target prediction, RNA secondary structure analysis, thermodynamic stability) into the model training and post-processing steps.
  - Perform PCA for dimensionality reduction and clustering to visualize relationships between generated sequences.
- **Output**:
  - Optimal siRNA sequence recommendations.
  - Promoter sequence embeddings.
  - PCA and clustering visualizations.
  - A ranked list of candidate siRNA sequences based on therapeutic potential.

## Expected Outputs
- **Candidate Gene List**:
  - Differentially expressed and heart-validated genes.
- **Regulatory Analysis**:
  - Peak distribution plots and chromosome-wise counts.
- **Sequence Analysis**:
  - Promoter sequences with corresponding DNABERT-2 embeddings.
  - PCA plots of regulatory element clusters.
- **siRNA Therapeutic Recommendations**:
  - A ranked list of optimal siRNA sequences.
  - Evaluation metrics for each candidate (specificity, off-target risk, thermodynamic stability).
- **Integrated Findings**:
  - A comprehensive multi-omics view of cardiac myopathy.
  - Detailed insights into sequence-based regulatory mechanisms and potential therapeutic interventions.

## Additional Functionality and Future Enhancements
- **Error Handling & Logging**:
  - Implement robust logging and error recovery mechanisms.
- **Interactive Visualizations**:
  - Develop interactive dashboards for exploratory data analysis.
- **Workflow Automation**:
  - Use a workflow management tool (e.g., Snakemake or Nextflow) for pipeline orchestration.
- **Parameterization**:
  - Allow user-configurable parameters for filtering, analysis thresholds, and visualization settings.
- **Data Integration**:
  - Plan for future integration of clinical metadata and additional gene annotations (ENSEMBL/UCSC).
- **Advanced Model Optimization**:
  - Continue to refine the fine-tuning process for DNABERT-2 by incorporating new siRNA data and feedback from experimental validations.
- **Containerization**:
  - Consider Dockerizing the pipeline for improved reproducibility and deployment.