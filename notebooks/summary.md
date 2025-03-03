# Analysis of Cardiac Myopathy and RNA Therapeutics using DNABERT-2

## Environment Setup

The project uses a dedicated conda environment with specific package versions to ensure reproducibility:

- Python 3.11
- PyTorch 2.0.0 (specific version for compatibility)
- numpy < 2.0.0 (pinned to avoid compatibility issues)
- Key packages:
  - transformers: For DNABERT-2 model
  - biopython: For DNA sequence handling
  - pandas, scikit-learn: For data analysis
  - jupyterlab: For notebook interface
  - geoparse: For GEO data parsing
  - pyarrow: For efficient data storage

### Platform-Specific Optimizations
- Mac: PyTorch installed with MPS support
- Other platforms: CPU version of PyTorch

## Data Sources and Input Files

### 1. GEO RNA-seq Data (GSE55296)
- **URL**: https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55296/matrix/GSE55296_series_matrix.txt.gz
- **Purpose**: Provides expression data for identifying candidate genes associated with cardiac myopathy
- **Contents**: 
  - Expression matrix
  - Sample metadata for cardiomyopathy and control groups
  - 26 cardiomyopathy samples (13 ischemic + 13 dilated)
  - 10 healthy control samples

### 2. GTEx Data (v10)
- **TPM Data URL**: https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz
- **Sample Attributes URL**: https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt
- **Purpose**: Provides tissue-specific expression profiles to validate cardiac relevance
- **Focus**: Heart-specific samples (Left Ventricle and Atrial Appendage)

### 3. ENCODE ChIP-seq Data
- **Source**: https://www.encodeproject.org/
- **Search Terms**: "heart tissue ChIP-seq"
- **Purpose**: Provides regulatory peak information from 188 ChIP-seq experiments
- **File Type**: BED narrowPeak format (compressed)

### 4. Reference Genome
- **URL**: https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
- **Version**: GRCh38
- **Purpose**: Used for extracting promoter sequences for candidate genes

## Analysis Pipeline Steps

### 1. GEO RNA-seq Analysis
- **Input**: GSE55296 series matrix and count data
- **Process**: 
  - Parse GEO series matrix for metadata
  - Extract expression matrix
  - Perform differential expression analysis
- **Output**: List of candidate genes up-regulated in cardiomyopathy

### 2. GTEx Data Processing
- **Input**: GTEx TPM data and sample attributes
- **Process**:
  - Filter for heart tissue samples
  - Extract heart-specific expression profiles
- **Output**: Heart-specific expression validation for candidate genes

### 3. ENCODE ChIP-seq Integration
- **Input**: 188 ENCODE BED files
- **Process**:
  - Aggregate all BED files
  - Analyze peak characteristics
  - Calculate peak length distributions
  - Generate chromosome-wise peak counts
- **Output**: Regulatory region annotations and peak statistics

### 4. Reference Genome Processing
- **Input**: GRCh38 primary assembly
- **Process**:
  - Extract promoter regions (1000bp upstream of TSS)
  - Generate sequences for candidate genes
- **Output**: Promoter sequences for further analysis

### 5. DNABERT-2 Analysis
- **Input**: Extracted promoter sequences
- **Process**:
  - Generate DNA embeddings using DNABERT-2
  - Perform PCA for dimensionality reduction
  - Cluster similar regulatory elements
- **Output**: 
  - Promoter sequence embeddings
  - PCA visualization of regulatory element clusters

## Expected Outputs

1. **Candidate Gene List**
   - Up-regulated genes in cardiomyopathy
   - Validated for heart tissue expression

2. **Regulatory Analysis**
   - Peak distribution plots
   - Chromosome-wise peak counts
   - Regulatory region annotations

3. **Sequence Analysis**
   - Promoter sequences for candidate genes
   - DNABERT-2 embeddings
   - PCA plots of regulatory element clusters

4. **Integrated Findings**
   - Multi-omics view of cardiac myopathy
   - Potential siRNA target recommendations
   - Regulatory motif patterns

## Future Steps

1. **GEO Analysis Enhancement**
   - Improve parsing robustness
   - Better metadata integration

2. **Annotation Integration**
   - Add ENSEMBL/UCSC gene coordinates
   - Improve promoter/enhancer sequence extraction

3. **DNABERT-2 Optimization**
   - Create labeled datasets for siRNA targeting
   - Fine-tune model for target prediction

4. **Motif Analysis**
   - Implement attention-based motif discovery
   - Validate against known regulatory elements