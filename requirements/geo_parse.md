- Description: Maps NCBI Gene IDs to gene symbols, Ensembl IDs, and genomic annotations.

3. **Sample Description File Path**:
- Path: e.g., `"path/to/GSE55296_processed_data_readme.txt"`
- Format: Plain text, assumed to describe sample groups (ischemic, dilated, healthy)
- Expected Content: Metadata linking GSM IDs to conditions (e.g., ischemic cardiomyopathy, dilated cardiomyopathy, healthy control). For GSE55296, this should describe 13 ischemic, 13 dilated, and 10 control samples. Exact format needs clarification but assumed parseable (e.g., tab-separated or structured text).

## Output
- **Return Type**: `pandas.DataFrame`
- **Headers**:
- `GeneID`: NCBI Gene ID (e.g., `100287102`)
- `Symbol`: HGNC gene symbol (e.g., `DDX11L1`)
- `Description`: Gene description (e.g., `DEAD/H-box helicase 11 like 1 (pseudogene)`)
- `EnsemblGeneID`: Ensembl gene ID (e.g., `ENSG00000290825`)
- `Log2FC`: Log2 fold change between cardiomyopathy and control groups
- `MeanExprCardio`: Mean FPKM expression in cardiomyopathy samples
- `MeanExprControl`: Mean FPKM expression in control samples
- **Description**: A table of the top differentially regulated genes (up-regulated in cardiomyopathy vs. controls), sorted by `Log2FC` in descending order. The number of genes returned is configurable via a parameter with a default of 50.

## Requirements
### Functionality
1. **File Loading**:
- Read compressed TSV files (`FPKM` and `annotation`) using `gzip` and `pandas`.
- Parse the sample description file to categorize GSM samples into groups (ischemic, dilated, healthy).

2. **Sample Categorization**:
- Identify 26 cardiomyopathy samples (13 ischemic, 13 dilated) and 10 control samples based on `GSE55296_processed_data_readme.txt`.
- Map GSM IDs to conditions (e.g., ischemic, dilated, healthy).

3. **Differential Expression Analysis**:
- Calculate mean FPKM expression for cardiomyopathy and control groups.
- Compute log2 fold change with a pseudocount (e.g., `log2(mean_cardio + 1) - log2(mean_control + 1)`).
- Filter and sort genes by `Log2FC` to select the top N up-regulated genes, where N is a parameter (default 50).

4. **Gene Annotation**:
- Merge FPKM data with annotation data on `GeneID` to include `Symbol`, `Description`, and `EnsemblGeneID`.

5. **Error Handling**:
- Handle missing files, malformed data, or mismatched GSM IDs with informative exceptions.

### Dependencies
- **Python Version**: 3.8+
- **Packages**:
- `pandas` (v2.0+): For DataFrame operations and TSV parsing.
- Install: `pip install pandas`
- `numpy` (v1.24+): For numerical computations (e.g., log2 fold change).
- Install: `pip install numpy`
- `gzip`: Built-in Python module for handling compressed files.
- (Optional) `requests`: For potential API extensions (e.g., sequence retrieval).
- Install: `pip install requests`

### Module Interface
```python
# geo_module.py
def get_differentially_regulated_genes(fpkm_file_path: str, annot_file_path: str, sample_desc_path: str, top_n: int = 50) -> pd.DataFrame:
"""
Identify top N differentially regulated genes from GEO FPKM data.

Args:
   fpkm_file_path (str): Path to compressed FPKM file (e.g., 'GSE55296_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz').
   annot_file_path (str): Path to compressed annotation file (e.g., 'Human.GRCh38.p13.annot.tsv.gz').
   sample_desc_path (str): Path to sample description file (e.g., 'GSE55296_processed_data_readme.txt').
   top_n (int, optional): Number of top genes to return. Defaults to 50.
   