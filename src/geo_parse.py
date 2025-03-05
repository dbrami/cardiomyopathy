import pandas as pd
import numpy as np
import gzip
import os

def get_differentially_regulated_genes(fpkm_file_path: str, annot_file_path: str, sample_desc_path: str, top_n: int = 50) -> pd.DataFrame:
    """
    Identify top N differentially regulated genes from GEO FPKM data.

    Args:
        fpkm_file_path (str): Path to compressed FPKM file (e.g., 'GSE55296_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz').
        annot_file_path (str): Path to compressed annotation file (e.g., 'Human.GRCh38.p13.annot.tsv.gz').
        sample_desc_path (str): Path to sample description file (e.g., 'GSE55296_processed_data_readme.txt').
        top_n (int, optional): Number of top genes to return. Defaults to 50.

    Returns:
        pd.DataFrame: Table with columns ['GeneID', 'Symbol', 'Description', 'EnsemblGeneID', 'Log2FC', 
                      'MeanExprCardio', 'MeanExprControl'] containing top N up-regulated genes.

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: If data formats are invalid, GSM IDs don’t match expected groups, or top_n is invalid.
    """
    # Validate inputs
    for path in [fpkm_file_path, annot_file_path, sample_desc_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    if not isinstance(top_n, int) or top_n <= 0:
        raise ValueError(f"top_n must be a positive integer, got {top_n}")

    # Load FPKM data
    print(f"Loading FPKM data from {fpkm_file_path}...")
    with gzip.open(fpkm_file_path, 'rt') as f:
        fpkm_df = pd.read_csv(f, sep='\t')
    if 'GeneID' not in fpkm_df.columns:
        raise ValueError("FPKM file must contain a 'GeneID' column")

    # Load annotation data
    print(f"Loading annotation data from {annot_file_path}...")
    with gzip.open(annot_file_path, 'rt') as f:
        annot_df = pd.read_csv(f, sep='\t')
    required_annot_cols = ['GeneID', 'Symbol', 'Description', 'EnsemblGeneID']
    if not all(col in annot_df.columns for col in required_annot_cols):
        raise ValueError("Annotation file must contain 'GeneID', 'Symbol', 'Description', 'EnsemblGeneID' columns")

    # Load and parse sample description
    print(f"Loading sample description from {sample_desc_path}...")
    with open(sample_desc_path, 'r') as f:
        lines = f.readlines()
    
    # Assume TSV format: GSM_ID\tCondition (e.g., "GSM1333767\tischemic")
    try:
        sample_groups = {line.split('\t')[0].strip(): line.split('\t')[1].strip().lower() 
                        for line in lines if line.strip() and '\t' in line and 'GSM' in line}
    except IndexError:
        raise ValueError("Sample description file must be tab-separated with GSM ID and condition columns")

    # Validate sample groups
    expected_samples = 36  # 13 ischemic, 13 dilated, 10 controls
    if len(sample_groups) != expected_samples:
        print(f"Warning: Expected {expected_samples} samples, found {len(sample_groups)} in sample description")

    # Categorize samples
    sample_cols = [col for col in fpkm_df.columns if col.startswith('GSM')]
    cardio_cols = [col for col in sample_cols if col in sample_groups and 
                   ('ischemic' in sample_groups[col] or 'dilated' in sample_groups[col])]
    control_cols = [col for col in sample_cols if col in sample_groups and 'healthy' in sample_groups[col]]

    if len(cardio_cols) != 26 or len(control_cols) != 10:
        print(f"Warning: Expected 26 cardiomyopathy and 10 control samples, found {len(cardio_cols)} and {len(control_cols)}")
    
    if not cardio_cols or not control_cols:
        raise ValueError("No valid cardiomyopathy or control sample groups identified")

    print(f"Found {len(cardio_cols)} cardiomyopathy samples and {len(control_cols)} control samples.")
    print("Cardiomyopathy samples:", cardio_cols)
    print("Control samples:", control_cols)

    # Differential expression analysis
    mean_expr_cardio = fpkm_df[cardio_cols].mean(axis=1)
    mean_expr_control = fpkm_df[control_cols].mean(axis=1)
    log2_fc = np.log2(mean_expr_cardio + 1) - np.log2(mean_expr_control + 1)

    # Create differential expression results
    de_results = pd.DataFrame({
        'GeneID': fpkm_df['GeneID'],
        'Log2FC': log2_fc,
        'MeanExprCardio': mean_expr_cardio,
        'MeanExprControl': mean_expr_control
    })

    # Merge with annotation data
    result_df = de_results.merge(annot_df[required_annot_cols], on='GeneID', how='left')

    # Handle missing annotations
    result_df['Symbol'] = result_df['Symbol'].fillna('Unknown')
    result_df['Description'] = result_df['Description'].fillna('No description')
    result_df['EnsemblGeneID'] = result_df['EnsemblGeneID'].fillna('N/A')

    # Filter out invalid Log2FC and sort
    result_df = result_df.dropna(subset=['Log2FC'])
    top_genes = result_df.sort_values('Log2FC', ascending=False).head(top_n)

    # Return required columns
    return top_genes[['GeneID', 'Symbol', 'Description', 'EnsemblGeneID', 'Log2FC', 'MeanExprCardio', 'MeanExprControl']]

# Example usage (uncomment to test)
if __name__ == "__main__":
    fpkm_path = "../data/geo/GSE55296_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz"
    annot_path = "../data/geo/Human.GRCh38.p13.annot.tsv.gz"
    desc_path = "../data/geo/GSE55296_processed_data_readme.txt"

    try:
        # Get top 50 genes (default)
        top_genes_df = get_differentially_regulated_genes(fpkm_path, annot_path, desc_path)
        print("\nTop 50 differentially regulated genes:")
        print(top_genes_df)

        # Get top 10 genes
        top_10_genes_df = get_differentially_regulated_genes(fpkm_path, annot_path, desc_path, top_n=10)
        print("\nTop 10 differentially regulated genes:")
        print(top_10_genes_df)
    except Exception as e:
        print(f"Error: {e}")