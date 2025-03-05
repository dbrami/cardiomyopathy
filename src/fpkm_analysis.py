import pandas as pd
import gzip
import numpy as np
from scipy import stats
from pathlib import Path

def get_path(base_dir, filename):
    """
    Construct path and ensure it exists
    
    Parameters:
    base_dir (str or Path): Base directory
    filename (str): Filename to append
    
    Returns:
    Path: Complete path
    """
    path = Path(base_dir) / filename
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def process_fpkm_data(series_matrix_path, counts_path, logger=None):
    """
    Process FPKM data and sample information
    
    Parameters:
    series_matrix_path (str): Path to series matrix file
    counts_path (str): Path to counts/FPKM file
    logger (logging.Logger, optional): Logger instance
    
    Returns:
    tuple: (processed DataFrame, sample info dictionary)
    """
    try:
        # Load FPKM data
        with gzip.open(counts_path, 'rt') as f:
            fpkm_df = pd.read_csv(f, sep='\t', index_col=0)
            
        # Load sample info from series matrix
        sample_info = {}
        with open(series_matrix_path, 'r') as f:
            for line in f:
                if line.startswith('!Sample_'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        sample_id = parts[1].strip('"')
                        if line.startswith('!Sample_title'):
                            condition = 'cardiomyopathy' if 'disease' in parts[1].lower() else 'healthy'
                            if sample_id in sample_info:
                                sample_info[sample_id]['group'] = condition
                            else:
                                sample_info[sample_id] = {'group': condition}
                                
        if logger:
            logger.info(f"Loaded {len(fpkm_df)} genes and {len(sample_info)} samples")
            
        return fpkm_df, sample_info
        
    except Exception as e:
        if logger:
            logger.error(f"Error processing FPKM data: {e}")
        raise

def load_sample_info(readme_path):
    """
    Load sample information from the readme file.
    
    Parameters:
    readme_path (str): Path to the processed_data_readme.txt file
    
    Returns:
    dict: Dictionary mapping sample accession to condition
    """
    sample_info = {}
    try:
        with open(readme_path, 'r') as f:
            # Skip header lines
            lines = f.readlines()[2:]  # Skip first two lines based on sample data
            for line in lines:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        accession = parts[0]
                        title = parts[1]
                        # Extract condition from title
                        if 'Control' in title:
                            sample_info[accession] = 'Control'
                        elif 'Ischemic' in title:
                            sample_info[accession] = 'Ischemic'
                        elif 'Dilated' in title:
                            sample_info[accession] = 'Dilated'
        return sample_info
    except Exception as e:
        print(f"Error loading sample info: {e}")
        return None

def load_and_analyze_fpkm(fpkm_path, readme_path, fold_change_threshold=2.0, p_value_threshold=0.05):
    """
    Load compressed FPKM file and identify differentially expressed genes.
    
    Parameters:
    fpkm_path (str): Path to the compressed FPKM file
    readme_path (str): Path to the processed_data_readme.txt file
    fold_change_threshold (float): Minimum fold change for differential expression (default: 2.0)
    p_value_threshold (float): Maximum p-value for significance (default: 0.05)
    
    Returns:
    dict: Dictionary containing up-regulated and down-regulated gene IDs
    """
    
    # Load sample information
    sample_info = load_sample_info(readme_path)
    if sample_info is None:
        return None

    # Load the compressed FPKM file
    try:
        with gzip.open(fpkm_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error loading FPKM file: {e}")
        return None

    # Filter for samples in sample_info
    available_samples = [col for col in df.columns if col in sample_info]
    df = df[available_samples]

    # Separate Control and Ischemic samples
    control_samples = [sample for sample, condition in sample_info.items() 
                      if condition == 'Control' and sample in df.columns]
    ischemic_samples = [sample for sample, condition in sample_info.items() 
                       if condition == 'Ischemic' and sample in df.columns]

    # Calculate mean expression for each condition
    control_mean = df[control_samples].mean(axis=1)
    ischemic_mean = df[ischemic_samples].mean(axis=1)

    # Calculate fold change (add small constant to avoid division by zero)
    epsilon = 1e-8
    fold_change = ischemic_mean / (control_mean + epsilon)

    # Perform t-test for each gene
    p_values = []
    for gene in df.index:
        control_vals = df.loc[gene, control_samples].values
        ischemic_vals = df.loc[gene, ischemic_samples].values
        t_stat, p_val = stats.ttest_ind(control_vals, ischemic_vals, equal_var=False)
        p_values.append(p_val)
    
    # Create results dataframe
    results = pd.DataFrame({
        'fold_change': fold_change,
        'p_value': p_values,
        'control_mean': control_mean,
        'ischemic_mean': ischemic_mean
    }, index=df.index)

    # Filter for differentially expressed genes
    up_regulated = results[
        (results['fold_change'] > fold_change_threshold) & 
        (results['p_value'] < p_value_threshold) &
        (results['ischemic_mean'] > 0.1)  # Minimum expression threshold
    ]
    
    down_regulated = results[
        (results['fold_change'] < 1/fold_change_threshold) & 
        (results['p_value'] < p_value_threshold) &
        (results['control_mean'] > 0.1)  # Minimum expression threshold
    ]

    # Return gene IDs
    return {
        'up_regulated': up_regulated.index.tolist(),
        'down_regulated': down_regulated.index.tolist()
    }