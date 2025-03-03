"""
Initialize test data directories with sample files
"""

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip

def create_directories(project_root):
    """Create required data directories"""
    dirs = [
        'data/geo',
        'data/encode',
        'data/gtex',
        'data/reference',
        'data/logs'
    ]
    
    for dir_path in dirs:
        Path(project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
def create_geo_data(project_root):
    """Create sample GEO data files"""
    geo_dir = project_root / 'data/geo'
    
    # Create count data
    count_data = pd.DataFrame({
        'gene_id': [f'ENSG{i:08d}' for i in range(100)],
        'gene_name': [f'GENE{i}' for i in range(100)]
    })
    
    # Add sample columns (26 cardio + 10 control)
    for i in range(36):
        col_name = f'GSM{1000+i}'
        if i < 26:  # Cardio samples
            count_data[col_name] = np.random.normal(100, 10, size=100)
        else:  # Control samples
            count_data[col_name] = np.random.normal(50, 10, size=100)
    
    # Save as gzipped file
    count_file = geo_dir / 'GSE55296_count_data.txt.gz'
    with gzip.open(count_file, 'wt') as f:
        count_data.to_csv(f, sep='\t', index=False)

def create_reference_genome(project_root):
    """Create sample reference genome file"""
    ref_dir = project_root / 'data/reference'
    
    # Create sample chromosomes
    records = []
    for i in range(5):
        seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], size=10000))
        record = SeqRecord(
            Seq(seq),
            id=f'chr{i+1}',
            description=f'Test chromosome {i+1}'
        )
        records.append(record)
    
    # Save as gzipped FASTA
    genome_file = ref_dir / 'test_genome.fa.gz'
    with gzip.open(genome_file, 'wt') as f:
        SeqIO.write(records, f, 'fasta')

def create_gtex_data(project_root):
    """Create sample GTEx data files"""
    gtex_dir = project_root / 'data/gtex'
    
    # Create TPM data
    tpm_data = pd.DataFrame(
        np.random.normal(50, 10, size=(100, 20)),
        columns=[f'GTEX{i:08d}' for i in range(20)]
    )
    
    # Save as gzipped GCT
    tpm_file = gtex_dir / 'GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz'
    with gzip.open(tpm_file, 'wt') as f:
        f.write("#1.2\n")  # GCT version
        f.write(f"{tpm_data.shape[0]}\t{tpm_data.shape[1]}\n")  # Dimensions
        tpm_data.to_csv(f, sep='\t')

def create_encode_data(project_root):
    """Create sample ENCODE peak data files"""
    encode_dir = project_root / 'data/encode'
    
    # Create sample peak data
    for i in range(3):  # Create 3 sample files
        peaks = pd.DataFrame({
            'chr': [f'chr{random.randint(1, 22)}' for _ in range(100)],
            'start': [random.randint(1, 1000000) for _ in range(100)],
            'end': [random.randint(1000000, 2000000) for _ in range(100)],
            'name': [f'peak_{j}' for j in range(100)],
            'score': np.random.randint(0, 1000, 100),
            'strand': np.random.choice(['+', '-'], 100)
        })
        
        # Save as gzipped BED
        peak_file = encode_dir / f'sample_peaks_{i+1}.bed.gz'
        with gzip.open(peak_file, 'wt') as f:
            peaks.to_csv(f, sep='\t', header=False, index=False)

def main():
    """Main function to initialize test data"""
    project_root = Path(__file__).parent.parent
    print(f"Initializing test data in {project_root}")
    
    try:
        # Create directory structure
        create_directories(project_root)
        
        # Create sample data files
        print("Creating GEO data...")
        create_geo_data(project_root)
        
        print("Creating reference genome...")
        create_reference_genome(project_root)
        
        print("Creating GTEx data...")
        create_gtex_data(project_root)
        
        print("Creating ENCODE data...")
        create_encode_data(project_root)
        
        print("Test data initialization complete!")
        
    except Exception as e:
        print(f"Error initializing test data: {e}")
        raise

if __name__ == '__main__':
    import random
    random.seed(42)  # For reproducibility
    np.random.seed(42)
    main()