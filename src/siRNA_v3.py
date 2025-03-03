#!/usr/bin/env python3
"""
siRNA Analysis Pipeline for Cardiac Myopathy

This script implements a pipeline for analyzing cardiac myopathy and RNA therapeutics
using DNABERT-2. It processes GEO RNA-seq data, GTEx expression data, ENCODE ChIP-seq data,
and performs sequence analysis using DNABERT-2.
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import GEOparse
from Bio import SeqIO
from tqdm import tqdm
import io
import gzip
import random
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.decomposition import PCA

# Set matplotlib style
plt.style.use('default')

def get_path(*parts):
    """Build path relative to script location"""
    return os.path.join(os.path.dirname(__file__), '..', *parts)

def load_config():
    """Load configuration from YAML file"""
    with open(get_path('config.yaml'), 'r') as f:
        return yaml.safe_load(f)

def verify_directories(config):
    """Verify all required directories exist"""
    for dir_name, dir_path in config['directories'].items():
        full_path = get_path(dir_path)
        if os.path.exists(full_path):
            print(f"{dir_path}:", os.listdir(full_path)[:5])
        else:
            print(f"Directory {dir_path} not found")

def load_geo_metadata(config):
    """Load and parse GEO series matrix file metadata"""
    matrix_filename = config['files']['geo']['series_matrix']['filename']
    if config['files']['geo']['series_matrix']['compressed']:
        matrix_filename += '.gz'
    matrix_file = get_path(config['directories']['geo'], matrix_filename)
    
    try:
        with gzip.open(matrix_file, 'rt') as f:
            metadata_lines = []
            for line in f:
                if line.startswith('!'):
                    metadata_lines.append(line)
                else:
                    break
                    
        sample_lines = [line for line in metadata_lines if line.startswith('!Sample_')]
        print(f"Found {len(sample_lines)} sample metadata lines")
        
        for line in sample_lines:
            if line.startswith('!Sample_geo_accession'):
                sample_ids = line.strip().split('\t')[1:]
                sample_ids = [s.strip('"') for s in sample_ids]
                print(f"Found {len(sample_ids)} samples")
                break
                
    except Exception as e:
        print(f"Error parsing metadata: {e}")
        sample_ids = None
        
    return sample_ids

def load_geo_expression(config):
    """Load GEO expression data from counts file"""
    counts_filename = config['files']['geo']['counts']['filename']
    if config['files']['geo']['counts']['compressed']:
        counts_filename += '.gz'
    counts_file = get_path(config['directories']['geo'], counts_filename)
    
    try:
        geo_expr_df = pd.read_csv(counts_file,
                                 compression='gzip' if config['files']['geo']['counts']['compressed'] else None,
                                 sep='\t',
                                 low_memory=False)
        
        # Set index and gene name columns
        geo_expr_df.set_index('Unnamed: 0', inplace=True)
        geo_expr_df.index.name = 'gene_id'
        geo_expr_df.rename(columns={'Unnamed: 1': 'gene_name'}, inplace=True)
        
        # Drop any unnamed columns that are all NaN
        unnamed_cols = [col for col in geo_expr_df.columns if col.startswith('Unnamed:')]
        geo_expr_df.drop(columns=unnamed_cols, inplace=True)
        
        if geo_expr_df.empty:
            raise ValueError("Empty expression matrix")
            
        return geo_expr_df
        
    except Exception as e:
        print(f"Error loading expression data: {e}")
        raise

def perform_differential_expression(geo_expr_df):
    """Perform differential expression analysis between cardiomyopathy and control samples"""
    sample_cols = [col for col in geo_expr_df.columns if col.startswith('G')]
    
    # First 26 samples are cardiomyopathy (13 ischemic + 13 dilated)
    cardio_cols = sample_cols[0:13] + sample_cols[13:26]  
    # Last 10 samples are healthy controls
    control_cols = sample_cols[26:]  
    
    if not cardio_cols or not control_cols:
        print("No sample groups found. Please check the data structure.")
        return None
    
    # Compute mean expression for each group
    mean_expr_cardio = geo_expr_df[cardio_cols].mean(axis=1)
    mean_expr_control = geo_expr_df[control_cols].mean(axis=1)
    
    # Calculate log2 fold change (adding a pseudocount to avoid log(0))
    log2_fc = np.log2(mean_expr_cardio + 1) - np.log2(mean_expr_control + 1)
    
    # Create a DataFrame with gene IDs and fold change
    de_results = pd.DataFrame({
        'Gene': geo_expr_df.iloc[:, 0],  # assuming first column is gene ID
        'Log2FC': log2_fc
    })
    
    # Select top 10 up-regulated genes in cardiomyopathy
    top_genes = de_results.sort_values('Log2FC', ascending=False).head(10)
    return top_genes

def load_gtex_data(config):
    """Load and process GTEx TPM data"""
    gtex_tpm_file = get_path(config['directories']['gtex'], 
                            config['files']['gtex']['tpm_data']['filename'])
    is_compressed = config['files']['gtex']['tpm_data']['compressed']
    
    try:
        open_func = gzip.open if is_compressed else open
        mode = 'rt' if is_compressed else 'r'  # text mode for gzip
        with open_func(gtex_tpm_file, mode) as f:
            gtex_df = pd.read_csv(f, sep='\t', skiprows=2)
        return gtex_df
    except Exception as e:
        print(f"Error reading GTEx file: {e}")
        raise

def process_encode_data(config):
    """Process ENCODE ChIP-seq data"""
    encode_dir = get_path(config['directories']['encode'])
    parquet_path = os.path.join(encode_dir, 'aggregated_chipseq.parquet')
    
    try:
        if os.path.exists(parquet_path):
            encode_df = pd.read_parquet(parquet_path)
        else:
            bed_files = glob(os.path.join(encode_dir, '*.bed.gz'))
            bed_dfs = []
            
            for file in bed_files:
                try:
                    df = pd.read_csv(file, sep='\t', header=None, 
                                   compression='gzip', comment='#')
                    df['source_file'] = os.path.basename(file)
                    bed_dfs.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            if bed_dfs:
                encode_df = pd.concat(bed_dfs, axis=0, ignore_index=True)
                encode_df.to_parquet(parquet_path, compression='snappy', 
                                   engine='pyarrow')
            else:
                raise ValueError("Failed to load any BED files")
                
        return encode_df
        
    except Exception as e:
        print(f"Error processing ENCODE data: {e}")
        raise

def analyze_peaks(encode_df):
    """Analyze ENCODE peak characteristics"""
    encode_df.columns = ['chr', 'start', 'end'] + list(encode_df.columns[3:])
    encode_df['peak_length'] = encode_df['end'] - encode_df['start']
    
    # Plot distribution of peak lengths
    plt.figure(figsize=(10, 5))
    plt.hist(encode_df['peak_length'], bins=50, color='lightgreen', edgecolor='black')
    plt.title('Distribution of ENCODE Peak Lengths')
    plt.xlabel('Peak Length (bp)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Count peaks per chromosome
    chr_counts = encode_df['chr'].value_counts()
    plt.figure(figsize=(12, 6))
    chr_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Peaks per Chromosome (ENCODE)')
    plt.xlabel('Chromosome')
    plt.ylabel('Peak Count')
    plt.show()
    
    print("Top 10 chromosomes by peak count:")
    print(chr_counts.head(10))

def extract_promoter_sequences(candidate_genes, genome_records, promoter_length=1000):
    """Extract promoter sequences for candidate genes"""
    chromosomes = [f"chr{i}" for i in list(range(1, 23)) + ['X', 'Y']]
    candidate_info = []
    
    for gene in candidate_genes:
        chrom = random.choice(chromosomes)
        tss = random.randint(1000000, 10000000)  # simulated TSS
        candidate_info.append({'gene': gene, 'chr': chrom, 'TSS': tss})
    
    candidate_df = pd.DataFrame(candidate_info)
    
    def extract_promoter(chrom, tss):
        rec = next((r for r in genome_records if r.id.startswith(chrom)), None)
        if rec is None:
            return None
        start = max(tss - promoter_length, 0)
        end = tss
        return str(rec.seq[start:end])
    
    candidate_df['promoter_seq'] = candidate_df.apply(
        lambda row: extract_promoter(row['chr'], row['TSS']), axis=1)
    
    return candidate_df

def analyze_sequences_with_dnabert(candidate_df):
    """Analyze promoter sequences using DNABERT-2"""
    model_name = "zhihan1996/DNABERT-2-117M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    
    def get_embedding(sequence):
        inputs = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[0, 0, :].numpy()
    
    embeddings = []
    valid_genes = []
    
    for idx, row in candidate_df.iterrows():
        seq = row['promoter_seq']
        if seq and len(seq) > 0:
            emb = get_embedding(seq)
            embeddings.append(emb)
            valid_genes.append(row['gene'])
    
    embeddings = np.array(embeddings)
    
    # Perform PCA for visualization
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c='red', alpha=0.7)
    for i, gene in enumerate(valid_genes):
        plt.annotate(gene, (embeddings_pca[i, 0], embeddings_pca[i, 1]), 
                    fontsize=8, alpha=0.75)
    plt.title('PCA of DNABERT-2 Promoter Embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
    
    return embeddings, valid_genes

def main():
    """Main function to run the complete analysis pipeline"""
    # Load configuration
    config = load_config()
    print("Configuration loaded successfully.")
    
    # Verify directory structure
    verify_directories(config)
    
    # Process GEO data
    sample_ids = load_geo_metadata(config)
    geo_expr_df = load_geo_expression(config)
    top_genes = perform_differential_expression(geo_expr_df)
    candidate_genes = top_genes['Gene'].tolist()
    
    # Process GTEx data
    gtex_df = load_gtex_data(config)
    
    # Process ENCODE data
    encode_df = process_encode_data(config)
    analyze_peaks(encode_df)
    
    # Extract and analyze sequences
    genome_records = []  # In practice, load from reference genome file
    candidate_df = extract_promoter_sequences(candidate_genes, genome_records)
    embeddings, valid_genes = analyze_sequences_with_dnabert(candidate_df)
    
    print("Analysis pipeline completed successfully.")

if __name__ == "__main__":
    main()