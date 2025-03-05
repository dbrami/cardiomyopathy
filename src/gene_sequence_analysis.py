import pandas as pd
import gzip
import requests
import json
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction
import io

def fetch_ensembl_id_from_ncbi(ncbi_id, species='homo_sapiens'):
    """
    Map NCBI GeneID to Ensembl Gene ID using Ensembl REST API.
    
    Parameters:
    ncbi_id (str): NCBI GeneID
    species (str): Species name (default: 'homo_sapiens')
    
    Returns:
    str: Ensembl Gene ID or None if not found
    """
    server = "https://rest.ensembl.org"
    ext = f"/xrefs/symbol/{species}/{ncbi_id}?external_db=EntrezGene"
    
    try:
        r = requests.get(server + ext, headers={"Content-Type": "application/json"})
        if r.ok:
            data = r.json()
            if data and len(data) > 0:
                return data[0]['id']  # Return the first Ensembl ID
        return None
    except Exception as e:
        print(f"Error fetching Ensembl ID for NCBI {ncbi_id}: {e}")
        return None

def fetch_gene_sequence(ensembl_id):
    """
    Fetch gene sequence and promoter region from Ensembl REST API.
    
    Parameters:
    ensembl_id (str): Ensembl Gene ID
    
    Returns:
    dict: Contains sequence, promoter sequence, and gene info
    """
    server = "https://rest.ensembl.org"
    
    # Fetch sequence (includes exons, introns, and some flanking regions)
    ext_seq = f"/sequence/id/{ensembl_id}?type=genomic"
    try:
        r = requests.get(server + ext_seq, headers={"Content-Type": "text/plain"})
        if r.ok:
            sequence = r.text
        else:
            sequence = None
    except Exception as e:
        print(f"Error fetching sequence for {ensembl_id}: {e}")
        sequence = None
    
    # Fetch regulatory regions (promoter approximated as 2000bp upstream)
    ext_region = f"/sequence/id/{ensembl_id}?type=genomic;expand_5prime=2000"
    try:
        r = requests.get(server + ext_region, headers={"Content-Type": "text/plain"})
        if r.ok:
            promoter_seq = r.text[:2000]  # Take first 2000 bp as promoter approximation
        else:
            promoter_seq = None
    except Exception as e:
        print(f"Error fetching promoter for {ensembl_id}: {e}")
        promoter_seq = None
    
    return {
        'sequence': sequence,
        'promoter': promoter_seq
    }

def calculate_sequence_stats(sequence):
    """
    Calculate GC percentage and count of Ns in a sequence.
    
    Parameters:
    sequence (str): DNA sequence
    
    Returns:
    tuple: (GC percentage, N count)
    """
    if sequence and isinstance(sequence, str):
        seq = Seq(sequence)
        gc_percent = gc_fraction(seq) * 100  # Convert to percentage
        n_count = sequence.upper().count('N')
        return gc_percent, n_count
    return None, None

def process_gene_sequences(annotation_path, fpkm_path, diff_expr_result):
    """
    Process differentially expressed genes to retrieve sequences and stats.
    
    Parameters:
    annotation_path (str): Path to compressed annotation file (not used here, assumed integrated in FPKM)
    fpkm_path (str): Path to compressed FPKM file
    diff_expr_result (dict): Result from load_and_analyze_fpkm with gene IDs
    
    Returns:
    pd.DataFrame: DataFrame with gene info and sequence stats
    """
    # Load FPKM data to get gene IDs (assuming GeneID is the index)
    try:
        with gzip.open(fpkm_path, 'rt') as f:
            fpkm_df = pd.read_csv(f, sep='\t', index_col=0)
    except Exception as e:
        print(f"Error loading FPKM file: {e}")
        return None

    # Get differentially expressed genes
    de_genes = diff_expr_result['up_regulated'] + diff_expr_result['down_regulated']
    
    # Prepare result storage
    results = {
        'GeneID': [],
        'EnsemblID': [],
        'Sequence': [],
        'Promoter': [],
        'GC_Percent': [],
        'N_Count': [],
        'Regulation': []
    }

    for gene_id in de_genes:
        if str(gene_id) not in fpkm_df.index:
            continue  # Skip if gene not in FPKM data
        
        # Map NCBI GeneID to Ensembl ID
        ensembl_id = fetch_ensembl_id_from_ncbi(str(gene_id))
        if not ensembl_id:
            print(f"No Ensembl ID found for GeneID {gene_id}")
            continue
        
        # Fetch sequence and promoter
        seq_data = fetch_gene_sequence(ensembl_id)
        sequence = seq_data['sequence']
        promoter = seq_data['promoter']
        
        # Calculate sequence stats
        gc_percent, n_count = calculate_sequence_stats(sequence)
        
        # Determine regulation status
        regulation = 'Up' if gene_id in diff_expr_result['up_regulated'] else 'Down'
        
        # Store results
        results['GeneID'].append(gene_id)
        results['EnsemblID'].append(ensembl_id)
        results['Sequence'].append(sequence)
        results['Promoter'].append(promoter)
        results['GC_Percent'].append(gc_percent)
        results['N_Count'].append(n_count)
        results['Regulation'].append(regulation)
        
        print(f"Processed {gene_id} -> {ensembl_id}")

    # Create DataFrame
    result_df = pd.DataFrame(results)
    return result_df

# Example usage within the module (for testing purposes)
if __name__ == "__main__":
    from fpkm_analysis import load_and_analyze_fpkm
    
    fpkm_path = "../data/geo/GSE55296_norm_counts_FPKM_GRCh38.p13_NCBI.tsv.gz"
    readme_path = "../data/geo/GSE55296_processed_data_readme.txt"
    annotation_path = "../data/geo/Human.GRCh38.p13.annot.tsv.gz"
    
    # Get differentially expressed genes
    diff_expr_result = load_and_analyze_fpkm(fpkm_path, readme_path)
    if diff_expr_result:
        # Process sequences
        sequence_df = process_gene_sequences(annotation_path, fpkm_path, diff_expr_result)
        if sequence_df is not None:
            print(sequence_df.head())
            # Optionally save to file
            sequence_df.to_csv("gene_sequences_for_dnabert.csv", index=False)