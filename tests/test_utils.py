"""
Test utilities for siRNA analysis pipeline
"""

import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Helper class for generating test data from actual project data"""
    
    def __init__(self, project_root=None):
        """
        Initialize test data generator
        
        Args:
            project_root: Path to project root directory. If None, assumes current directory
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / 'data'
        
    def _find_file(self, directory: Path, patterns: list) -> Path:
        """
        Find first file matching any of the given patterns
        
        Args:
            directory: Directory to search in
            patterns: List of glob patterns to match
            
        Returns:
            Path to matching file or None if not found
        """
        for pattern in patterns:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]
        return None
        
    def load_geo_sample(self, sample_size=100):
        """
        Load and sample GEO expression data
        
        Args:
            sample_size: Number of genes to sample
            
        Returns:
            pd.DataFrame: Sampled expression data
        """
        try:
            geo_dir = self.data_dir / 'geo'
            counts_file = self._find_file(geo_dir, ['*count_data.txt*'])
            
            if not counts_file:
                logger.warning("No GEO count data file found")
                return self._generate_mock_expression_data(sample_size)
            
            # Handle gzipped or plain text files
            open_func = gzip.open if counts_file.suffix == '.gz' else open
            
            with open_func(counts_file, 'rt') as f:
                # Read first chunk to get total rows
                chunk = pd.read_csv(f, sep='\t', nrows=1000)
                total_rows = len(chunk)
                
                # Generate random row indices
                sample_indices = random.sample(range(total_rows), min(sample_size, total_rows))
                
            # Read only sampled rows
            df = pd.read_csv(counts_file, sep='\t', skiprows=lambda x: x not in sample_indices)
            
            logger.info(f"Sampled {len(df)} genes from GEO data")
            return df
            
        except Exception as e:
            logger.error(f"Error loading GEO sample: {e}")
            return self._generate_mock_expression_data(sample_size)
    
    def load_genome_sample(self, num_sequences=5, seq_length=1000):
        """
        Load and sample genome sequences
        
        Args:
            num_sequences: Number of sequences to sample
            seq_length: Length of each sequence
            
        Returns:
            list: List of sampled SeqRecord objects
        """
        try:
            ref_dir = self.data_dir / 'reference'
            genome_file = self._find_file(ref_dir, ['*.fa.gz', '*.fa', '*.fasta.gz', '*.fasta'])
            
            if not genome_file:
                logger.warning("No genome reference file found")
                return self._generate_mock_genome_data(num_sequences, seq_length)
            
            sequences = []
            open_func = gzip.open if genome_file.suffix == '.gz' else open
            
            with open_func(genome_file, 'rt') as f:
                records = list(SeqIO.parse(f, 'fasta'))
                
                if records:
                    sampled_records = random.sample(records, min(num_sequences, len(records)))
                    
                    for record in sampled_records:
                        # Sample a random region of desired length
                        if len(record.seq) > seq_length:
                            start = random.randint(0, len(record.seq) - seq_length)
                            sub_seq = record.seq[start:start + seq_length]
                            sequences.append(
                                SeqRecord(sub_seq, id=f"{record.id}_region_{start}", 
                                        description=f"Sampled from {record.id}")
                            )
                            
            if sequences:
                logger.info(f"Sampled {len(sequences)} sequences from genome data")
                return sequences
            else:
                logger.warning("No valid sequences found in genome file")
                return self._generate_mock_genome_data(num_sequences, seq_length)
            
        except Exception as e:
            logger.error(f"Error loading genome sample: {e}")
            return self._generate_mock_genome_data(num_sequences, seq_length)
    
    def load_gtex_sample(self, sample_size=100):
        """
        Load and sample GTEx expression data
        
        Args:
            sample_size: Number of genes to sample
            
        Returns:
            pd.DataFrame: Sampled GTEx data
        """
        try:
            gtex_dir = self.data_dir / 'gtex'
            tpm_file = self._find_file(gtex_dir, ['*tpm.gct*'])
            
            if not tpm_file:
                logger.warning("No GTEx TPM file found")
                return self._generate_mock_gtex_data(sample_size)
            
            # Handle gzipped files
            open_func = gzip.open if tpm_file.suffix == '.gz' else open
            
            with open_func(tpm_file, 'rt') as f:
                # Skip header lines
                for _ in range(2):
                    next(f)
                
                chunk = pd.read_csv(f, sep='\t', nrows=1000)
                total_rows = len(chunk)
                
                # Generate random row indices
                sample_indices = random.sample(range(total_rows), min(sample_size, total_rows))
                
            # Read only sampled rows
            df = pd.read_csv(tpm_file, sep='\t', skiprows=lambda x: x not in sample_indices and x > 2)
            
            logger.info(f"Sampled {len(df)} genes from GTEx data")
            return df
            
        except Exception as e:
            logger.error(f"Error loading GTEx sample: {e}")
            return self._generate_mock_gtex_data(sample_size)
    
    def load_encode_peaks(self, sample_size=100):
        """
        Load and sample ENCODE ChIP-seq peak data
        
        Args:
            sample_size: Number of peaks to sample
            
        Returns:
            pd.DataFrame: Sampled peak data
        """
        try:
            encode_dir = self.data_dir / 'encode'
            bed_files = list(encode_dir.glob('*.bed.gz'))
            
            if not bed_files:
                logger.warning("No ENCODE BED files found")
                return self._generate_mock_peaks_data(sample_size)
            
            # Randomly select a bed file
            bed_file = random.choice(bed_files)
            
            with gzip.open(bed_file, 'rt') as f:
                # Read random sample of peaks
                peaks = pd.read_csv(f, sep='\t', header=None, 
                                 nrows=sample_size, skiprows=lambda x: random.random() > 0.1)
                
            logger.info(f"Sampled {len(peaks)} peaks from ENCODE data")
            return peaks
                
        except Exception as e:
            logger.error(f"Error loading ENCODE sample: {e}")
            return self._generate_mock_peaks_data(sample_size)
    
    # Fallback mock data generation methods
    def _generate_mock_expression_data(self, num_genes=100):
        """Generate mock expression data"""
        mock_df = pd.DataFrame({
            'gene_id': [f'ENSG{i:08d}' for i in range(num_genes)],
            'gene_name': [f'GENE{i}' for i in range(num_genes)]
        })
        logger.info("Generated mock expression data")
        return mock_df
    
    def _generate_mock_genome_data(self, num_sequences=5, seq_length=1000):
        """Generate mock genome data"""
        mock_records = [
            SeqRecord(
                Seq(''.join(np.random.choice(['A', 'T', 'G', 'C'], size=seq_length))),
                id=f'chr{i+1}',
                description=f'Mock chromosome {i+1}'
            )
            for i in range(num_sequences)
        ]
        logger.info("Generated mock genome data")
        return mock_records
    
    def _generate_mock_gtex_data(self, num_genes=100):
        """Generate mock GTEx data"""
        mock_df = pd.DataFrame(
            np.random.normal(50, 10, size=(num_genes, 20)),
            columns=[f'GTEX{i:08d}' for i in range(20)]
        )
        logger.info("Generated mock GTEx data")
        return mock_df
    
    def _generate_mock_peaks_data(self, num_peaks=100):
        """Generate mock ENCODE peak data"""
        mock_df = pd.DataFrame({
            'chr': [f'chr{random.randint(1, 22)}' for _ in range(num_peaks)],
            'start': [random.randint(1, 1000000) for _ in range(num_peaks)],
            'end': [random.randint(1000000, 2000000) for _ in range(num_peaks)]
        })
        logger.info("Generated mock ENCODE peak data")
        return mock_df