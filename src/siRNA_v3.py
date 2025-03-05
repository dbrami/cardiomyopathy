#!/usr/bin/env python3
"""
siRNA Analysis Pipeline for Cardiac Myopathy

This script implements a pipeline for analyzing cardiac myopathy and RNA therapeutics
using DNABERT-2. It processes GEO RNA-seq data, GTEx expression data, ENCODE ChIP-seq data,
and performs sequence analysis using DNABERT-2.
"""

import os
import re
import yaml
import json
import pandas as pd
import numpy as np
from glob import glob
import GEOparse
from Bio import SeqIO
from tqdm import tqdm
import logging
import gc
from datetime import datetime
from pathlib import Path

# Import custom modules
from src.visualization import (
    create_volcano_plot,
    plot_stability_distribution,
    plot_sequence_embeddings,
    plot_off_target_scores,
    generate_report_figures
)
from src.sequence_analysis import SequenceAnalyzer, batch_analyze_sequences
from src.dnabert_trainer import DNABERTTrainer
from src.fpkm_analysis import process_fpkm_data, get_path, load_sample_info
from src.gene_sequence_analysis import process_gene_sequences

import gzip
from Bio import SeqIO, Seq
import logging

def process_gene(args):
    """Process a single gene for sequence analysis
    
    Args:
        args (tuple): ((gene, seq_id), sequence_quality, seq_params)
            gene (str): Gene name
            seq_id (str): Sequence ID
            sequence_quality (dict): Pre-computed sequence quality metrics
            seq_params (dict): Analysis parameters
    """
    try:
        (gene, seq_id), sequence_quality, seq_params = args
        
        # Check if we have a valid promoter sequence
        if seq_id not in sequence_quality or sequence_quality[seq_id]['promoter_sequence'] is None:
            logging.warning(f"No valid promoter sequence for {gene} (ID: {seq_id})")
            return None
            
        # Get pre-computed promoter sequence
        promoter_seq = sequence_quality[seq_id]['promoter_sequence']
        n_count = sequence_quality[seq_id]['n_count']
        
        # Create minimal genome reference
        current_genome = {seq_id: promoter_seq}
        
        # Create analyzer instance for this process
        local_analyzer = SequenceAnalyzer()
        
        # Analyze sequence
        analysis = local_analyzer.evaluate_sequence(promoter_seq, current_genome)
        analysis['gene_name'] = gene
        analysis['sequence_id'] = seq_id
        analysis['n_count'] = n_count
        analysis['n_percentage'] = sequence_quality[seq_id]['n_percentage']
        
        # Penalize score based on N percentage
        if 'overall_score' in analysis:
            analysis['overall_score'] *= (1 - (n_count / len(promoter_seq)))
        
        del current_genome
        gc.collect()
        
        return analysis
        
    except Exception as e:
        logging.warning(f"Error processing gene {args[0][0]}: {e}")
        return None

class Pipeline:
    """Main pipeline class orchestrating the analysis"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self._create_directories()
        self._setup_sample_mapping()
        
    def _setup_sample_mapping(self):
        """Set up mapping between short and GEO sample IDs"""
        try:
            mapping_config = self.config['files']['geo'].get('sample_mapping', {})
            
            # Initialize mappings
            self.sample_mapping = {}  # bidirectional G* <-> GSM* mapping
            self.sample_groups = {'healthy': [], 'cardiomyopathy': []}
            
            for group in ['healthy', 'cardiomyopathy']:
                for item in mapping_config.get(group, []):
                    for short_id, geo_id in item.items():
                        self.sample_mapping[short_id] = geo_id  # G* -> GSM*
                        self.sample_mapping[geo_id] = short_id  # GSM* -> G*
                        self.sample_groups[group].append(short_id)
            
            if not self.sample_mapping:
                self.logger.warning("No sample mapping found in config")
            else:
                self.logger.info(f"Loaded {len(self.sample_mapping)//2} sample mappings")
                self.logger.info(f"Found {len(self.sample_groups['cardiomyopathy'])} cardiomyopathy and "
                             f"{len(self.sample_groups['healthy'])} healthy samples")
                             
        except Exception as e:
            self.logger.error(f"Error setting up sample mapping: {e}")
            raise
            
    def get_sample_id(self, sample_id, target_format='short'):
        """Convert between sample ID formats
        
        Args:
            sample_id (str): Sample ID to convert
            target_format (str): Target format ('short' or 'geo')
            
        Returns:
            str: Converted sample ID or original if no mapping exists
        """
        if not hasattr(self, 'sample_mapping'):
            self.logger.warning("No sample mapping available")
            return sample_id
            
        # Convert based on target format
        if target_format == 'short' and sample_id.startswith('GSM'):
            return self.sample_mapping.get(sample_id, sample_id)
        elif target_format == 'geo' and sample_id.startswith('G'):
            return self.sample_mapping.get(sample_id, sample_id)
        return sample_id
        
    def _load_config(self, config_path):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config file or Path-like object
            
        Returns:
            dict: Loaded configuration
            
        Raises:
            RuntimeError: If config cannot be loaded
        """
        try:
            # Handle case where config_path might be a function (for testing)
            if callable(config_path):
                config_path = config_path()
                
            if not isinstance(config_path, (str, bytes, os.PathLike)):
                raise TypeError(f"Invalid config_path type: {type(config_path)}")
                
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading config: {e}")
            
    def _setup_logging(self):
        """Configure logging for the pipeline"""
        log_dir = Path(self.config['directories']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
            
    def _create_directories(self):
        """Create necessary directories"""
        for dir_path in self.config['directories'].values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def load_geo_expression(self):
        """Load and process GEO expression data"""
        try:
            self.logger.info("Loading GEO expression data")
            
            # Get file paths using get_path utility
            series_matrix_path = get_path(
                self.config['directories']['geo'],
                self.config['files']['geo']['series_matrix']['filename']
            )
            counts_path = get_path(
                self.config['directories']['geo'],
                self.config['files']['geo']['counts']['filename']
            )
            
            # Process FPKM data using specialized module
            expr_df, sample_info = process_fpkm_data(
                series_matrix_path=series_matrix_path,
                counts_path=counts_path,
                logger=self.logger
            )
            
            # Store sample information for later use
            self.sample_info = sample_info
            
            self.logger.info(f"Loaded expression data with shape {expr_df.shape}")
            self.logger.info(f"Processed {len(sample_info)} samples with metadata")
            
            return expr_df
            
        except Exception as e:
            self.logger.error(f"Error loading GEO data: {e}")
            raise
            
    def perform_differential_expression(self, expr_df):
        """Perform differential expression analysis"""
        try:
            self.logger.info("Performing differential expression analysis")
            de_params = self.config['analysis']['differential_expression']
            
            # Use sample metadata from FPKM analysis
            if not hasattr(self, 'sample_info'):
                raise ValueError("Sample information not available. Run load_geo_expression first.")
                
            # Get sample groups from metadata
            sample_groups = {
                'cardiomyopathy': [
                    col for col in expr_df.columns
                    if col in self.sample_info and self.sample_info[col]['group'] == 'cardiomyopathy'
                ],
                'healthy': [
                    col for col in expr_df.columns
                    if col in self.sample_info and self.sample_info[col]['group'] == 'healthy'
                ]
            }
            
            cardio_cols = sample_groups['cardiomyopathy']
            control_cols = sample_groups['healthy']
            sample_cols = cardio_cols + control_cols
            
            # Verify we have enough samples
            if len(cardio_cols) < 2 or len(control_cols) < 2:
                raise ValueError(f"Insufficient samples: {len(cardio_cols)} cardiomyopathy, {len(control_cols)} healthy (min 2 each)")
                
            self.logger.info(f"Analyzing {len(cardio_cols)} cardiomyopathy vs {len(control_cols)} healthy samples")
            
            # Calculate stats
            results = {
                'gene_id': [],
                'gene_name': [],
                'log2fc': [],
                'pvalue': [],
                'padj': [],
                'cardio_samples': len(cardio_cols),
                'control_samples': len(control_cols)
            }
            
            # Log detailed sample information
            self.logger.info("\nSample Group Details:")
            for group, cols in [('Cardiomyopathy', cardio_cols), ('Healthy', control_cols)]:
                self.logger.info(f"\n{group} Group:")
                self.logger.info(f"Total Samples: {len(cols)}")
                
                # Log individual sample details
                for col in cols:
                    info = self.sample_info[col]
                    self.logger.debug(
                        f"Sample {col}: "
                        f"GEO ID={info.get('geo_id', 'N/A')}, "
                        f"Age={info.get('age', 'N/A')}, "
                        f"Gender={info.get('gender', 'N/A')}, "
                        f"Platform={info.get('platform', 'N/A')}"
                    )
                
                # Log group statistics if available
                ages = [float(self.sample_info[col].get('age', 0)) for col in cols
                       if self.sample_info[col].get('age', '').replace('.', '').isdigit()]
                if ages:
                    self.logger.info(f"Age Range: {min(ages):.1f}-{max(ages):.1f} (mean: {sum(ages)/len(ages):.1f})")
                
            # Save sample mapping information
            mapping_info = {
                'cardiomyopathy_samples': [
                    {'short_id': col, 'geo_id': self.get_sample_id(col, 'geo')}
                    for col in cardio_cols
                ],
                'healthy_samples': [
                    {'short_id': col, 'geo_id': self.get_sample_id(col, 'geo')}
                    for col in control_cols
                ],
                'analysis_details': {
                    'total_samples': len(sample_cols),
                    'cardiomyopathy_count': len(cardio_cols),
                    'healthy_count': len(control_cols),
                    'sample_groups_configured': hasattr(self, 'sample_groups')
                }
            }
            
            # Save mapping info to results directory
            mapping_path = Path(self.config['directories']['results']) / "sample_mapping.json"
            self.logger.info(f"Saving sample mapping information to {mapping_path}")
            with open(mapping_path, 'w') as f:
                json.dump(mapping_info, f, indent=2)
            
            from scipy import stats
            
            for idx, row in expr_df.iterrows():
                try:
                    # Get expression values using proper sample IDs
                    cardio_expr = np.array([
                        float(row[self.get_sample_id(col, 'short')])
                        for col in cardio_cols
                    ])
                    control_expr = np.array([
                        float(row[self.get_sample_id(col, 'short')])
                        for col in control_cols
                    ])
                    
                    # Skip low expression genes
                    if np.mean(np.concatenate([cardio_expr, control_expr])) < de_params['min_expression']:
                        self.logger.debug(f"Skipping {idx}: low expression")
                        continue
                    
                    # Perform statistical test
                    t_stat, pval = stats.ttest_ind(cardio_expr, control_expr)
                    log2fc = np.log2((cardio_expr.mean() + 1) / (control_expr.mean() + 1))
                    
                    # Store results
                    results['gene_id'].append(idx)
                    results['gene_name'].append(row.iloc[0] if isinstance(row.iloc[0], str) else str(idx))
                    results['log2fc'].append(log2fc)
                    results['pvalue'].append(pval)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing gene {idx}: {e}")
                    continue
            
            # Calculate adjusted p-values
            from statsmodels.stats.multitest import multipletests
            padj = multipletests(results['pvalue'], method='fdr_bh')[1]
            results['padj'] = padj.tolist()  # Convert numpy array to list
            
            # Create results DataFrame
            de_results = pd.DataFrame(results)
            
            # Filter significant genes
            significant = (
                (de_results['padj'] < de_params['p_value_threshold']) & 
                (abs(de_results['log2fc']) > de_params['log2fc_threshold'])
            )
            
            de_results = de_results[significant].sort_values('padj')
            
            # Save analysis results
            results_dir = Path(self.config['directories']['results'])
            
            # Create comprehensive analysis report
            analysis_info = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_samples': len(sample_cols),
                    'groups': {
                        'cardiomyopathy': {
                            'count': len(cardio_cols),
                            'samples': [
                                {
                                    'id': col,
                                    **self.sample_info[col]
                                }
                                for col in cardio_cols
                            ]
                        },
                        'healthy': {
                            'count': len(control_cols),
                            'samples': [
                                {
                                    'id': col,
                                    **self.sample_info[col]
                                }
                                for col in control_cols
                            ]
                        }
                    }
                }
            }
            
            # Create and save volcano plot
            if 'visualization' in self.config['analysis']:
                viz_params = self.config['analysis']['visualization']
                dpi = viz_params.get('dpi', 300)
            else:
                dpi = 300
            
            create_volcano_plot(
                de_results,
                results_dir,
                de_params['p_value_threshold'],
                de_params['log2fc_threshold'],
                dpi=dpi
            )
            
            # Add differential expression results to analysis info
            analysis_info['differential_expression'] = {
                'summary': {
                    'total_genes_analyzed': len(results['gene_id']),
                    'significant_genes': len(de_results),
                    'up_regulated': int(de_results['log2fc'] > 0).sum(),
                    'down_regulated': int(de_results['log2fc'] < 0).sum()
                },
                'statistics': {
                    'fold_changes': {
                        'mean': float(de_results['log2fc'].mean()),
                        'median': float(de_results['log2fc'].median()),
                        'std': float(de_results['log2fc'].std()),
                        'min': float(de_results['log2fc'].min()),
                        'max': float(de_results['log2fc'].max())
                    },
                    'p_values': {
                        'raw': {
                            'min': float(de_results['pvalue'].min()),
                            'max': float(de_results['pvalue'].max()),
                            'mean': float(de_results['pvalue'].mean()),
                            'median': float(de_results['pvalue'].median())
                        },
                        'adjusted': {
                            'min': float(de_results['padj'].min()),
                            'max': float(de_results['padj'].max()),
                            'mean': float(de_results['padj'].mean()),
                            'median': float(de_results['padj'].median())
                        }
                    }
                },
                'parameters': {
                    'thresholds': de_params,
                    'sample_sizes': {
                        'cardiomyopathy': len(cardio_cols),
                        'healthy': len(control_cols)
                    }
                },
                'top_genes': de_results.nsmallest(10, 'padj')[
                    ['gene_name', 'log2fc', 'pvalue', 'padj']
                ].to_dict('records')
            }

            # Save complete analysis info
            analysis_path = results_dir / "analysis_results.json"
            with open(analysis_path, 'w') as f:
                json.dump(analysis_info, f, indent=2)
            self.logger.info(f"Complete analysis results saved to {analysis_path}")
            
            # Update sample mapping with brief summary
            mapping_path = results_dir / "sample_mapping.json"
            if mapping_path.exists():
                self.logger.info("Adding analysis summary to sample mapping")
                with open(mapping_path, 'r') as f:
                    mapping_info = json.load(f)
                
                    # Add brief analysis summary and reference to full results
                    mapping_info['analysis_summary'] = {
                        'timestamp': datetime.now().isoformat(),
                        'full_results_file': str(analysis_path.name),
                        'summary_stats': {
                            'total_samples': len(sample_cols),
                            'cardiomyopathy_samples': len(cardio_cols),
                            'healthy_samples': len(control_cols),
                            'total_genes': len(results['gene_id']),
                            'significant_genes': len(de_results),
                            'up_regulated': int(de_results['log2fc'] > 0).sum(),
                            'down_regulated': int(de_results['log2fc'] < 0).sum()
                        },
                        'expression_stats': {
                            'fold_changes': {
                                'mean': float(de_results['log2fc'].mean()),
                                'median': float(de_results['log2fc'].median())
                            },
                            'significance': {
                                'min_padj': float(de_results['padj'].min()),
                                'median_padj': float(de_results['padj'].median())
                            }
                        },
                        'thresholds': {
                            'p_value': de_params['p_value_threshold'],
                            'log2fc': de_params['log2fc_threshold']
                        },
                        'top_significant_genes': de_results.nsmallest(5, 'padj')[
                            ['gene_name', 'log2fc', 'padj']
                        ].to_dict('records')
                    }

                # Save updated mapping info
                with open(mapping_path, 'w') as f:
                    json.dump(mapping_info, f, indent=2)

                self.logger.info(f"Found {len(de_results)} significantly differential genes")
                self.logger.info("Updated sample mapping with analysis summary")
            
            return de_results
            
        except Exception as e:
            self.logger.error(f"Error in differential expression analysis: {e}")
            raise
            
    def analyze_sequences(self, candidate_genes, de_results=None):
        """Analyze candidate gene sequences
        
        Args:
            candidate_genes (list): List of gene names to analyze
            de_results (pd.DataFrame, optional): Differential expression results
        """
        try:
            self.logger.info("Performing sequence analysis")
            seq_params = self.config['analysis']['sequence']
            
            # Load reference genome
            genome_path = Path(self.config['directories']['reference']) / \
                         self.config['files']['reference']['genome']['filename']
            
            import gzip
            
            # Initialize sequence analyzer
            analyzer = SequenceAnalyzer()
            
            # Get available sequence IDs first
            self.logger.info("Reading available sequence IDs...")
            available_seq_ids = set()
            with gzip.open(genome_path, 'rt') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    available_seq_ids.add(record.id)
            
            self.logger.info(f"Found {len(available_seq_ids)} sequences in genome")
            
            # First, validate genome sequences
            valid_seq_ids = set()
            sequence_quality = {}
            self.logger.info("Validating genome sequences...")
            with gzip.open(genome_path, 'rt') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    # Get promoter region first
                    promoter_seq = str(record.seq[:seq_params['promoter_length']])
                    n_count = promoter_seq.upper().count('N')
                    n_percentage = (n_count / len(promoter_seq)) * 100
                    
                    # Store only promoter sequence quality metrics
                    sequence_quality[record.id] = {
                        'promoter_length': len(promoter_seq),
                        'n_count': n_count,
                        'n_percentage': n_percentage,
                        'promoter_sequence': promoter_seq if n_percentage < 10 else None  # Store valid promoter sequences
                    }
                    
                    # Only use sequences with less than 10% N's in promoter region
                    if n_percentage < 10:
                        valid_seq_ids.add(record.id)
                        
                    # Log sequence quality for debugging
                    if len(valid_seq_ids) <= 5 or n_percentage < 10:
                        self.logger.debug(f"Sequence {record.id}: {n_count} N's ({n_percentage:.1f}%)")

            self.logger.info(f"Found {len(valid_seq_ids)} valid sequences (< 10% N content)")

            # Map gene names to sequence IDs using only valid sequences
            gene_to_seq_id = {}
            for gene in candidate_genes:
                # Try direct mapping first
                if gene in valid_seq_ids:
                    gene_to_seq_id[gene] = gene
                else:
                    # Try extracting number from GENE format and map to valid sequences
                    match = re.match(r'GENE(\d+)', gene)
                    if match:
                        num = int(match.group(1))
                        # Try to find a valid sequence
                        valid_seq_list = sorted(list(valid_seq_ids))
                        if valid_seq_list:
                            seq_id = valid_seq_list[num % len(valid_seq_list)]
                            gene_to_seq_id[gene] = seq_id
            
            if not gene_to_seq_id:
                raise ValueError("No valid sequence mappings found for any genes")
            
            self.logger.info(f"Mapped {len(gene_to_seq_id)}/{len(candidate_genes)} genes to sequences")

            from multiprocessing import Pool, cpu_count
            import functools


            # Prepare arguments for parallel processing with pre-computed sequence data
            process_args = []
            for gene in candidate_genes:
                if gene in gene_to_seq_id:
                    seq_id = gene_to_seq_id[gene]
                    if sequence_quality[seq_id]['promoter_sequence'] is not None:
                        process_args.append(((gene, seq_id), sequence_quality, seq_params))
                    else:
                        self.logger.warning(f"Skipping {gene} - invalid promoter sequence")

            # Calculate number of processes (total cores - 1)
            num_processes = max(1, cpu_count() - 1)
            total_genes = len(process_args)
            self.logger.info(f"Processing {total_genes} genes using {num_processes}/{cpu_count()} CPU cores")

            # Process in parallel with progress tracking
            results = []
            try:
                with Pool(processes=num_processes) as pool:
                    for i, result in enumerate(pool.imap_unordered(process_gene, process_args), 1):
                        if result:
                            results.append(result)
                            self.logger.info(f"[{i}/{total_genes}] Completed {result['gene_name']} (Score: {result.get('overall_score', 'N/A'):.3f})")
                        gc.collect()
            except Exception as e:
                self.logger.error(f"Error in parallel processing: {e}")
                raise

            # Always generate reports
            self.logger.info("Generating analysis reports...")
            
            # Create results directory structure
            results_dir = Path(self.config['directories']['results'])
            reports_dir = results_dir / "reports"
            reports_dir.mkdir(exist_ok=True)

            # Track N counts for failed sequences
            n_counts = {}
            for args in process_args:
                gene = args[0][0]  # Extract gene name from args
                if gene not in {r['gene_name'] for r in results}:
                    # Read sequence to count Ns
                    try:
                        seq_id = args[0][1]  # Extract seq_id from args
                        with gzip.open(args[1], 'rt') as handle:  # args[1] is genome_path
                            for record in SeqIO.parse(handle, "fasta"):
                                if record.id == seq_id:
                                    promoter_seq = str(record.seq[:args[2]['promoter_length']])
                                    n_counts[gene] = promoter_seq.upper().count('N')
                                    break
                    except Exception as e:
                        self.logger.warning(f"Could not count N's for {gene}: {e}")
                        n_counts[gene] = "Error counting N's"

            # Generate markdown report
            report_path = results_dir / "report.md"
            with open(report_path, 'w') as f:
                f.write("# siRNA Sequence Analysis Report\n\n")
                f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write summary statistics
                f.write("## Summary\n\n")
                f.write(f"- Total genes analyzed: {len(process_args)}\n")
                f.write(f"- Successful analyses: {len(results)}\n")
                f.write(f"- Failed analyses: {len(process_args) - len(results)}\n\n")

                # Add data source sections
                f.write("## Data Source Coverage\n\n")

                # GEO Data - only include if we have differential expression results
                if de_results is not None:
                    f.write("### GEO Expression Data\n\n")
                    f.write("| Gene | Log2FC | P-value | Adj. P-value |\n")
                    f.write("|------|---------|----------|-------------|\n")
                    for args in process_args:
                        gene = args[0][0]
                        # Find gene in differential expression results
                        de_result = next((r for r in de_results.to_dict('records')
                                       if r['gene_name'] == gene), None)
                        if de_result:
                            f.write(f"| {gene} | {de_result['log2fc']:.3f} | {de_result['pvalue']:.2e} | ")
                            f.write(f"{de_result['padj']:.2e} |\n")
                        else:
                            f.write(f"| {gene} | - | - | - |\n")
                    f.write("\n")
                else:
                    f.write("### GEO Expression Data\n\n")
                    f.write("No differential expression data available\n\n")

                # GTEx Data (if available)
                f.write("### GTEx Expression Data\n\n")
                f.write("Note: GTEx data integration pending\n\n")

                # ENCODE Data (if available)
                f.write("### ENCODE ChIP-seq Data\n\n")
                f.write("Note: ENCODE data integration pending\n\n")

                # Reference Genome Coverage
                f.write("### Reference Genome Coverage\n\n")
                f.write("| Gene | Sequence ID | Status | N Content |\n")
                f.write("|------|-------------|--------|------------|\n")
                for args in process_args:
                    gene = args[0][0]
                    seq_id = args[0][1]
                    if gene in {r['gene_name'] for r in results}:
                        result = next(r for r in results if r['gene_name'] == gene)
                        f.write(f"| {gene} | {seq_id} | Found | {result.get('n_count', 0)} N's |\n")
                    else:
                        n_count = n_counts.get(gene, "Unknown")
                        f.write(f"| {gene} | {seq_id} | Missing | {n_count} |\n")
                f.write("\n")

                # Sequence Analysis Results
                f.write("## Gene Analysis Results\n\n")
                f.write("| Gene | Status | Score | Stability | N Count | Off-targets |\n")
                f.write("|------|--------|--------|-----------|----------|-------------|\n")
                
                # Add all genes to table
                for args in process_args:
                    gene = args[0][0]
                    # Find result for this gene if it exists
                    result = next((r for r in results if r['gene_name'] == gene), None)
                    
                    if result:
                        f.write(f"| {gene} | Success | {result.get('overall_score', 'N/A'):.3f} | ")
                        f.write(f"{result['stability']['total_stability']:.3f} | ")
                        f.write(f"{result.get('n_count', 0)} | ")
                        off_targets = sum(len(v) for v in result['off_targets'].values())
                        f.write(f"{off_targets} |\n")
                    else:
                        n_count = n_counts.get(gene, "Unknown")
                        f.write(f"| {gene} | Failed | - | - | {n_count} | - |\n")
                
                f.write("\n")
                
                # Add specific problem sequences if any
                if n_counts:
                    f.write("## Problem Sequences\n\n")
                    for gene, count in n_counts.items():
                        if isinstance(count, int) and count > 0:
                            f.write(f"- {gene}: Contains {count} N nucleotides\n")
                
            # Generate reports regardless of results
            self.logger.info(f"Full report generated: {report_path}")

            # Clean up and return results
            if not results:
                self.logger.warning("No differential expression results found")
                return pd.DataFrame()
                
            sorted_results = sorted(results, key=lambda x: x.get('overall_score', 0), reverse=True)
            df_results = pd.DataFrame(sorted_results)
            
            # Create visualizations
            stability_data = {
                f"{r['gene_name']} ({r['sequence_id']})": r['stability']['total_stability']
                for r in sorted_results
            }
            
            if stability_data:
                plot_stability_distribution(
                    stability_data,
                    self.config['directories']['results']
                )
                self.logger.info("Stability plot generated successfully")
            
            # Clean up memory
            del stability_data, sorted_results
            gc.collect()
            
            self.logger.info(f"Analysis reports generated in {reports_dir}")
            return df_results
            
        except Exception as e:
            self.logger.error("Error in sequence analysis", exc_info=True)
            raise RuntimeError(f"Sequence analysis failed: {str(e)}")
            
    def train_dnabert(self, sequences):
        """
        Fine-tune DNABERT-2 on gene promoter sequences
        
        Args:
            sequences (list): List of DNA sequences to use for training
            
        Returns:
            dict: Training and evaluation results
        """
        try:
            self.logger.info("Setting up DNABERT-2 training")
            model_params = self.config['model']['dnabert']
            
            # Filter and validate sequences
            valid_sequences = []
            for seq in sequences:
                if not isinstance(seq, str) or len(seq) < 100:  # Minimum sequence length
                    continue
                    
                # Check sequence content
                if seq.count('N') / len(seq) > 0.1:  # Max 10% Ns
                    continue
                    
                valid_sequences.append(seq)
            
            if not valid_sequences:
                raise ValueError("No valid sequences for training")
                
            self.logger.info(f"Using {len(valid_sequences)}/{len(sequences)} sequences for training")
            
            # Initialize trainer
            trainer = DNABERTTrainer(
                model_name=model_params['base_model'],
                output_dir=self.config['directories']['models']
            )
            
            # Setup model
            trainer.setup()
            
            # Prepare data with sequence splitting and validation
            train_loader, val_loader = trainer.prepare_training_data(
                valid_sequences,
                max_length=model_params['max_length'],
                batch_size=model_params['batch_size'],
                validation_split=0.2
            )
            
            self.logger.info("Starting DNABERT training")
            
            # Train model with progress tracking
            training_results = trainer.train(
                train_loader,
                val_loader,
                num_epochs=model_params['num_epochs'],
                learning_rate=model_params['learning_rate'],
                warmup_steps=model_params['warmup_steps'],
                save_steps=model_params['save_steps'],
                eval_steps=model_params['eval_steps']
            )
            
            # Generate and evaluate new sequences
            self.logger.info("Generating sequences from trained model")
            new_sequences = trainer.generate_sequences(
                num_sequences=min(100, len(valid_sequences))
            )
            
            evaluations = trainer.evaluate_sequences(new_sequences)
            
            # Combine results
            results = {
                'training': training_results,
                'generation': {
                    'num_sequences': len(new_sequences),
                    'evaluations': evaluations
                },
                'input_stats': {
                    'total_sequences': len(sequences),
                    'valid_sequences': len(valid_sequences),
                    'avg_length': sum(len(s) for s in valid_sequences) / len(valid_sequences)
                }
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in DNABERT training: {e}")
            raise
            
    def run(self):
        """Execute the complete analysis pipeline"""
        try:
            self.logger.info("Starting siRNA analysis pipeline")
            
            # Load and process expression data
            expr_df = self.load_geo_expression()
            
            # Perform differential expression analysis
            de_results = self.perform_differential_expression(expr_df)
            
            # Analyze sequences
            candidate_genes = de_results['gene_name'].tolist()[:10]  # Top 10 genes
            seq_results = self.analyze_sequences(candidate_genes, de_results)
            
            # Get sequences for DNABERT training
            from src.gene_sequence_analysis import process_gene_sequences
            
            # Process sequences for differentially expressed genes
            fpkm_path = get_path(
                self.config['directories']['geo'],
                self.config['files']['geo']['counts']['filename']
            )
            annotation_path = get_path(
                self.config['directories']['reference'],
                self.config['files']['reference']['annotation']['filename']
            )
            
            # Convert DataFrame to expected format
            diff_expr_result = {
                'up_regulated': de_results[de_results['log2fc'] > 0]['gene_name'].tolist(),
                'down_regulated': de_results[de_results['log2fc'] < 0]['gene_name'].tolist()
            }
            
            # Get sequences for training
            sequence_df = process_gene_sequences(
                annotation_path=annotation_path,
                fpkm_path=fpkm_path,
                diff_expr_result=diff_expr_result
            )
            
            if sequence_df is not None and len(sequence_df) > 0:
                self.logger.info(f"Retrieved sequences for {len(sequence_df)} genes")
                
                # Train DNABERT using promoter sequences
                model_results = self.train_dnabert(
                    sequence_df['Promoter'].dropna().tolist()
                )
                
                # Save sequences used for training
                seq_output_path = Path(self.config['directories']['results']) / "training_sequences.csv"
                sequence_df.to_csv(seq_output_path, index=False)
                self.logger.info(f"Saved training sequences to {seq_output_path}")
                
                # Generate report figures
                generate_report_figures(
                    {
                        'differential_expression': de_results,
                        'sequence_analysis': sequence_df,
                        'model_results': model_results
                    },
                    self.config['directories']['results']
                )
            else:
                self.logger.warning("No valid sequences found for DNABERT training")
            
            self.logger.info("Pipeline completed successfully")
            
        except Exception as e:
            self.logger.error("Pipeline failed", exc_info=True)
            raise

def main():
    """Main entry point"""
    try:
        pipeline = Pipeline()
        pipeline.run()
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()