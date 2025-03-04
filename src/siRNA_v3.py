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

class Pipeline:
    """Main pipeline class orchestrating the analysis"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self._create_directories()
        
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
            counts_config = self.config['files']['geo']['counts']
            counts_path = Path(self.config['directories']['geo']) / counts_config['filename']
            
            if counts_config['compressed']:
                counts_path = counts_path.with_suffix(counts_path.suffix + '.gz')
            
            expr_df = pd.read_csv(counts_path, sep='\t', compression='gzip' 
                                if counts_config['compressed'] else None)
            
            # Process dataframe
            if 'gene_id' in expr_df.columns:
                gene_id_col = expr_df['gene_id']
                expr_df = expr_df.drop('gene_id', axis=1)
                expr_df.index = gene_id_col
                expr_df.index.name = 'gene_id'
            elif 'Unnamed: 0' in expr_df.columns:
                expr_df = expr_df.set_index('Unnamed: 0')
                expr_df.index.name = 'gene_id'
            else:
                raise ValueError("Missing gene_id column in expression data")

            # Ensure gene_name is first column if present
            if 'gene_name' in expr_df.columns:
                cols = ['gene_name'] + [col for col in expr_df.columns if col != 'gene_name']
                expr_df = expr_df[cols]
            
            self.logger.info(f"Loaded expression data with shape {expr_df.shape}")
            return expr_df
            
        except Exception as e:
            self.logger.error(f"Error loading GEO data: {e}")
            raise
            
    def perform_differential_expression(self, expr_df):
        """Perform differential expression analysis"""
        try:
            self.logger.info("Performing differential expression analysis")
            de_params = self.config['analysis']['differential_expression']
            
            # Get sample columns
            sample_cols = [col for col in expr_df.columns if col.startswith('G')]
            cardio_cols = sample_cols[:26]  # First 26 are cardiomyopathy
            control_cols = sample_cols[26:]  # Rest are controls
            
            # Calculate stats
            results = {
                'gene_id': [],
                'gene_name': [],
                'log2fc': [],
                'pvalue': [],
                'padj': []
            }
            
            for idx, row in expr_df.iterrows():
                # Skip low expression genes
                if row[sample_cols].mean() < de_params['min_expression']:
                    continue
                    
                cardio_expr = row[cardio_cols].astype(float)
                control_expr = row[control_cols].astype(float)
                
                from scipy import stats
                t_stat, pval = stats.ttest_ind(cardio_expr, control_expr)
                log2fc = np.log2((cardio_expr.mean() + 1) / (control_expr.mean() + 1))
                
                results['gene_id'].append(idx)
                results['gene_name'].append(row.iloc[0])
                results['log2fc'].append(log2fc)
                results['pvalue'].append(pval)
            
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
            
            # Create volcano plot
            if 'visualization' in self.config['analysis']:
                viz_params = self.config['analysis']['visualization']
                dpi = viz_params.get('dpi', 300)
            else:
                dpi = 300
                
            create_volcano_plot(
                de_results,
                self.config['directories']['results'],
                de_params['p_value_threshold'],
                de_params['log2fc_threshold'],
                dpi=dpi
            )
            
            self.logger.info(f"Found {len(de_results)} significantly differential genes")
            return de_results
            
        except Exception as e:
            self.logger.error(f"Error in differential expression analysis: {e}")
            raise
            
    def analyze_sequences(self, candidate_genes):
        """Analyze candidate gene sequences"""
        try:
            self.logger.info("Performing sequence analysis")
            seq_params = self.config['analysis']['sequence']
            
            # Load reference genome
            genome_path = Path(self.config['directories']['reference']) / \
                         self.config['files']['reference']['genome']['filename']
            
            import gzip
            
            # Initialize sequence analyzer
            analyzer = SequenceAnalyzer()
            
            # Create index of genome sequence locations for faster access
            self.logger.info("Creating genome sequence index...")
            sequence_index = {}
            with gzip.open(genome_path, 'rt') as handle:
                start_pos = handle.tell()
                for record in SeqIO.parse(handle, "fasta"):
                    sequence_index[record.id] = start_pos
                    start_pos = handle.tell()

            # Map between actual gene names and sequence IDs
            # In a real implementation, this would use proper gene annotations
            # For now, just use gene name as is and warn if no mapping exists
            gene_to_seq_id = {}
            for gene in candidate_genes:
                # Try to find a matching sequence ID
                potential_id = gene.split('_')[0]  # Take first part of gene name
                if potential_id in sequence_index:
                    gene_to_seq_id[gene] = potential_id
                else:
                    # Fallback to using gene number if it's in GENE<number> format
                    match = re.match(r'GENE(\d+)', gene)
                    if match:
                        seq_id = str(int(match.group(1)) % 13 + 1)
                        if seq_id in sequence_index:
                            gene_to_seq_id[gene] = seq_id

            if not gene_to_seq_id:
                raise ValueError("No valid sequence mappings found for any genes")

            self.logger.info(f"Found sequence mappings for {len(gene_to_seq_id)}/{len(candidate_genes)} genes")

            # Process genes sequence by sequence using indexed access
            results = []
            for i, gene in enumerate(candidate_genes, 1):
                try:
                    if gene not in gene_to_seq_id:
                        self.logger.warning(f"No sequence mapping found for gene: {gene}")
                        continue

                    self.logger.info(f"Processing gene {i}/{len(candidate_genes)}: {gene}")
                    
                    seq_id = gene_to_seq_id[gene]
                    if seq_id not in sequence_index:
                        self.logger.warning(f"Invalid sequence ID mapping for gene: {gene}")
                        continue

                    # Read only the needed sequence using index
                    with gzip.open(genome_path, 'rt') as handle:
                        handle.seek(sequence_index[seq_id])
                        record = next(SeqIO.parse(handle, "fasta"))
                        promoter_seq = str(record.seq[:seq_params['promoter_length']])
                        current_genome = {seq_id: record.seq}

                    # Analyze sequence
                    self.logger.info(f"Analyzing {gene} sequence ({len(promoter_seq)} bp)")
                    analysis = analyzer.evaluate_sequence(promoter_seq, current_genome)
                    analysis['gene_name'] = gene
                    analysis['sequence_id'] = seq_id
                    results.append(analysis)

                    # Log progress
                    self.logger.info(f"Completed analysis of {gene} (Score: {analysis.get('overall_score', 'N/A')})")

                    # Clean up memory
                    del current_genome

                except Exception as e:
                    self.logger.warning(f"Error processing gene {gene}: {e}")
                    continue
                
            # Create visualizations with progress tracking
            self.logger.info("Generating stability distribution plot...")
            stability_data = {
                f"{r['gene_name']} ({r['sequence_id']})": r['stability']['total_stability']
                for r in results
            }
            
            plot_stability_distribution(
                stability_data,
                self.config['directories']['results']
            )
            self.logger.info("Stability plot generated successfully")
            
            # Convert results to DataFrame and clean up memory
            df_results = pd.DataFrame(results)
            del results, stability_data
            gc.collect()  # Force garbage collection
            
            return df_results
            
        except Exception as e:
            self.logger.error(f"Error in sequence analysis: {e}")
            raise
            
    def train_dnabert(self, sequences):
        """Fine-tune DNABERT-2 on siRNA sequences"""
        try:
            self.logger.info("Setting up DNABERT-2 training")
            model_params = self.config['model']['dnabert']
            
            # Initialize trainer
            trainer = DNABERTTrainer(
                model_name=model_params['base_model'],
                output_dir=self.config['directories']['models']
            )
            
            # Setup model
            trainer.setup()
            
            # Prepare data
            train_loader, val_loader = trainer.prepare_training_data(
                sequences,
                max_length=model_params['max_length'],
                batch_size=model_params['batch_size']
            )
            
            # Train model
            trainer.train(
                train_loader,
                val_loader,
                num_epochs=model_params['num_epochs'],
                learning_rate=model_params['learning_rate'],
                warmup_steps=model_params['warmup_steps'],
                save_steps=model_params['save_steps'],
                eval_steps=model_params['eval_steps']
            )
            
            # Generate sequences
            new_sequences = trainer.generate_sequences()
            
            # Evaluate sequences
            evaluations = trainer.evaluate_sequences(new_sequences)
            
            return evaluations
            
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
            seq_results = self.analyze_sequences(candidate_genes)
            
            # Train DNABERT and generate sequences
            if seq_results is not None and len(seq_results) > 0:
                model_results = self.train_dnabert(
                    seq_results['sequence'].tolist()
                )
                
                # Generate report figures
                generate_report_figures(
                    {
                        'differential_expression': de_results,
                        'sequence_analysis': seq_results,
                        'model_results': model_results
                    },
                    self.config['directories']['results']
                )
            
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