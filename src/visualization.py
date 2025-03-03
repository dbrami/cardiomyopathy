"""
Visualization module for siRNA analysis pipeline

This module provides functions for creating various plots and visualizations
for the siRNA analysis pipeline results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def setup_plot_style():
    """Set consistent style for all plots"""
    plt.style.use('default')
    sns.set_style("whitegrid")
    sns.set_palette("husl")

def create_volcano_plot(de_results, output_dir,
                       p_thresh=0.05, fc_thresh=1.0,
                       dpi=300, filename='volcano_plot.pdf'):
    """
    Create volcano plot from differential expression results
    
    Args:
        de_results (pd.DataFrame): Differential expression results
        output_dir (str): Directory to save the plot
        p_thresh (float): P-value threshold for significance
        fc_thresh (float): Log2 fold change threshold
        dpi (int): Resolution for the output plot
        filename (str): Name of the output file
        
    Raises:
        ValueError: If the input data is invalid
    """
    try:
        # Validate input data
        if de_results is None:
            raise ValueError("Data cannot be None")
            
        if not isinstance(de_results, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        if len(de_results) == 0:
            raise ValueError("Empty DataFrame")
            
        required_cols = ['log2fc', 'padj']
        if not all(col in de_results.columns for col in required_cols):
            raise ValueError("Missing required columns")
            
        plt.figure(figsize=(12, 8))
        
        # Create the scatter plot
        plt.scatter(
            de_results['log2fc'],
            -np.log10(de_results['padj']),
            alpha=0.5,
            color='grey',
            label='Not Significant'
        )
        
        # Highlight significant points
        significant = (de_results['padj'] < p_thresh) & \
                     (abs(de_results['log2fc']) > fc_thresh)
        
        plt.scatter(
            de_results.loc[significant, 'log2fc'],
            -np.log10(de_results.loc[significant, 'padj']),
            alpha=0.8,
            color='red',
            label='Significant'
        )
        
        # Add labels for significant genes
        for idx, row in de_results[significant].iterrows():
            plt.annotate(
                row['gene_name'],
                (row['log2fc'], -np.log10(row['padj'])),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        
        # Add threshold lines
        plt.axhline(y=-np.log10(p_thresh), color='r', 
                   linestyle='--', alpha=0.3)
        plt.axvline(x=-fc_thresh, color='r', 
                   linestyle='--', alpha=0.3)
        plt.axvline(x=fc_thresh, color='r', 
                   linestyle='--', alpha=0.3)
        
        # Customize plot
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-log10(Adjusted P-value)')
        plt.title('Volcano Plot: Differential Expression Analysis')
        plt.legend()
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save plot
        output_path = Path(output_dir) / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Volcano plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating volcano plot: {e}")
        raise

def plot_stability_distribution(stability_scores, output_dir):
    """
    Plot distribution of siRNA stability scores
    
    Args:
        stability_scores (dict): Dictionary of sequence IDs and their stability scores
        output_dir (str): Directory to save the plot
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Create distribution plot
        sns.histplot(
            data=pd.DataFrame.from_dict(
                stability_scores, 
                orient='index', 
                columns=['stability_score']
            ),
            x='stability_score',
            kde=True
        )
        
        plt.xlabel('Stability Score')
        plt.ylabel('Count')
        plt.title('Distribution of siRNA Stability Scores')
        
        # Save plot
        output_path = Path(output_dir) / 'stability_distribution.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Stability distribution plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating stability distribution plot: {e}")
        raise

def plot_sequence_embeddings(embeddings, labels, output_dir):
    """
    Create PCA plot of sequence embeddings
    
    Args:
        embeddings (np.array): Matrix of sequence embeddings
        labels (list): List of sequence labels
        output_dir (str): Directory to save the plot
    """
    try:
        from sklearn.decomposition import PCA
        
        # Perform PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.6
        )
        
        # Add labels
        for i, label in enumerate(labels):
            plt.annotate(
                label,
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        
        # Customize plot
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA of Sequence Embeddings')
        
        # Save plot
        output_path = Path(output_dir) / 'sequence_embeddings_pca.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sequence embeddings plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating sequence embeddings plot: {e}")
        raise

def create_heatmap(matrix, row_labels, col_labels, 
                  title, output_dir, filename):
    """
    Create heatmap visualization
    
    Args:
        matrix (np.array): Data matrix to visualize
        row_labels (list): Labels for rows
        col_labels (list): Labels for columns
        title (str): Plot title
        output_dir (str): Directory to save the plot
        filename (str): Name of output file
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(
            matrix,
            xticklabels=col_labels,
            yticklabels=row_labels,
            cmap='YlOrRd',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Value'}
        )
        
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        raise

def plot_off_target_scores(off_target_data, output_dir):
    """
    Create visualization of off-target prediction scores
    
    Args:
        off_target_data (pd.DataFrame): DataFrame with off-target predictions
        output_dir (str): Directory to save the plot
    """
    try:
        plt.figure(figsize=(10, 6))
        
        sns.boxplot(
            data=off_target_data,
            x='sequence_id',
            y='off_target_score'
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('siRNA Sequence')
        plt.ylabel('Off-target Score')
        plt.title('Distribution of Off-target Prediction Scores')
        
        # Save plot
        output_path = Path(output_dir) / 'off_target_scores.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Off-target scores plot saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating off-target scores plot: {e}")
        raise

def generate_report_figures(results, output_dir):
    """
    Generate all figures for the analysis report
    
    Args:
        results (dict): Dictionary containing all analysis results
        output_dir (str): Directory to save the plots
    """
    try:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        setup_plot_style()
        
        # Generate all plots
        if 'differential_expression' in results:
            create_volcano_plot(
                results['differential_expression'],
                output_dir
            )
            
        if 'stability_scores' in results:
            plot_stability_distribution(
                results['stability_scores'],
                output_dir
            )
            
        if 'sequence_embeddings' in results:
            plot_sequence_embeddings(
                results['sequence_embeddings']['embeddings'],
                results['sequence_embeddings']['labels'],
                output_dir
            )
            
        if 'off_target_predictions' in results:
            plot_off_target_scores(
                results['off_target_predictions'],
                output_dir
            )
            
        logger.info("Successfully generated all report figures")
        
    except Exception as e:
        logger.error(f"Error generating report figures: {e}")
        raise