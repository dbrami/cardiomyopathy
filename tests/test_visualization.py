"""
Tests for visualization functionality
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.visualization import (
    create_volcano_plot,
    plot_stability_distribution,
    plot_sequence_embeddings,
    plot_off_target_scores,
    generate_report_figures
)

@pytest.fixture(scope="module")
def test_de_data():
    """Create test differential expression data"""
    return pd.DataFrame({
        'gene_name': ['gene1', 'gene2', 'gene3'],
        'log2fc': [2.5, -1.5, 0.5],
        'padj': [0.001, 0.01, 0.8],
    })

@pytest.fixture(scope="module")
def test_stability_scores():
    """Create test stability scores"""
    return {
        'seq1': -10.5,
        'seq2': -8.3,
        'seq3': -12.1
    }

@pytest.fixture(scope="module")
def test_embeddings():
    """Create test sequence embeddings"""
    np.random.seed(42)  # For reproducibility
    return {
        'embeddings': np.random.randn(5, 10),  # 5 sequences, 10 dimensions
        'labels': ['seq1', 'seq2', 'seq3', 'seq4', 'seq5']
    }

@pytest.fixture(scope="module")
def test_off_target_data():
    """Create test off-target data"""
    return pd.DataFrame({
        'sequence_id': ['seq1', 'seq2', 'seq3'] * 3,
        'off_target_score': np.random.rand(9)
    })

@pytest.fixture(autouse=True)
def setup_matplotlib():
    """Configure matplotlib for testing"""
    plt.switch_backend('Agg')  # Use non-interactive backend
    yield
    plt.close('all')  # Cleanup after each test

def test_create_volcano_plot(test_de_data, test_output_dir):
    """Test volcano plot creation"""
    output_file = test_output_dir / 'results' / 'volcano_plot.pdf'
    
    create_volcano_plot(
        test_de_data,
        test_output_dir / 'results',
        p_thresh=0.05,
        fc_thresh=1.0
    )
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_plot_stability_distribution(test_stability_scores, test_output_dir):
    """Test stability distribution plot creation"""
    output_file = test_output_dir / 'results' / 'stability_distribution.pdf'
    
    plot_stability_distribution(
        test_stability_scores,
        test_output_dir / 'results'
    )
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_plot_sequence_embeddings(test_embeddings, test_output_dir):
    """Test sequence embeddings plot creation"""
    output_file = test_output_dir / 'results' / 'sequence_embeddings_pca.pdf'
    
    plot_sequence_embeddings(
        test_embeddings['embeddings'],
        test_embeddings['labels'],
        test_output_dir / 'results'
    )
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_plot_off_target_scores(test_off_target_data, test_output_dir):
    """Test off-target scores plot creation"""
    output_file = test_output_dir / 'results' / 'off_target_scores.pdf'
    
    plot_off_target_scores(
        test_off_target_data,
        test_output_dir / 'results'
    )
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_generate_report_figures(test_output_dir, test_de_data, 
                               test_stability_scores, test_embeddings,
                               test_off_target_data):
    """Test generation of all report figures"""
    results = {
        'differential_expression': test_de_data,
        'stability_scores': test_stability_scores,
        'sequence_embeddings': {
            'embeddings': test_embeddings['embeddings'],
            'labels': test_embeddings['labels']
        },
        'off_target_predictions': test_off_target_data
    }
    
    generate_report_figures(results, test_output_dir / 'results')
    
    # Check all expected files exist
    expected_files = [
        'volcano_plot.pdf',
        'stability_distribution.pdf',
        'sequence_embeddings_pca.pdf',
        'off_target_scores.pdf'
    ]
    
    for filename in expected_files:
        file_path = test_output_dir / 'results' / filename
        assert file_path.exists(), f"Expected file {filename} not found"
        assert file_path.stat().st_size > 0, f"File {filename} is empty"

@pytest.mark.parametrize("data,expected_error", [
    (pd.DataFrame(), "Empty DataFrame"),
    (None, "Data cannot be None"),
    (pd.DataFrame({'invalid': [1, 2, 3]}), "Missing required columns")
])
def test_invalid_data_handling(data, expected_error, test_output_dir):
    """Test handling of invalid input data"""
    with pytest.raises(ValueError, match=expected_error):
        create_volcano_plot(data, test_output_dir / 'results')

def test_plot_with_missing_directory(test_de_data):
    """Test plot creation with missing directory"""
    nonexistent_dir = Path("/nonexistent/directory")
    
    with pytest.raises(FileNotFoundError):
        create_volcano_plot(test_de_data, nonexistent_dir)

@pytest.mark.parametrize("dpi,figsize", [
    (100, (8, 6)),
    (300, (12, 8)),
    (600, (16, 10))
])
def test_plot_configurations(test_de_data, test_output_dir, dpi, figsize):
    """Test plot creation with different configurations"""
    output_file = test_output_dir / 'results' / f'volcano_plot_{dpi}dpi.pdf'
    
    plt.figure(figsize=figsize, dpi=dpi)
    filename = f'volcano_plot_{dpi}dpi.pdf'
    create_volcano_plot(
        test_de_data,
        test_output_dir / 'results',
        p_thresh=0.05,
        fc_thresh=1.0,
        dpi=dpi,
        filename=filename
    )
    
    assert output_file.exists()
    file_size = output_file.stat().st_size
    assert file_size > 0

@pytest.mark.integration
def test_real_data_visualization(data_generator, test_output_dir):
    """Test visualization with real data"""
    # Load real data samples
    geo_data = data_generator.load_geo_sample()
    
    # Create basic plots
    create_volcano_plot(
        geo_data,
        test_output_dir / 'results',
        p_thresh=0.05,
        fc_thresh=1.0
    )
    
    assert (test_output_dir / 'results' / 'volcano_plot.pdf').exists()