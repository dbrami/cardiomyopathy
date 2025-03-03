"""
Tests for main siRNA analysis pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from src.siRNA_v3 import Pipeline

@pytest.fixture
def mock_config(temp_dir):
    """Create mock configuration for testing"""
    config = {
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'directories': {
            'data': str(temp_dir / 'data'),
            'geo': str(temp_dir / 'data/geo'),
            'encode': str(temp_dir / 'data/encode'),
            'gtex': str(temp_dir / 'data/gtex'),
            'reference': str(temp_dir / 'data/reference'),
            'logs': str(temp_dir / 'data/logs'),
            'models': str(temp_dir / 'models'),
            'results': str(temp_dir / 'results')
        },
        'files': {
            'geo': {
                'counts': {
                    'filename': 'test_counts.txt',
                    'compressed': False
                }
            },
            'reference': {
                'genome': {
                    'filename': 'test_genome.fa',
                    'compressed': False
                }
            }
        },
        'model': {
            'dnabert': {
                'base_model': "zhihan1996/DNABERT-2-117M",
                'max_length': 32,
                'batch_size': 2
            }
        },
        'analysis': {
            'differential_expression': {
                'p_value_threshold': 0.05,
                'log2fc_threshold': 1.0,
                'min_expression': 10
            },
            'sequence': {
                'promoter_length': 100,
                'max_mismatches': 3,
                'seed_region': 2
            }
        }
    }
    
    # Create directory structure
    for dir_path in config['directories'].values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    # Write config file
    config_path = temp_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
        
    return config_path

@pytest.fixture
def pipeline(mock_config):
    """Create pipeline instance for testing"""
    return Pipeline(config_path=mock_config)

@pytest.fixture
def mock_expression_data():
    """Create mock expression data"""
    # Create data with gene_id as index
    gene_ids = ['gene1', 'gene2', 'gene3']
    gene_names = ['GENE1', 'GENE2', 'GENE3']
    
    data = pd.DataFrame(index=pd.Index(gene_ids, name='gene_id'))
    data['gene_name'] = gene_names
    
    # Add sample columns (26 cardio + 10 control)
    for i in range(36):
        col_name = f'GSM{1000+i}'
        if i < 26:  # Cardio samples
            data[col_name] = np.random.normal(100, 10, size=3)
        else:  # Control samples
            data[col_name] = np.random.normal(50, 10, size=3)
            
    # Reset index to include gene_id as a column for saving
    data = data.reset_index()
    return data

def test_pipeline_initialization(pipeline, mock_config):
    """Test pipeline initialization"""
    assert pipeline.config is not None
    assert pipeline.logger is not None
    
    # Check directory creation
    for dir_path in pipeline.config['directories'].values():
        assert Path(dir_path).exists()

def test_load_geo_expression(pipeline, mock_expression_data, temp_dir):
    """Test loading GEO expression data"""
    # Save mock data
    data_path = Path(pipeline.config['directories']['geo']) / 'test_counts.txt'
    mock_expression_data.to_csv(data_path, sep='\t', index=False)
    
    # Load data
    expr_df = pipeline.load_geo_expression()
    
    assert isinstance(expr_df, pd.DataFrame)
    assert len(expr_df) > 0
    assert len(expr_df.columns) == 37  # 36 samples + gene_name (gene_id becomes index)
    assert 'gene_name' in expr_df.columns
    assert expr_df.index.name == 'gene_id'
    assert all(col.startswith('GSM') for col in expr_df.columns if col != 'gene_name')

def test_differential_expression(pipeline, mock_expression_data):
    """Test differential expression analysis"""
    de_results = pipeline.perform_differential_expression(mock_expression_data)
    
    assert isinstance(de_results, pd.DataFrame)
    assert 'gene_name' in de_results.columns
    assert 'log2fc' in de_results.columns
    assert 'padj' in de_results.columns

@pytest.mark.integration
def test_sequence_analysis(pipeline, data_generator):
    """Test sequence analysis with real data"""
    # Get sample sequences
    candidates = ['GENE1', 'GENE2']
    genome_data = data_generator.load_genome_sample(num_sequences=2)
    
    # Create mock genome records
    mock_genome = {record.id: record.seq for record in genome_data}
    
    # Analyze sequences
    with pytest.MonkeyPatch.context() as m:
        # Mock genome loading
        m.setattr('Bio.SeqIO.parse', lambda *args, **kwargs: genome_data)
        results = pipeline.analyze_sequences(candidates)
    
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0

@pytest.mark.slow
def test_complete_pipeline_run(pipeline, mock_expression_data, temp_dir, data_generator):
    """Test complete pipeline execution"""
    # Setup test data
    data_path = Path(pipeline.config['directories']['geo']) / 'test_counts.txt'
    mock_expression_data.to_csv(data_path, sep='\t', index=False)
    
    # Mock genome data
    genome_data = data_generator.load_genome_sample()
    with pytest.MonkeyPatch.context() as m:
        m.setattr('Bio.SeqIO.parse', lambda *args, **kwargs: genome_data)
        
        # Run pipeline
        try:
            pipeline.run()
            
            # Check results
            results_dir = Path(pipeline.config['directories']['results'])
            expected_files = [
                'volcano_plot.pdf',
                'stability_distribution.pdf',
                'sequence_embeddings_pca.pdf'
            ]
            
            for filename in expected_files:
                assert (results_dir / filename).exists()
                
        except Exception as e:
            pytest.fail(f"Pipeline execution failed: {str(e)}")

def test_error_handling(temp_dir, mock_config):
    """Test error handling in pipeline"""
    # Test with invalid configuration
    with pytest.raises(RuntimeError, match="Error loading config"):
        Pipeline(config_path=temp_dir / "nonexistent.yaml")
    
    # Test with invalid data file
    pipeline = Pipeline(config_path=mock_config)
    pipeline.config['files']['geo']['counts']['filename'] = "nonexistent.txt"
    with pytest.raises(FileNotFoundError):
        pipeline.load_geo_expression()

@pytest.mark.parametrize("sample_count,expected_genes", [
(5, 5),
(10, 10),
(20, 20)
])
def test_pipeline_with_different_sample_sizes(pipeline, sample_count, expected_genes):
    """Test pipeline with different sample sizes"""
    # Create test data with different sizes
    data = pd.DataFrame({
        'gene_id': [f'gene{i}' for i in range(sample_count)],
        'gene_name': [f'GENE{i}' for i in range(sample_count)]
    })
    
    # Add sample columns
    for i in range(36):
        col_name = f'GSM{1000+i}'
        data[col_name] = np.random.normal(100 if i < 26 else 50, 10, size=sample_count)
    
    # Run analysis
    de_results = pipeline.perform_differential_expression(data)
    assert len(de_results) <= sample_count

@pytest.mark.integration
def test_pipeline_with_real_data(pipeline, data_generator):
    """Test pipeline with real data samples"""
    # Load real data
    geo_data = data_generator.load_geo_sample()
    genome_data = data_generator.load_genome_sample()
    
    # Run analysis steps
    de_results = pipeline.perform_differential_expression(geo_data)
    assert isinstance(de_results, pd.DataFrame)
    
    # Test sequence analysis with real genome data
    with pytest.MonkeyPatch.context() as m:
        m.setattr('Bio.SeqIO.parse', lambda *args, **kwargs: genome_data)
        seq_results = pipeline.analyze_sequences(de_results['gene_name'].head().tolist())
        assert isinstance(seq_results, pd.DataFrame)