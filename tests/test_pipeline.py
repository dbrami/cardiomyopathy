"""
Tests for main siRNA analysis pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import json
from unittest.mock import patch, MagicMock
from src.siRNA_v3 import Pipeline
from src.gene_sequence_analysis import process_gene_sequences

# Test fixtures and utilities
def create_test_data(path, data):
    """Helper to create test files"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(data)

class TestPipeline:
    """Test class for siRNA analysis pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment"""
        # Create directory structure
        self.dirs = {
            'root': tmp_path,
            'data': tmp_path / 'data',
            'geo': tmp_path / 'data/geo',
            'reference': tmp_path / 'data/reference',
            'logs': tmp_path / 'logs',
            'models': tmp_path / 'models',
            'results': tmp_path / 'results'
        }
        
        # Create directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create test data
        self.create_test_files()
        self.create_test_config()
        
        # Initialize pipeline
        self.pipeline = Pipeline(config_path=self.dirs['root'] / 'config.yaml')
        
    def create_test_config(self):
        """Create test configuration file"""
        config = {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'directories': {
                'data': str(self.dirs['data']),
                'geo': str(self.dirs['geo']),
                'reference': str(self.dirs['reference']),
                'logs': str(self.dirs['logs']),
                'models': str(self.dirs['models']),
                'results': str(self.dirs['results'])
            },
            'files': {
                'geo': {
                    'counts': {
                        'filename': 'test_counts.txt',
                        'compressed': False
                    },
                    'series_matrix': {
                        'filename': 'test_series_matrix.txt'
                    },
                    'sample_mapping': {
                        'cardiomyopathy': [
                            {'G1': 'GSM1000'},
                            {'G2': 'GSM1001'}
                        ],
                        'healthy': [
                            {'G3': 'GSM1002'},
                            {'G4': 'GSM1003'}
                        ]
                    }
                },
                'reference': {
                    'genome': {
                        'filename': 'test_genome.fa',
                        'compressed': False
                    },
                    'annotation': {
                        'filename': 'test_annotation.gtf'
                    }
                }
            },
            'model': {
                'dnabert': {
                    'base_model': 'zhihan1996/DNABERT-2-117M',
                    'max_length': 32,
                    'batch_size': 2,
                    'num_epochs': 1,
                    'learning_rate': 1e-4,
                    'warmup_steps': 10,
                    'save_steps': 50,
                    'eval_steps': 50
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
        
        config_path = self.dirs['root'] / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
        
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert isinstance(self.pipeline, Pipeline)
        assert self.pipeline.config is not None
        assert self.pipeline.logger is not None
        
    def test_sample_mapping(self):
        """Test sample ID mapping functionality"""
        assert self.pipeline.get_sample_id('GSM1000', 'short') == 'G1'
        assert self.pipeline.get_sample_id('GSM1001', 'short') == 'G2'
        assert self.pipeline.get_sample_id('G1', 'geo') == 'GSM1000'
        assert self.pipeline.get_sample_id('G2', 'geo') == 'GSM1001'
        
    @patch('src.gene_sequence_analysis.fetch_gene_sequence')
    def test_differential_expression(self, mock_fetch):
        """Test differential expression analysis"""
        mock_fetch.return_value = {
            'sequence': 'ATCG' * 25,
            'promoter': 'GCTA' * 25
        }
        
        expr_df = self.pipeline.load_geo_expression()
        assert isinstance(expr_df, pd.DataFrame)
        assert not expr_df.empty
        
        de_results = self.pipeline.perform_differential_expression(expr_df)
        assert isinstance(de_results, pd.DataFrame)
        assert not de_results.empty
        assert all(col in de_results.columns
                  for col in ['gene_name', 'log2fc', 'pvalue', 'padj'])
        
    @patch('src.siRNA_v3.DNABERTTrainer')
    def test_dnabert_training(self, mock_trainer):
        """Test DNABERT model training"""
        trainer_instance = mock_trainer.return_value
        trainer_instance.prepare_training_data.return_value = (MagicMock(), MagicMock())
        trainer_instance.train.return_value = {'loss': 0.1, 'accuracy': 0.9}
        trainer_instance.generate_sequences.return_value = ['ATCG' * 10]
        
        sequences = ['ATCG' * 25, 'GCTA' * 25]
        results = self.pipeline.train_dnabert(sequences)
        
        assert isinstance(results, dict)
        assert all(key in results for key in ['training', 'generation', 'input_stats'])
        assert results['input_stats']['total_sequences'] == len(sequences)
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'directories': {
                'data': str(self.dirs['data']),
                'geo': str(self.dirs['geo']),
                'reference': str(self.dirs['reference']),
                'logs': str(self.dirs['logs']),
                'models': str(self.dirs['models']),
                'results': str(self.dirs['results'])
            },
            'files': {
                'geo': {
                    'counts': {
                        'filename': 'test_counts.txt',
                        'compressed': False
                    },
                    'series_matrix': {
                        'filename': 'test_series_matrix.txt'
                    },
                    'sample_mapping': {
                        'cardiomyopathy': [
                            {'G1': 'GSM1000'},
                            {'G2': 'GSM1001'}
                        ],
                        'healthy': [
                            {'G3': 'GSM1002'},
                            {'G4': 'GSM1003'}
                        ]
                    }
                },
                'reference': {
                    'genome': {
                        'filename': 'test_genome.fa',
                        'compressed': False
                    },
                    'annotation': {
                        'filename': 'test_annotation.gtf'
                    }
                }
            },
            'model': {
                'dnabert': {
                    'base_model': 'zhihan1996/DNABERT-2-117M',
                    'max_length': 32,
                    'batch_size': 2,
                    'num_epochs': 1,
                    'learning_rate': 1e-4,
                    'warmup_steps': 10,
                    'save_steps': 50,
                    'eval_steps': 50
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
        
        config_path = self.dirs['root'] / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config
        # Create series matrix
        series_matrix = self.data_dir / 'geo/test_series_matrix.txt'
        series_data = (
            '!Sample_title\t"cardiomyopathy sample 1"\n'
            '!Sample_geo_accession\t"GSM1000"\n'
            '!Sample_title\t"healthy control 1"\n'
            '!Sample_geo_accession\t"GSM1002"\n'
        )
        create_test_data(series_matrix, series_data)
        
        # Create counts file
        counts_file = self.data_dir / 'geo/test_counts.txt'
        counts_data = (
            'gene_id\tgene_name\tG1\tG2\tG3\tG4\n'
            'GENE1\tGENE1\t100\t110\t50\t55\n'
            'GENE2\tGENE2\t200\t220\t100\t110\n'
        )
        create_test_data(counts_file, counts_data)
        
        # Create annotation file
        annot_file = self.data_dir / 'reference/test_annotation.gtf'
        annot_data = (
            'chr1\tGENCODE\tgene\t1000\t2000\t.\t+\t.\tgene_id "GENE1"; gene_name "GENE1";\n'
            'chr1\tGENCODE\tgene\t3000\t4000\t.\t+\t.\tgene_id "GENE2"; gene_name "GENE2";\n'
        )
        create_test_data(annot_file, annot_data)
        
        return {
            'series_matrix': series_matrix,
            'counts': counts_file,
            'annotation': annot_file
        }
    @pytest.fixture
    def mock_config(self):
    """Create mock configuration for testing"""
    config = {
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'directories': {
            'data': str(tmp_path / 'data'),
            'geo': str(tmp_path / 'data/geo'),
            'reference': str(tmp_path / 'data/reference'),
            'logs': str(tmp_path / 'logs'),
            'models': str(tmp_path / 'models'),
            'results': str(tmp_path / 'results')
        },
        'files': {
            'geo': {
                'counts': {'filename': 'test_counts.txt', 'compressed': False},
                'series_matrix': {'filename': 'test_series_matrix.txt'},
                'sample_mapping': {
                    'cardiomyopathy': [{'G1': 'GSM1000'}, {'G2': 'GSM1001'}],
                    'healthy': [{'G3': 'GSM1002'}, {'G4': 'GSM1003'}]
                }
            },
            'reference': {
                'genome': {'filename': 'test_genome.fa', 'compressed': False},
                'annotation': {'filename': 'test_annotation.gtf'}
            }
        },
        'model': {
            'dnabert': {
                'base_model': 'zhihan1996/DNABERT-2-117M',
                'max_length': 32,
                'batch_size': 2,
                'num_epochs': 1,
                'learning_rate': 1e-4,
                'warmup_steps': 10,
                'save_steps': 50,
                'eval_steps': 50
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
    
    config_path = tmp_path / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

@pytest.fixture
def pipeline(mock_config):
    """Create pipeline instance"""
    return Pipeline(config_path=mock_config)

@pytest.fixture
def mock_expression_data():
    """Create mock expression data"""
    data = pd.DataFrame(index=pd.Index(['gene1', 'gene2', 'gene3'], name='gene_id'))
    data['gene_name'] = ['GENE1', 'GENE2', 'GENE3']
    
    # Add sample columns with sample mapping
    for geo_id, short_id in [('GSM1000', 'G1'), ('GSM1001', 'G2'),
                            ('GSM1002', 'G3'), ('GSM1003', 'G4')]:
        data[short_id] = np.random.normal(100 if 'G1' <= short_id <= 'G2' else 50, 10, size=3)
    
    return data

@pytest.fixture
def setup_test_files(temp_paths):
    """Create test files needed for pipeline testing"""
    # Create series matrix file
    series_matrix = temp_paths['geo'] / 'test_series_matrix.txt'
    with open(series_matrix, 'w') as f:
        f.write('!Sample_title\t"cardiomyopathy sample 1"\n')
        f.write('!Sample_geo_accession\t"GSM1000"\n')
        f.write('!Sample_title\t"healthy control 1"\n')
        f.write('!Sample_geo_accession\t"GSM1002"\n')
    
    # Create annotation file
    annot_file = temp_paths['reference'] / 'test_annotation.gtf'
    with open(annot_file, 'w') as f:
        f.write('chr1\tGENCODE\tgene\t1000\t2000\t.\t+\t.\tgene_id "GENE1"; gene_name "GENE1";\n')
        f.write('chr1\tGENCODE\tgene\t3000\t4000\t.\t+\t.\tgene_id "GENE2"; gene_name "GENE2";\n')

@pytest.mark.usefixtures("setup_test_files")
def test_sample_mapping(pipeline):
    """Test sample ID mapping functionality"""
    # Test GEO to short ID mapping
    assert pipeline.get_sample_id('GSM1000', 'short') == 'G1'
    assert pipeline.get_sample_id('GSM1001', 'short') == 'G2'
    
    # Test short to GEO ID mapping
    assert pipeline.get_sample_id('G1', 'geo') == 'GSM1000'
    assert pipeline.get_sample_id('G2', 'geo') == 'GSM1001'
    
    # Test invalid IDs
    assert pipeline.get_sample_id('INVALID', 'short') == 'INVALID'
    assert pipeline.get_sample_id('INVALID', 'geo') == 'INVALID'

def test_sample_groups(pipeline):
    """Test sample grouping functionality"""
    assert hasattr(pipeline, 'sample_groups')
    assert 'cardiomyopathy' in pipeline.sample_groups
    assert 'healthy' in pipeline.sample_groups
    assert len(pipeline.sample_groups['cardiomyopathy']) == 2
    assert len(pipeline.sample_groups['healthy']) == 2

def test_fpkm_data_integration(pipeline, mock_gene_sequence_data, mock_expression_data, temp_paths):
    """Test integration of FPKM data with gene sequence analysis"""
    # Setup test data
    fpkm_path = temp_paths['geo'] / 'test_counts.txt'
    mock_expression_data.to_csv(fpkm_path, sep='\t', index=True)
    
    # Mock gene sequence fetching
    with patch('src.gene_sequence_analysis.fetch_gene_sequence') as mock_fetch:
        mock_fetch.side_effect = lambda gene_id: mock_gene_sequence_data.get(gene_id, {})
        
        # Load and process FPKM data
        expr_df = pipeline.load_geo_expression()
        assert isinstance(expr_df, pd.DataFrame)
        assert not expr_df.empty
        
        # Perform differential expression analysis
        de_results = pipeline.perform_differential_expression(expr_df)
        assert isinstance(de_results, pd.DataFrame)
        assert not de_results.empty
        
        # Convert results for sequence analysis
        diff_expr_result = {
            'up_regulated': de_results[de_results['log2fc'] > 0]['gene_name'].tolist(),
            'down_regulated': de_results[de_results['log2fc'] < 0]['gene_name'].tolist()
        }
        
        # Run sequence analysis
        sequence_df = process_gene_sequences(
            annotation_path=str(temp_paths['reference'] / 'test_annotation.gtf'),
            fpkm_path=str(fpkm_path),
            diff_expr_result=diff_expr_result
        )
        
        # Verify integration results
        assert isinstance(sequence_df, pd.DataFrame)
        assert not sequence_df.empty
        assert all(col in sequence_df.columns for col in ['GeneID', 'Sequence', 'Promoter', 'Regulation'])
        assert len(sequence_df) <= len(de_results)

@patch('src.siRNA_v3.DNABERTTrainer')
def test_dnabert_integration(mock_trainer, pipeline, mock_gene_sequence_data, setup_test_files):
    """Test DNABERT training integration"""
    # Configure mock trainer
    trainer_instance = mock_trainer.return_value
    trainer_instance.prepare_training_data.return_value = (MagicMock(), MagicMock())
    trainer_instance.train.return_value = {
        'loss': 0.1,
        'accuracy': 0.9,
        'epochs_completed': 1
    }
    trainer_instance.generate_sequences.return_value = ['ATCG' * 10]
    
    # Get test sequences
    test_sequences = [seq['promoter'] for seq in mock_gene_sequence_data.values()]
    
    # Run training
    results = pipeline.train_dnabert(test_sequences)
    
    # Verify trainer usage
    mock_trainer.assert_called_once()
    trainer_instance.prepare_training_data.assert_called_once()
    trainer_instance.train.assert_called_once()
    
    # Check results
    assert isinstance(results, dict)
    assert 'training' in results
    assert 'generation' in results
    assert 'input_stats' in results
    assert results['input_stats']['total_sequences'] == len(test_sequences)
    assert results['training']['accuracy'] == 0.9
    assert len(results['generation']['sequences']) > 0

@pytest.fixture
def temp_paths(tmp_path):
    """Create temporary paths for testing"""
    paths = {
        'data': tmp_path / 'data',
        'geo': tmp_path / 'data/geo',
        'reference': tmp_path / 'data/reference',
        'logs': tmp_path / 'logs',
        'results': tmp_path / 'results'
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

@pytest.fixture
def mock_config(temp_paths):
    """Create mock configuration for testing"""
    config = {
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'directories': {
            'data': str(temp_paths['data']),
            'geo': str(temp_paths['geo']),
            'reference': str(temp_paths['reference']),
            'logs': str(temp_paths['logs']),
            'results': str(temp_paths['results'])
        },
        'files': {
            'geo': {
                'counts': {
                    'filename': 'test_counts.txt',
                    'compressed': False
                },
                'series_matrix': {
                    'filename': 'test_series_matrix.txt'
                },
                'sample_mapping': {
                    'cardiomyopathy': [
                        {'G1': 'GSM1000'},
                        {'G2': 'GSM1001'}
                    ],
                    'healthy': [
                        {'G3': 'GSM1002'},
                        {'G4': 'GSM1003'}
                    ]
                }
            },
            'reference': {
                'genome': {
                    'filename': 'test_genome.fa',
                    'compressed': False
                },
                'annotation': {
                    'filename': 'test_annotation.gtf'
                }
            }
        },
        'model': {
            'dnabert': {
                'base_model': "zhihan1996/DNABERT-2-117M",
                'max_length': 32,
                'batch_size': 2,
                'num_epochs': 1,
                'learning_rate': 1e-4,
                'warmup_steps': 10,
                'save_steps': 50,
                'eval_steps': 50
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
    
    config_path = temp_paths['data'] / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    return config_path

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
                },
                'series_matrix': {
                    'filename': 'test_series_matrix.txt'
                },
                'sample_mapping': {
                    'cardiomyopathy': [
                        {'G1': 'GSM1000'},
                        {'G2': 'GSM1001'}
                    ],
                    'healthy': [
                        {'G3': 'GSM1002'},
                        {'G4': 'GSM1003'}
                    ]
                }
            },
            'reference': {
                'genome': {
                    'filename': 'test_genome.fa',
                    'compressed': False
                },
                'annotation': {
                    'filename': 'test_annotation.gtf'
                }
            }
        },
        'model': {
            'dnabert': {
                'base_model': "zhihan1996/DNABERT-2-117M",
                'max_length': 32,
                'batch_size': 2,
                'num_epochs': 1,
                'learning_rate': 1e-4,
                'warmup_steps': 10,
                'save_steps': 50,
                'eval_steps': 50
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
    
    # Create test files
    series_matrix = temp_paths['geo'] / 'test_series_matrix.txt'
    with open(series_matrix, 'w') as f:
        f.write('!Sample_title\t"healthy_1"\n')
        f.write('!Sample_geo_accession\t"GSM1000"\n')
        
    return config_path

@pytest.mark.dnabert
def test_dnabert_training(pipeline, mock_gene_sequence_data):
    """Test DNABERT model training"""
    test_sequences = [seq['promoter'] for seq in mock_gene_sequence_data.values()]
    
    with patch('src.siRNA_v3.DNABERTTrainer') as mock_trainer:
        # Configure mock trainer
        trainer_instance = mock_trainer.return_value
        trainer_instance.prepare_training_data.return_value = (MagicMock(), MagicMock())
        trainer_instance.train.return_value = {
            'loss': 0.1,
            'accuracy': 0.9,
            'epochs_completed': 1
        }
        trainer_instance.generate_sequences.return_value = ['ATCG' * 10]
        
        # Run training
        results = pipeline.train_dnabert(test_sequences)
        
        # Verify trainer usage
        assert mock_trainer.called
        trainer_instance.prepare_training_data.assert_called_once()
        trainer_instance.train.assert_called_once()
        
        # Verify results structure
        assert isinstance(results, dict)
        assert all(key in results for key in ['training', 'generation', 'input_stats'])
        assert results['input_stats']['total_sequences'] == len(test_sequences)
        assert results['training']['accuracy'] == 0.9

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

def test_sample_mapping(pipeline):
    """Test sample ID mapping functionality"""
    # Test GEO to short ID mapping
    assert pipeline.get_sample_id('GSM1000', 'short') == 'G1'
    assert pipeline.get_sample_id('GSM1001', 'short') == 'G2'
    
    # Test short to GEO ID mapping
    assert pipeline.get_sample_id('G1', 'geo') == 'GSM1000'
    assert pipeline.get_sample_id('G2', 'geo') == 'GSM1001'
    
    # Test invalid IDs
    assert pipeline.get_sample_id('INVALID', 'short') == 'INVALID'
    assert pipeline.get_sample_id('INVALID', 'geo') == 'INVALID'

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