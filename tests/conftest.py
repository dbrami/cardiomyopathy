"""
Pytest configuration and shared fixtures
"""

import os
import sys
import pytest
import shutil
from pathlib import Path
import yaml
import logging

# Add tests directory to Python path
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from test_utils import TestDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def config(project_root):
    """Load project configuration"""
    config_path = project_root / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def data_generator(project_root):
    """Create TestDataGenerator instance"""
    return TestDataGenerator(project_root)

@pytest.fixture(scope="session")
def geo_data(data_generator):
    """Load sample GEO expression data"""
    return data_generator.load_geo_sample()

@pytest.fixture(scope="session")
def genome_data(data_generator):
    """Load sample genome sequences"""
    return data_generator.load_genome_sample()

@pytest.fixture(scope="session")
def gtex_data(data_generator):
    """Load sample GTEx expression data"""
    return data_generator.load_gtex_sample()

@pytest.fixture(scope="session")
def encode_peaks(data_generator):
    """Load sample ENCODE peak data"""
    return data_generator.load_encode_peaks()

@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """Create and cleanup a temporary directory"""
    yield tmp_path
    # Cleanup is handled automatically by pytest

@pytest.fixture(scope="session")
def test_output_dir(project_root):
    """Create and manage test output directory"""
    output_dir = project_root / 'test_output'
    output_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'results').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    yield output_dir
    
    # Clean up on session end
    shutil.rmtree(output_dir, ignore_errors=True)

def pytest_configure(config):
    """Pytest configuration hook"""
    # Add marks
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add slow marker to tests that use real data
    for item in items:
        if "data_generator" in item.fixturenames:
            item.add_marker(pytest.mark.slow)

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    # Verify environment
    if os.environ.get('CONDA_DEFAULT_ENV') != 'dnabert_env':
        pytest.exit("Tests must be run in dnabert_env conda environment")
        
    # Verify data directories
    required_dirs = ['geo', 'encode', 'gtex', 'reference', 'logs']
    project_root = Path(__file__).parent.parent
    
    for dir_name in required_dirs:
        dir_path = project_root / 'data' / dir_name
        if not dir_path.exists():
            pytest.exit(f"Required directory missing: {dir_path}")

@pytest.fixture(autouse=True)
def _setup_test_env(monkeypatch, project_root):
    """Automatically setup test environment for each test"""
    # Add src to Python path
    monkeypatch.syspath_prepend(str(project_root / 'src'))
    
    # Set test environment variable
    monkeypatch.setenv('TESTING', 'true')
