#!/usr/bin/env python3
"""
Test runner script for siRNA analysis pipeline

This script executes all unit tests and generates a coverage report.
Ensures proper environment setup and data availability before running tests.
"""

import unittest
import coverage
import sys
import os
from pathlib import Path
import subprocess
from importlib.metadata import distribution, PackageNotFoundError
import shutil
import yaml
import logging
from typing import List, Dict

def setup_logging():
    """Configure logging for test execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_execution.log')
        ]
    )
    return logging.getLogger(__name__)

def validate_environment() -> bool:
    """
    Validate that we're running in the correct conda environment
    with all required packages installed
    """
    logger = logging.getLogger(__name__)
    
    # Check if we're in dnabert_env
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env != 'dnabert_env':
        logger.error(f"Tests must be run in dnabert_env conda environment, current: {conda_env}")
        return False
    
    # Check required packages from requirements.txt
    requirements_path = Path(__file__).parent.parent / 'requirements.txt'
    try:
        with open(requirements_path) as f:
            for line in f:
                # Skip comments and empty lines
                if line.strip() and not line.startswith('#'):
                    # Parse package name from requirement
                    pkg_name = line.split('==')[0].split('>=')[0].strip()
                    try:
                        distribution(pkg_name)
                    except PackageNotFoundError:
                        logger.error(f"Required package not found: {pkg_name}")
                        return False
    except Exception as e:
        logger.error(f"Error checking requirements: {e}")
        return False
        
    return True

def verify_data_directories() -> bool:
    """
    Verify that required data directories exist and contain necessary files
    """
    logger = logging.getLogger(__name__)
    project_root = Path(__file__).parent.parent
    
    # Load config to get required directories
    try:
        with open(project_root / 'config.yaml') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return False
    
    # Check each data directory with required file patterns
    required_dirs = [
        ('geo', ['*count_data.txt*', '*series_matrix.txt*']),
        ('encode', ['*.bed.gz']),
        ('gtex', ['*tpm.gct*']),
        ('reference', ['*.fa*', '*.fa.gz']),  # Updated patterns for reference genome
        ('logs', [])
    ]
    
    for dir_name, required_patterns in required_dirs:
        dir_path = project_root / 'data' / dir_name
        if not dir_path.exists():
            logger.error(f"Required directory missing: {dir_path}")
            return False
            
        # Skip file checks if no patterns specified
        if not required_patterns:
            continue
            
        # Check for required files using patterns
        found_files = False
        for pattern in required_patterns:
            if list(dir_path.glob(pattern)):
                found_files = True
                break
                
        if not found_files and required_patterns:
            logger.error(f"No files matching patterns {required_patterns} found in {dir_path}")
            return False
                
    return True

def setup_test_env():
    """Setup test environment"""
    logger = logging.getLogger(__name__)
    
    # Add src directory to Python path
    src_dir = str(Path(__file__).parent.parent / 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        
    # Create test output directories
    test_dirs = [
        'test_output/results',
        'test_output/logs',
        'test_output/models'
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    logger.info("Test environment setup completed")

def run_tests() -> bool:
    """
    Run all unit tests with coverage
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Start coverage measurement
    cov = coverage.Coverage(
        branch=True,
        source=['src'],
        omit=[
            '*/__init__.py',
            'tests/*',
            'src/DNABERT_2/*'
        ]
    )
    cov.start()
    
    try:
        # Discover and run tests
        loader = unittest.TestLoader()
        start_dir = os.path.dirname(os.path.abspath(__file__))
        suite = loader.discover(start_dir, pattern='test_*.py')
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Stop coverage measurement
        cov.stop()
        cov.save()
        
        # Generate coverage reports
        cov.html_report(directory='coverage_report')
        
        # Log test summary
        logger.info(f"Tests Run: {result.testsRun}")
        logger.info(f"Failures: {len(result.failures)}")
        logger.info(f"Errors: {len(result.errors)}")
        
        return result.wasSuccessful()
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

def cleanup_test_env():
    """Clean up test environment"""
    cleanup_dirs = [
        'test_output',
        'coverage_report',
        '__pycache__',
        '.pytest_cache'
    ]
    
    for dir_path in cleanup_dirs:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)

def main():
    """Main function to run tests"""
    logger = setup_logging()
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
            
        # Verify data directories
        if not verify_data_directories():
            logger.error("Data directory verification failed")
            sys.exit(1)
            
        # Setup test environment
        setup_test_env()
        
        # Run tests
        success = run_tests()
        
        # Clean up
        cleanup_test_env()
        
        # Exit with appropriate status
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        sys.exit(1)
    finally:
        cleanup_test_env()

if __name__ == '__main__':
    main()