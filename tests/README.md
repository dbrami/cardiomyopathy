# Testing Documentation for siRNA Analysis Pipeline

## Overview
This directory contains the pytest-based test suite for the siRNA analysis pipeline. Tests use actual data samples from the project's data directories when available, with fallback to mock data when needed.

## Prerequisites

- Conda installed and in PATH
- Project data directories populated:
  - `data/geo/`: GEO expression data
  - `data/encode/`: ENCODE ChIP-seq data
  - `data/gtex/`: GTEx expression data
  - `data/reference/`: Reference genome data
  - `data/logs/`: Log files directory

## Setup

### Environment Setup
1. From the project root directory, run the test environment setup script:
```bash
chmod +x setup_test_env.sh
./setup_test_env.sh
```

This script will:
- Create/update the conda environment with required dependencies
- Setup Jupyter environment
- Run the main setup script (setup_env.sh) for PyTorch installation
- Install additional test dependencies

2. Activate the environment:
```bash
conda activate dnabert_env
```

## Running Tests

### Basic Usage
Run all tests:
```bash
pytest
```

### Common Options
- Run specific test file:
```bash
pytest tests/test_sequence_analysis.py
```

- Run tests matching a pattern:
```bash
pytest -k "sequence"
```

- Run with verbose output:
```bash
pytest -v
```

- Show print output:
```bash
pytest -s
```

- Run tests by marker:
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m "integration"  # Run only integration tests
```

### Test Categories

Tests are organized using markers:
- `slow`: Time-consuming tests (e.g., those using real data)
- `integration`: End-to-end integration tests
- `unit`: Unit tests
- `visualization`: Tests for plotting functions
- `dnabert`: Tests for DNABERT-related functionality
- `sequence`: Tests for sequence analysis

## Test Structure

### Unit Tests
- `test_sequence_analysis.py`: Tests for sequence analysis functionality
- `test_dnabert_trainer.py`: Tests for DNABERT-2 model training
- `test_visualization.py`: Tests for visualization functions
- `test_pipeline.py`: Tests for main pipeline integration

### Fixtures
Common test fixtures are defined in `conftest.py`:
- `data_generator`: Provides access to test data samples
- `project_root`: Project root directory
- `config`: Project configuration
- `temp_dir`: Temporary directory for test files
- `test_output_dir`: Directory for test outputs

### Test Data Strategy
Tests use a hierarchical data strategy:
1. Attempt to use actual project data from data/ directories
2. Fall back to pre-generated test data if available
3. Generate mock data as last resort

## Configuration

### Test Configuration Files
- `pytest.ini`: Main pytest configuration
- `config_test.yaml`: Test-specific settings and data configurations

### Coverage Configuration
Coverage settings are defined in pytest.ini:
- Branch coverage enabled
- HTML and terminal reports
- Excludes DNABERT_2 directory and test files
- Shows missing lines in reports

### Test Timeouts
- Default timeout: 300 seconds (5 minutes)
- Configurable per test with @pytest.mark.timeout decorator

## Writing Tests

### Basic Test Structure
```python
import pytest

@pytest.mark.unit
def test_function_name(data_generator):
    # Arrange
    data = data_generator.load_geo_sample()
    
    # Act
    result = perform_operation(data)
    
    # Assert
    assert result is not None
```

### Using Fixtures
```python
@pytest.mark.integration
def test_pipeline(data_generator, test_output_dir):
    # Test implementation using fixtures
    pass
```

### Parameterized Tests
```python
@pytest.mark.parametrize("input,expected", [
    ("ATGC", True),
    ("INVALID", False),
])
def test_sequence_validation(input, expected):
    assert validate_sequence(input) == expected
```

## Coverage Reports

After running tests, coverage reports are available:
- HTML: `coverage_report/index.html`
- Terminal: Shown after test execution

## Troubleshooting

### Common Issues

1. Missing Dependencies
```bash
# Update environment
conda env update -f environment.yml
```

2. Data Directory Issues
```bash
# Verify data directories
ls -R data/
```

3. Test Discovery Issues
```bash
# Run with increased verbosity
pytest --collect-only -v
```

### Getting Help
- Check test_execution.log for detailed logs
- Verify correct conda environment activation
- Check the project's issue tracker

## Best Practices

1. Test Data Management
- Prefer actual project data when available
- Use small, focused data samples
- Include fallback mock data generation

2. Test Organization
- Use appropriate markers for categorization
- Keep tests focused and independent
- Use meaningful test names

3. Performance
- Use `@pytest.mark.slow` for time-consuming tests
- Minimize file I/O in tests
- Use appropriate fixture scopes

4. Maintenance
- Keep test data up to date
- Review and update test configurations
- Maintain test coverage with new features