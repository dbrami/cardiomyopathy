"""
Tests for sequence analysis functionality
"""

import pytest
import numpy as np
from src.sequence_analysis import SequenceAnalyzer

def test_calculate_gc_content(data_generator):
    """Test GC content calculation"""
    analyzer = SequenceAnalyzer()
    
    # Test sequence with known GC content
    sequence = "GCGCTA"  # 4 GC out of 6 = 66.67%
    gc_content = analyzer.calculate_gc_content(sequence)
    assert abs(gc_content - 66.67) < 0.01
    
    # Test empty sequence
    with pytest.raises(ValueError):
        analyzer.calculate_gc_content("")

@pytest.mark.slow
def test_predict_off_targets(genome_data):
    """Test off-target prediction with real genome data"""
    analyzer = SequenceAnalyzer()
    test_sequence = str(genome_data[0].seq[:15])  # Take first 15 bases
    
    # Create test genome dictionary
    test_genome = {record.id: record.seq for record in genome_data}
    
    # Test with exact match
    off_targets = analyzer.predict_off_targets(
        test_sequence,
        test_genome,
        max_mismatches=0
    )
    
    # Should find at least one match (the original sequence)
    assert any(len(matches) > 0 for matches in off_targets.values())
    
    # Test with mismatches
    off_targets_with_mismatches = analyzer.predict_off_targets(
        test_sequence,
        test_genome,
        max_mismatches=2
    )
    
    # Should find more matches with mismatches allowed
    total_matches = sum(len(matches) for matches in off_targets_with_mismatches.values())
    assert total_matches >= sum(len(matches) for matches in off_targets.values())

def test_analyze_secondary_structure():
    """Test RNA secondary structure analysis"""
    analyzer = SequenceAnalyzer()
    test_sequence = "ATGCTAGCTAGCTAG"
    
    result = analyzer.analyze_secondary_structure(test_sequence)
    
    # Check required keys
    required_keys = ['sequence', 'mfe_structure', 'mfe', 'ensemble_energy']
    for key in required_keys:
        assert key in result
    
    # MFE should be a negative float (stable structure)
    assert isinstance(result['mfe'], float)
    assert result['mfe'] < 0

@pytest.mark.parametrize("sequence", [
    "ATGCTAGCTAGCTAG",
    "GCGCGCGCGCGCGCG",
    "TATATATATATATA"
])
def test_calculate_stability(sequence):
    """Test thermodynamic stability calculation with different sequences"""
    analyzer = SequenceAnalyzer()
    result = analyzer.calculate_stability(sequence)
    
    # Check required keys
    required_keys = ['sequence', 'gc_content', 'end_stability', 
                    'duplex_stability', 'total_stability']
    for key in required_keys:
        assert key in result
    
    # Validate values
    assert isinstance(result['gc_content'], (int, float))
    assert 0 <= result['gc_content'] <= 100
    assert isinstance(result['total_stability'], float)

@pytest.mark.integration
def test_evaluate_sequence(genome_data):
    """Test comprehensive sequence evaluation"""
    analyzer = SequenceAnalyzer()
    test_sequence = str(genome_data[0].seq[:20])  # Take first 20 bases
    test_genome = {record.id: record.seq for record in genome_data}
    
    result = analyzer.evaluate_sequence(test_sequence, test_genome)
    
    # Check main categories
    main_keys = ['sequence', 'off_targets', 'secondary_structure', 
                'stability', 'overall_score']
    for key in main_keys:
        assert key in result
    
    # Validate overall score
    assert isinstance(result['overall_score'], float)
    assert 0 <= result['overall_score'] <= 1

@pytest.mark.slow
def test_batch_sequence_analysis(genome_data):
    """Test batch analysis of multiple sequences"""
    test_sequences = [str(record.seq[:20]) for record in genome_data[:3]]
    test_genome = {record.id: record.seq for record in genome_data}
    
    results = analyzer.batch_analyze_sequences(test_sequences, test_genome)
    
    assert len(results) == len(test_sequences)
    for result in results:
        assert 'overall_score' in result
        assert 0 <= result['overall_score'] <= 1

def test_invalid_inputs():
    """Test handling of invalid inputs"""
    analyzer = SequenceAnalyzer()
    
    # Test invalid sequence
    with pytest.raises(ValueError):
        analyzer.calculate_stability("INVALID")
    
    # Test None inputs
    with pytest.raises(ValueError):
        analyzer.predict_off_targets(None, {})
    
    # Test empty genome
    with pytest.raises(ValueError):
        analyzer.evaluate_sequence("ATGC", {})

@pytest.fixture
def analyzer():
    """Fixture for SequenceAnalyzer instance"""
    return SequenceAnalyzer()