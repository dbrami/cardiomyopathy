"""
Sequence Analysis Module for siRNA Pipeline

This module provides functionality for analyzing siRNA sequences, including:
- Off-target prediction
- RNA secondary structure analysis
- Thermodynamic stability calculations
"""

import numpy as np
import pandas as pd
import logging
from Bio import SeqIO, Seq
from Bio.SeqUtils import gc_fraction as GC
import RNA  # Vienna RNA package
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

class SequenceAnalyzer:
    """Analyze siRNA sequences for efficacy and safety"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @staticmethod
    def calculate_gc_content(sequence):
        """Calculate GC content of a sequence"""
        return GC(Seq.Seq(sequence))
        
    def predict_off_targets(self, sequence, genome_ref,
                           max_mismatches=3, seed_region=2):
        """
        Predict potential off-target effects
        
        Args:
            sequence (str): siRNA sequence to analyze
            genome_ref (dict): Reference genome sequences
            max_mismatches (int): Maximum allowed mismatches
            seed_region (int): Seed region length to consider
            
        Returns:
            dict: Off-target prediction results
            
        Raises:
            ValueError: If input sequence or genome reference is invalid
        """
        try:
            if sequence is None:
                raise ValueError("Sequence cannot be None")
                
            if not isinstance(sequence, str):
                raise ValueError("Sequence must be a string")
                
            if not sequence:
                raise ValueError("Sequence cannot be empty")
                
            if not genome_ref or not isinstance(genome_ref, dict):
                raise ValueError("Invalid genome reference")
                
            off_targets = defaultdict(list)
            sequence = sequence.upper()
            seed = sequence[:seed_region]
            
            for chrom, seq in genome_ref.items():
                # Find potential binding sites starting with seed match
                for match in re.finditer(seed, str(seq), re.IGNORECASE):
                    start = match.start()
                    potential_target = str(seq[start:start + len(sequence)])
                    
                    if len(potential_target) == len(sequence):
                        mismatches = sum(a != b for a, b in 
                                       zip(sequence, potential_target))
                        
                        if mismatches <= max_mismatches:
                            off_targets[chrom].append({
                                'position': start,
                                'mismatches': mismatches,
                                'sequence': potential_target
                            })
            
            return dict(off_targets)
            
        except Exception as e:
            self.logger.error(f"Error in off-target prediction: {e}")
            raise
            
    def analyze_secondary_structure(self, sequence):
        """
        Analyze RNA secondary structure using Vienna RNA package
        
        Args:
            sequence (str): RNA sequence to analyze
            
        Returns:
            dict: Secondary structure analysis results
        """
        try:
            # Convert DNA to RNA sequence
            rna_seq = sequence.replace('T', 'U')
            
            # Calculate minimum free energy and structure
            mfe_struct, mfe = RNA.fold(rna_seq)
            
            # Calculate ensemble diversity
            _, ensemble_energy = RNA.pf_fold(rna_seq)
            
            # Calculate base pairing probability matrix
            seq_length = len(rna_seq)
            bpp = [[RNA.get_pr(i, j) for j in range(seq_length)] for i in range(seq_length)]
            
            return {
                'sequence': rna_seq,
                'mfe_structure': mfe_struct,
                'mfe': mfe,
                'ensemble_energy': ensemble_energy,
                'base_pair_probs': bpp
            }
            
        except Exception as e:
            self.logger.error(f"Error in secondary structure analysis: {e}")
            raise
            
    def calculate_stability(self, sequence):
        """
        Calculate thermodynamic stability parameters
        
        Args:
            sequence (str): siRNA sequence
            
        Returns:
            dict: Stability analysis results
            
        Raises:
            ValueError: If sequence is invalid
        """
        try:
            # Validate sequence
            if not sequence or not isinstance(sequence, str):
                raise ValueError("Sequence must be a non-empty string")
                
            # Check for valid nucleotides
            valid_nucleotides = set('ATGC')
            if not all(nucleotide in valid_nucleotides for nucleotide in sequence.upper()):
                raise ValueError("Sequence contains invalid nucleotides. Only A, T, G, C are allowed.")
            
            # Calculate basic sequence properties
            gc_content = self.calculate_gc_content(sequence)
            
            # Calculate thermodynamic parameters
            dg_end = self._calculate_end_stability(sequence)
            dg_duplex = self._calculate_duplex_stability(sequence)
            
            return {
                'sequence': sequence,
                'gc_content': gc_content,
                'end_stability': dg_end,
                'duplex_stability': dg_duplex,
                'total_stability': dg_end + dg_duplex
            }
            
        except Exception as e:
            self.logger.error(f"Error in stability calculation: {e}")
            raise
            
    def _calculate_end_stability(self, sequence, window_size=4):
        """Calculate stability of sequence ends"""
        try:
            # Get terminal sequences
            five_prime = sequence[:window_size]
            three_prime = sequence[-window_size:]
            
            # Simple scoring based on GC content of ends
            five_prime_score = self.calculate_gc_content(five_prime) / 100.0
            three_prime_score = self.calculate_gc_content(three_prime) / 100.0
            
            return (five_prime_score + three_prime_score) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating end stability: {e}")
            raise
            
    def _calculate_duplex_stability(self, sequence):
        """Calculate stability of siRNA duplex"""
        try:
            # Convert to RNA
            rna_seq = sequence.replace('T', 'U')
            
            # Create complement sequence
            complement = str(Seq.Seq(rna_seq).complement())
            
            # Calculate duplex minimum free energy
            duplex_struct, duplex_mfe = RNA.cofold(f"{rna_seq}&{complement}")
            
            return duplex_mfe
            
        except Exception as e:
            self.logger.error(f"Error calculating duplex stability: {e}")
            raise
            
    def evaluate_sequence(self, sequence, genome_ref):
        """
        Comprehensive evaluation of siRNA sequence
        
        Args:
            sequence (str): siRNA sequence to evaluate
            genome_ref (dict): Reference genome sequences
            
        Returns:
            dict: Complete evaluation results
        """
        try:
            results = {
                'sequence': sequence,
                'length': len(sequence)
            }
            
            # Off-target analysis
            results['off_targets'] = self.predict_off_targets(
                sequence, genome_ref
            )
            
            # Secondary structure analysis
            results['secondary_structure'] = self.analyze_secondary_structure(
                sequence
            )
            
            # Stability analysis
            results['stability'] = self.calculate_stability(sequence)
            
            # Calculate overall score
            results['overall_score'] = self._calculate_overall_score(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in sequence evaluation: {e}")
            raise
            
    def _calculate_overall_score(self, results):
        """Calculate overall sequence quality score"""
        try:
            scores = []
            
            # Off-target score (fewer is better)
            off_target_count = sum(len(v) for v in results['off_targets'].values())
            off_target_score = 1.0 / (1.0 + off_target_count)
            scores.append(off_target_score)
            
            # Stability score
            stability_score = results['stability']['total_stability']
            normalized_stability = 1.0 / (1.0 + abs(stability_score))
            scores.append(normalized_stability)
            
            # GC content score (optimal around 50%)
            gc_score = 1.0 - abs(results['stability']['gc_content'] - 50) / 50
            scores.append(gc_score)
            
            # Calculate weighted average
            weights = [0.4, 0.3, 0.3]  # Weights for each component
            overall_score = sum(s * w for s, w in zip(scores, weights))
            
            return overall_score
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            raise

def batch_analyze_sequences(sequences, genome_ref, n_jobs=1):
    """
    Analyze multiple sequences in parallel
    
    Args:
        sequences (list): List of sequences to analyze
        genome_ref (dict): Reference genome sequences
        n_jobs (int): Number of parallel jobs
        
    Returns:
        pd.DataFrame: Analysis results for all sequences
    """
    try:
        analyzer = SequenceAnalyzer()
        results = []
        
        for seq in sequences:
            result = analyzer.evaluate_sequence(seq, genome_ref)
            results.append(result)
            
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error in batch sequence analysis: {e}")
        raise