#!/bin/bash

# Exit immediately if a command fails.
set -e

# Create subdirectories for each data source
mkdir -p data/geo data/encode data/gtex data/reference

echo "Setting up data downloads..."

# GEO RNA-seq dataset (GSE55296)
echo "Downloading GEO RNA-seq dataset (GSE55296) into data/geo..."
wget -O data/geo/GSE55296_series_matrix.txt.gz \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55296/matrix/GSE55296_series_matrix.txt.gz"
gunzip -f data/geo/GSE55296_series_matrix.txt.gz

# ENCODE data note
echo "Note: ENCODE heart tissue data requires manual download from encodeproject.org:"
echo "1. Visit https://www.encodeproject.org/"
echo "2. Search for 'heart tissue ChIP-seq'"
echo "3. Download relevant BED files to data/encode/"

# GTEx data
echo "Downloading GTEx data into data/gtex/..."

# 1. GTEx RNA-seq Gene TPM Data (v8)
# Contains transcript-per-million (TPM) values for all genes across all GTEx samples
echo "Downloading GTEx RNA-seq TPM data..."
wget -O data/gtex/GTEx_Analysis_v8_RNA-seq_RNA-SeQCv1.1.9_gene_tpm.gct.gz \
    "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_v8_RNA-seq_RNA-SeQCv1.1.9_gene_tpm.gct.gz"

# 2. GTEx Sample Attributes
# Provides metadata for each sample (including tissue type)
echo "Downloading GTEx sample attributes..."
wget -O data/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt \
    "https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"

# 3. GTEx Subject Phenotypes
# Contains phenotypic data for GTEx donors
echo "Downloading GTEx subject phenotypes..."
wget -O data/gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt \
    "https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"

echo "Note: Filter GTEx data for heart tissues using Sample Tissue Description (SMTSD):"
echo "- 'Heart - Left Ventricle'"
echo "- 'Heart - Atrial Appendage'"

# Reference genome
echo "Downloading human genome reference (GRCh38) into data/reference..."
wget -O data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
    "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"

echo
echo "Downloads completed!"
echo
echo "Data directories populated:"
echo "- data/geo/: GEO RNA-seq data (GSE55296 series matrix)"
echo "- data/encode/: For ENCODE ChIP-seq data (manual download required)"
echo "- data/gtex/: GTEx data:"
echo "  * RNA-seq TPM data (~1-2 GB)"
echo "  * Sample attributes (tissue information)"
echo "  * Subject phenotypes"
echo "- data/reference/: Reference genome (GRCh38)"
echo
echo "Next steps:"
echo "1. Download ENCODE heart tissue ChIP-seq data from encodeproject.org"
echo "2. Process GTEx data to extract heart-specific samples using the Sample Attributes file"
