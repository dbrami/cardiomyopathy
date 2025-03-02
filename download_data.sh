#!/bin/bash

# Exit immediately if a command fails.
set -e

# Create log file and directories
mkdir -p data/geo data/encode data/gtex data/reference data/logs
LOG_FILE="data/logs/download_log_$(date +%Y%m%d_%H%M%S).txt"

# Arrays to store file statuses
declare -a SKIPPED_FILES=()
declare -a SUCCESS_FILES=()
declare -a FAILED_FILES=()

# Function to log messages
log_message() {
    echo "$1" | tee -a "$LOG_FILE"
}

# Function to log download status
log_download() {
    local file="$1"
    local url="$2"
    local status="$3"
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $file | $url | $status" >> "$LOG_FILE"
    
    # Store files in appropriate arrays
    case "$status" in
        "SKIPPED (already exists)")
            SKIPPED_FILES+=("$file")
            ;;
        "SUCCESS")
            SUCCESS_FILES+=("$file")
            ;;
        "FAILED")
            FAILED_FILES+=("$file")
            ;;
    esac
}

# Function to download file if it doesn't exist
download_file() {
    local url="$1"
    local output="$2"
    local description="$3"

    if [ -f "$output" ]; then
        log_download "$output" "$url" "SKIPPED (already exists)"
        log_message "File already exists: $output"
    else
        log_message "Downloading $description..."
        echo "URL: $url"
        echo "Output: $output"
        echo "----------------------------------------"
        if wget --progress=bar:force:noscroll --show-progress -O "$output" "$url" 2>&1; then
            echo "----------------------------------------"
            log_download "$output" "$url" "SUCCESS"
            log_message "Successfully downloaded: $output"
        else
            echo "----------------------------------------"
            log_download "$output" "$url" "FAILED"
            log_message "Failed to download: $output"
            return 1
        fi
    fi
}

# Function to print array contents with bullet points
print_array() {
    local arr=("$@")
    for item in "${arr[@]}"; do
        echo "  - $item"
    done
}

log_message "Starting downloads at $(date)"
log_message "----------------------------------------"

# GEO RNA-seq dataset (GSE55296)
GEO_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55296/matrix/GSE55296_series_matrix.txt.gz"
download_file "$GEO_URL" "data/geo/GSE55296_series_matrix.txt.gz" "GEO RNA-seq dataset" && \
    gunzip -f "data/geo/GSE55296_series_matrix.txt.gz"

# GTEx data
# RNA-seq TPM Data (v10)
GTEX_TPM_URL="https://storage.googleapis.com/adult-gtex/bulk-gex/v10/rna-seq/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz"
download_file "$GTEX_TPM_URL" "data/gtex/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz" "GTEx RNA-seq TPM data"

# Sample Attributes (v10)
GTEX_SAMPLE_URL="https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt"
download_file "$GTEX_SAMPLE_URL" "data/gtex/GTEx_Analysis_v10_Annotations_SampleAttributesDS.txt" "GTEx sample attributes"

# Subject Phenotypes (v10)
GTEX_PHENOTYPE_URL="https://storage.googleapis.com/adult-gtex/annotations/v10/metadata-files/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt"
download_file "$GTEX_PHENOTYPE_URL" "data/gtex/GTEx_Analysis_v10_Annotations_SubjectPhenotypesDS.txt" "GTEx subject phenotypes"

# Reference genome
REF_URL="https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
download_file "$REF_URL" "data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz" "Reference genome"

# ENCODE data note
log_message "----------------------------------------"
log_message "ENCODE data requires manual download:"
log_message "1. Visit https://www.encodeproject.org/"
log_message "2. Search for 'heart tissue ChIP-seq'"
log_message "3. Download relevant BED files to data/encode/"
ENCODE_URL="https://www.encodeproject.org/search/?type=Experiment&status=released&searchTerm=heart+tissue&assay_title=Histone+ChIP-seq&biosample_ontology.term_name=heart+left+ventricle&files.file_type=bed+narrowPeak"
# save the files.txt into 'data/encode/' and then download them using 'xargs -n 1 curl -O -L < files.txt'

# Print summary
log_message "----------------------------------------"
log_message "Download Summary:"
log_message "Log file created at: $LOG_FILE"
log_message ""

if [ ${#SKIPPED_FILES[@]} -gt 0 ]; then
    log_message "Files already present (no download needed):"
    print_array "${SKIPPED_FILES[@]}" | tee -a "$LOG_FILE"
    log_message ""
fi

if [ ${#SUCCESS_FILES[@]} -gt 0 ]; then
    log_message "Successfully downloaded files:"
    print_array "${SUCCESS_FILES[@]}" | tee -a "$LOG_FILE"
    log_message ""
fi

if [ ${#FAILED_FILES[@]} -gt 0 ]; then
    log_message "Failed downloads:"
    print_array "${FAILED_FILES[@]}" | tee -a "$LOG_FILE"
    log_message ""
fi

echo
echo "Data directories status:"
echo "- data/geo/: GEO RNA-seq data"
echo "- data/encode/: Requires manual download"
echo "- data/gtex/: GTEx data files (v10 RNA-seq TPM + annotations)"
echo "- data/reference/: Reference genome"
echo "- data/logs/: Download logs"
echo
echo "For detailed download status and existing files list, check: $LOG_FILE"
