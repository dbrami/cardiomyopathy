#!/bin/bash

# Exit immediately if a command fails
set -e

# Check if yq is installed (for YAML parsing)
if ! command -v yq &> /dev/null; then
    echo "yq is required but not installed. Please install it:"
    echo "On macOS: brew install yq"
    echo "On Linux: snap install yq"
    exit 1
fi

# Create log file and directories
data_dir=$(yq '.directories.data' config.yaml)
geo_dir=$(yq '.directories.geo' config.yaml)
encode_dir=$(yq '.directories.encode' config.yaml)
gtex_dir=$(yq '.directories.gtex' config.yaml)
reference_dir=$(yq '.directories.reference' config.yaml)
logs_dir=$(yq '.directories.logs' config.yaml)

mkdir -p "$geo_dir" "$encode_dir" "$gtex_dir" "$reference_dir" "$logs_dir"
LOG_FILE="${logs_dir}/download_log_$(date +%Y%m%d_%H%M%S).txt"

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
    local is_compressed="$4"

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
            
            # If compressed and ends with .gz, extract it
            if [ "$is_compressed" = "true" ] && [[ "$output" == *.gz ]]; then
                log_message "Extracting $output..."
                gunzip -f "$output"
                log_message "Extraction complete"
            fi
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

# GEO RNA-seq datasets (GSE55296)
# Download series matrix
geo_matrix_url=$(yq '.files.geo.series_matrix.url' config.yaml)
geo_matrix_file="${geo_dir}/$(yq '.files.geo.series_matrix.filename' config.yaml)"
geo_matrix_compressed=$(yq '.files.geo.series_matrix.compressed' config.yaml)
download_file "$geo_matrix_url" "$geo_matrix_file.gz" "GEO RNA-seq series matrix" "$geo_matrix_compressed"

# Download counts data
geo_counts_url=$(yq '.files.geo.counts.url' config.yaml)
geo_counts_file="${geo_dir}/$(yq '.files.geo.counts.filename' config.yaml)"
geo_counts_compressed=$(yq '.files.geo.counts.compressed' config.yaml)
download_file "$geo_counts_url" "$geo_counts_file.gz" "GEO RNA-seq counts data" "$geo_counts_compressed"

# GTEx data
gtex_tpm_url=$(yq '.files.gtex.tpm_data.url' config.yaml)
gtex_tpm_file="${gtex_dir}/$(yq '.files.gtex.tpm_data.filename' config.yaml)"
gtex_tpm_compressed=$(yq '.files.gtex.tpm_data.compressed' config.yaml)
download_file "$gtex_tpm_url" "$gtex_tpm_file" "GTEx RNA-seq TPM data" "$gtex_tpm_compressed"

# Sample Attributes
sample_attr_url=$(yq '.files.gtex.sample_attributes.url' config.yaml)
sample_attr_file="${gtex_dir}/$(yq '.files.gtex.sample_attributes.filename' config.yaml)"
sample_attr_compressed=$(yq '.files.gtex.sample_attributes.compressed' config.yaml)
download_file "$sample_attr_url" "$sample_attr_file" "GTEx sample attributes" "$sample_attr_compressed"

# Subject Phenotypes
subject_phen_url=$(yq '.files.gtex.subject_phenotypes.url' config.yaml)
subject_phen_file="${gtex_dir}/$(yq '.files.gtex.subject_phenotypes.filename' config.yaml)"
subject_phen_compressed=$(yq '.files.gtex.subject_phenotypes.compressed' config.yaml)
download_file "$subject_phen_url" "$subject_phen_file" "GTEx subject phenotypes" "$subject_phen_compressed"

# Reference genome
ref_url=$(yq '.files.reference.genome.url' config.yaml)
ref_file="${reference_dir}/$(yq '.files.reference.genome.filename' config.yaml)"
ref_compressed=$(yq '.files.reference.genome.compressed' config.yaml)
download_file "$ref_url" "$ref_file" "Reference genome" "$ref_compressed"

# ENCODE data note
log_message "----------------------------------------"
log_message "ENCODE data requires manual download:"
encode_desc=$(yq '.files.encode.description' config.yaml)
echo "$encode_desc" | while IFS= read -r line; do
    log_message "$line"
done

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
echo "- $geo_dir/: GEO RNA-seq data"
echo "- $encode_dir/: Requires manual download"
echo "- $gtex_dir/: GTEx data files ($(yq '.files.gtex.version' config.yaml) RNA-seq TPM + annotations)"
echo "- $reference_dir/: Reference genome"
echo "- $logs_dir/: Download logs"
echo
echo "For detailed download status and existing files list, check: $LOG_FILE"
