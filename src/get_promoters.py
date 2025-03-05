import os
import csv
import time
import io
from Bio import Entrez, SeqIO
from tqdm import tqdm

# Set your email address for NCBI (required by NCBI)
Entrez.email = "your.email@example.com"

def fetch_promoter_by_gene_symbol(gene_symbol, promoter_length=1000, organism="Homo sapiens"):
    """
    Fetch the promoter sequence for a gene given its symbol.
    The function searches the NCBI Gene database for the gene symbol,
    extracts genomic location info, and then fetches the promoter region
    from the nucleotide database.
    
    Returns:
      promoter_seq (Bio.Seq): The promoter sequence.
      accession (str): The nucleotide accession used.
      strand (int): The gene strand (1 for plus, -1 for minus).
    """
    # Search the gene database using the gene symbol
    query = f"{gene_symbol}[sym] AND {organism}[orgn]"
    try:
        search_handle = Entrez.esearch(db="gene", term=query)
        search_results = Entrez.read(search_handle)
        search_handle.close()
    except Exception as e:
        tqdm.write(f"Error searching for gene {gene_symbol}: {e}")
        return None, None, None

    if not search_results["IdList"]:
        tqdm.write(f"No gene found for {gene_symbol}")
        return None, None, None
    gene_id = search_results["IdList"][0]

    # Fetch gene summary to get genomic location
    try:
        summary_handle = Entrez.esummary(db="gene", id=gene_id)
        summary = Entrez.read(summary_handle)
        summary_handle.close()
    except Exception as e:
        tqdm.write(f"Error fetching summary for gene {gene_symbol}: {e}")
        return None, None, None

    try:
        # The gene summary is in a nested structure
        gene_summary = summary["DocumentSummarySet"]["DocumentSummary"][0]
    except (KeyError, IndexError) as e:
        tqdm.write(f"Error parsing summary for gene {gene_symbol}: {e}")
        return None, None, None

    genomic_info = gene_summary.get("GenomicInfo")
    if not genomic_info:
        tqdm.write(f"No genomic info found for {gene_symbol}")
        return None, None, None

    # Use the first available genomic record
    gi = genomic_info[0]
    accession = gi.get("ChrAccVer")
    if accession is None:
        tqdm.write(f"No accession found in genomic info for {gene_symbol}")
        return None, None, None
    chr_start = int(gi.get("ChrStart", 0))
    chr_stop = int(gi.get("ChrStop", 0))
    
    # Attempt to fetch strand information from multiple possible keys.
    if "ChrStrand" in gi:
        strand = int(gi["ChrStrand"])
    elif "strand" in gi:
        strand = int(gi["strand"])
    elif "Strand" in gene_summary:
        strand = int(gene_summary["Strand"])
    else:
        tqdm.write(f"Strand information not found for gene {gene_symbol}. Assuming plus strand.")
        strand = 1

    # Compute promoter region coordinates
    if strand == 1:
        # For plus strand, promoter is upstream of the gene's start.
        promoter_start = max(chr_start - promoter_length, 0)
        promoter_end = chr_start
        # efetch expects 1-indexed positions
        fetch_start = promoter_start + 1
        fetch_end = promoter_end
    elif strand == -1:
        # For minus strand, promoter is downstream of the gene's end.
        promoter_start = chr_stop
        promoter_end = chr_stop + promoter_length
        fetch_start = promoter_start + 1
        fetch_end = promoter_end
    else:
        tqdm.write(f"Unknown strand value for gene {gene_symbol}.")
        return None, None, None

    # Fetch the genomic region from the nucleotide database
    try:
        fetch_handle = Entrez.efetch(
            db="nucleotide",
            id=accession,
            rettype="fasta",
            retmode="text",
            seq_start=fetch_start,
            seq_stop=fetch_end
        )
        seq_data = fetch_handle.read()
        fetch_handle.close()
    except Exception as e:
        tqdm.write(f"Error fetching nucleotide record for gene {gene_symbol}: {e}")
        return None, None, None

    fasta_io = io.StringIO(seq_data)
    record = next(SeqIO.parse(fasta_io, "fasta"), None)
    if record is None:
        tqdm.write(f"Could not parse sequence for gene {gene_symbol}")
        return None, None, None

    promoter_seq = record.seq
    # For minus strand, take the reverse complement
    if strand == -1:
        promoter_seq = promoter_seq.reverse_complement()

    return promoter_seq, accession, strand

def process_genes(csv_file, output_fasta, gene_id_column="gene_id", promoter_length=1000, delay=0.5):
    """
    Process gene symbols from a CSV file, fetch their promoter sequences,
    and write the results to a FASTA file.
    """
    fasta_entries = []
    
    # Read all rows from the CSV
    with open(csv_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    # Process each gene with a progress bar
    for row in tqdm(rows, desc="Processing Genes"):
        gene_symbol = row.get(gene_id_column)
        if gene_symbol is None:
            raise KeyError(f"CSV row is missing the column '{gene_id_column}'. Available columns: {list(row.keys())}")
        tqdm.write(f"Processing gene: {gene_symbol}")
        promoter_seq, accession, strand = fetch_promoter_by_gene_symbol(gene_symbol, promoter_length)
        if promoter_seq:
            header = f">{gene_symbol} {accession} Strand:{strand}"
            fasta_entries.append(f"{header}\n{promoter_seq}\n")
            tqdm.write(f"Fetched promoter for gene {gene_symbol}")
        else:
            tqdm.write(f"Could not fetch promoter for gene {gene_symbol}")
        time.sleep(delay)  # Respect NCBI's rate limits
    
    # Write all promoter sequences to the FASTA file
    with open(output_fasta, "w") as fasta_file:
        fasta_file.write("".join(fasta_entries))
    tqdm.write(f"Promoter sequences saved to {output_fasta}")

if __name__ == "__main__":
    # Build file paths relative to the script's location.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_file = os.path.join(script_dir, "../data/promoters/genes.csv")
    fasta_file = os.path.join(script_dir, "../data/promoters/promoter_sequences.fasta")
    
    process_genes(csv_file, fasta_file)