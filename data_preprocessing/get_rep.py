import pandas as pd
import subprocess

# Load your substrate dataset
hybrid_reads_df = pd.read_csv("/home/groups/xjgao/irClash/data/test_seq.txt",sep='\t',header=None,names=["substrate","arm","chromosome", "strand","start", "end", "sequence"])  # Assuming columns: chromosome, start, end, sequence

# Load RepeatMasker data
rmsk_df = pd.read_csv("/home/groups/xjgao/irClash/data/fastq/rmsk.txt", sep="\t", header=None, names=[
    "version", "sw_score", "perc_subst", "perc_insertion", "perc_deletion", "chromosome", 
    "start", "end", "offset", "strand", "motif", "class", "subclass", "subfamily", 
    "align_start", "align_end", "score", "repeat_id"
])

# Filter Alu elements and other repetitive elements based on subclass and subfamily
rmsk_df["subfamily"] = rmsk_df["subfamily"].astype(str)
def categorize_repeat(row):
    if 'Alu' in row['subclass']:
        return 'Alu'
    elif row['subclass'] in ['LINE', 'SINE', 'LTR', 'Satellite', 'DNA']:
        return 'Non-Alu Repetitive'
    else:
        return 'Non-Repetitive'
rmsk_df['repeat_feature'] = rmsk_df.apply(categorize_repeat, axis=1)
rmsk_df = rmsk_df[['chrom', 'start', 'end', 'repeat_feature']]

# Load ENSEMBL GTF file (replace with your actual file path)
gtf_file = "/home/groups/xjgao/irClash/data/fastq/Homo_sapiens.GRCh37.75.gtf.gz"
gtf_df = pd.read_csv(gtf_file, sep="\t", comment="#", header=None)
gtf_df.columns = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]

# Define the feature precedence for genic locations
feature_precedence = {
    "protein_coding": ["gene"],  # Highest priority
    "lncRNA": ["lincRNA", "processed_transcript", "antisense", "sense_intronic", "sense_overlapping", "3′ prime_overlapping_ncrna", "pseudogene", "transcribed_processed_pseudogene", "unprocessed_pseudogene", "transcribed_unprocessed_pseudogene"],
    "miRNA": ["miRNA"],
    "rRNA": ["rRNA", "MT_rRNA"],
    "tRNA": ["tRNA", "MT_tRNA"],
    "ncRNA": ["other ncRNA"],
    "intergenic": ["intergenic"]
}

# Filter by gene features and assign genetic location
def filter_by_feature(row):
    if row["feature"] == "gene":
        # Extract gene name from the attributes column
        attributes = row["attribute"]
        gene_name = [attr.split('"')[1] for attr in attributes.split(";") if "gene_name" in attr]
        gene_name = gene_name[0] if gene_name else None
        
        # Assign categories based on feature precedence
        if any(lc in row["source"] for lc in feature_precedence["protein_coding"]):
            return "intron", gene_name  # Protein-coding genes are categorized as 'intron'
        if any(lc in row["source"] for lc in feature_precedence["lncRNA"]):
            return "lncRNA", gene_name
        if any(lc in row["source"] for lc in feature_precedence["miRNA"]):
            return "miRNA", gene_name
        if any(lc in row["source"] for lc in feature_precedence["rRNA"]):
            return "rRNA", gene_name
        if any(lc in row["source"] for lc in feature_precedence["tRNA"]):
            return "tRNA", gene_name
        if "other" in row["source"]:
            return "ncRNA", gene_name
        else:
            return "intergenic", gene_name
    return "none", None

def get_repeat_feature_for_read(row):
    chrom = row["chromosome"]
    start = row["start"]
    end = row["end"]

    # Find matching repeats in rmsk_df (within the region)
    matching_repeats = rmsk_df[(rmsk_df["chrom"] == chrom) & 
                               (rmsk_df["start"] <= end) & 
                               (rmsk_df["end"] >= start)]
    
    if not matching_repeats.empty:
        # Assign the most relevant repeat feature (Alu > Non-Alu Repetitive > Non-Repetitive)
        best_match = matching_repeats.iloc[0]  
        return best_match["repeat_feature"]
    else:
        return "Non-Repetitive"  # Default to non-repetitive if no match is found

# Apply the repeat feature assignment to the hybrid reads
hybrid_reads_df["repeat_feature"] = hybrid_reads_df.apply(get_repeat_feature_for_read, axis=1)
# Apply filtering function to GTF data
hybrid_reads_df["genetic_location"], hybrid_reads_df["gene_name"] = zip(*hybrid_reads_df.apply(filter_by_feature, axis=1))
output_file = "test_anno.csv"
hybrid_reads_df.to_csv(output_file, index=False)
