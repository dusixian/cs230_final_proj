from Bio import SeqIO

# Load the reference genome into a dictionary
genome = SeqIO.to_dict(SeqIO.parse("/scratch/users/dhy/irClash/data/fastq/bowtie_index/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa", "fasta"))

def extract_sequence(chrom, strand, start, end):
    # Convert positions to zero-based index
    start = int(start) - 1
    end = int(end)
    sequence = genome[chrom].seq[start:end]
    if strand == '-':
        sequence = sequence.reverse_complement()
    return str(sequence)

# Process the input file
with open("GSM4045976_ADAR1-irCLASH-rep1-hybrid-reads.txt", "r") as infile, open("GSM4045976_ADAR1-irCLASH-rep1-hybrid-reads_seq.txt", "w") as outfile:
    for line in infile:
        parts = line.strip().split("\t")
        chrom = parts[0]
        strand = parts[1]
        start1 = parts[2]
        end1 = parts[3]
        start2 = parts[4]
        end2 = parts[5]
        
        # Extract sequences for each region
        sequence1 = extract_sequence(chrom, strand, start1, end1)
        sequence2 = extract_sequence(chrom, strand, start2, end2)
        
        # Recreate the original line and add the extracted sequences separately
        output_line = f"{chrom}\t{strand}\t{start1}\t{end1}\t{start2}\t{end2}\t{parts[6]}\t{parts[7]}\t{sequence1}\t{sequence2}\n"
        outfile.write(output_line)
