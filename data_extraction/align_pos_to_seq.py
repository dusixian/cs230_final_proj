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
with open("ADAR.txt", "r") as infile, open("ADAR_seq.txt", "w") as outfile:
    for line in infile:
        parts = line.strip().split("\t")
        seqid=parts[0]
        arm=parts[1]
        chrom = parts[2]
        strand = parts[3]
        start1 = parts[4]
        end1 = parts[5]
        loc = parts[6]
        name=parts[7]
        ann=parts[8]
       # typeid=parts[9]
       # aff=parts[10]
 # Extract sequences for each region
        sequence1 = extract_sequence(chrom, strand, start1, end1)
        
        # Recreate the original line and add the extracted sequences separately
        output_line = f"{seqid}\t{arm}\t{chrom}\t{strand}\t{start1}\t{end1}\t{sequence1}\t{loc}\t{name}\t{ann}\n"
        #output_line = f"{seqid}\t{arm}\t{chrom}\t{strand}\t{start1}\t{end1}\t{sequence1}\t{loc}\t{name}\t{ann}\t{typeid}\t{aff}\n"
        outfile.write(output_line)
