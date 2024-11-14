# Define input and output file paths
input_file = "GSM4045976_ADAR1-irCLASH-rep1-hybrid-reads_seq.txt"
output_file1 = "test_arm1.fasta"
output_file2 = "test_arm2.fasta"

# Function to write sequences to a FASTA file
def write_fasta(output_file, sequence_id, sequence):
    with open(output_file, "a") as f:
        f.write(f">{sequence_id}\n{sequence}\n")

# Process the input file and extract sequences
with open(input_file, "r") as infile:
    for line in infile:
        columns = line.strip().split("\t")
        if len(columns) >= 10:
            sequence_id = columns[6]  # Sequence ID
            sequence1 = columns[8]    # Sequence 1
            sequence2 = columns[9]    # Sequence 2

            # Write sequences to respective FASTA files
            write_fasta(output_file1, sequence_id, sequence1)
            write_fasta(output_file2, sequence_id, sequence2)

print("FASTA files created successfully!")
