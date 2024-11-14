def sam_to_custom_format(sam_file, output_file):
    with open(sam_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            parts = line.strip().split("\t")
            chr = parts[2]
            flag = int(parts[1])
            strand = '-' if flag & 16 else '+'
            start_point = int(parts[3])
            end_point = start_point + len(parts[9])  # Assume the end point is start + length of the sequence
            id = parts[0]
            
            outfile.write(f"{chr}\t{strand}\t{start_point}\t{end_point}\t{id}\n")

# Example usage
sam_file = "test.sam"  # Replace with your SAM file path
output_file = "nonhybrid_test.txt"  # Replace with your desired output file path
sam_to_custom_format(sam_file, output_file)

