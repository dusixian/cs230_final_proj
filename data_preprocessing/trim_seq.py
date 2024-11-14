import re

def parse_rnahybrid_output(file_path):
    """
    Parse the RNAhybrid output to extract the unpaired bases at the ends of the sequences.
    """
    substrates = {}
    
    with open(file_path, 'r') as file:
        rnahybrid_output = file.read()
    
    rows = rnahybrid_output.strip().split('\n\n')
    
    for row in rows:
        header = row.splitlines()[0]
        substrate_name = header.split(":")[0][1:]
        
        # Extract the target and miRNA sequences and calculate unpaired regions
        target_seq = ''
        miRNA_seq = ''
        for line in row.splitlines():
            if line.startswith('target'):
                target_seq = line.split()[1].replace(" ", "")  # Target sequence
            elif line.startswith('miRNA'):
                miRNA_seq = line.split()[1].replace(" ", "")  # miRNA sequence
        
        # Calculate the number of unpaired bases at the start and end
        left_unpaired = len(re.findall(r"^[AUGC]+", target_seq))  # Left unpaired bases
        right_unpaired = len(re.findall(r"[AUGC]+$", miRNA_seq))  # Right unpaired bases
        
        # Store the unpaired information for the substrate
        substrates[substrate_name] = (left_unpaired, right_unpaired)
    
    return substrates

def parse_chromosome_positions(chromosome_position_data):
    """
    Parse the chromosome position file to extract the left and right substrate positions.
    """
    position_data = []
    
    rows = chromosome_position_data.strip().split('\n')
    
    for row in rows:
        parts = row.split('\t')
        chrom = parts[0]
        strand = parts[1]
        left_start = int(parts[2])
        left_end = int(parts[3])
        right_start = int(parts[4])
        right_end = int(parts[5])
        substrate_id = parts[6]
        
        position_data.append({
            'chrom': chrom,
            'strand': strand,
            'left_start': left_start,
            'left_end': left_end,
            'right_start': right_start,
            'right_end': right_end,
            'substrate_id': substrate_id
        })
    
    return position_data

def adjust_positions_with_unpaired_length(rnahybrid_file, chromosome_position_data):
    """
    Adjust the start and end positions of left and right sequences based on unpaired regions from RNAhybrid.
    """
    # Step 1: Parse RNAhybrid output and chromosome position data
    rnahybrid_info = parse_rnahybrid_output(rnahybrid_file)
    chromosome_positions = parse_chromosome_positions(chromosome_position_data)
    
    final_output = []

    for position in chromosome_positions:
        substrate_id = position['substrate_id']
        left_start = position['left_start']
        left_end = position['left_end']
        right_start = position['right_start']
        right_end = position['right_end']
        substrate_num = substrate_counter
        if substrate_id in rnahybrid_info:
            left_unpaired, right_unpaired = rnahybrid_info[substrate_id]
            
            # Adjust the positions by trimming the unpaired bases
            adjusted_left_start = left_start + left_unpaired
            adjusted_left_end = left_end - left_unpaired
            adjusted_right_start = right_start + right_unpaired
            adjusted_right_end = right_end - right_unpaired
            
            # Output the adjusted positions for both left and right sequences
            final_output.append(f"{Substrate_{substrate_id}\tL\tposition['chrom']}\t{position['strand']}\t{adjusted_left_start}\t{adjusted_left_end}\t")
            final_output.append(f"{Substrate_{substrate_id}\tR\tposition['chrom']}\t{position['strand']}\t{adjusted_right_start}\t{adjusted_right_end}\t")
    
    return "\n".join(final_output)

# File paths
rnahybrid_file = 'ADAR_RNAhybrid.txt'  # RNAhybrid output file
chromosome_position_file = 'pos_final.txt'  # Input chromosome position file

# Read the chromosome position data
with open(chromosome_position_file, 'r') as file:
    chromosome_position_data = file.read()

# Running the function to adjust positions
final_output = adjust_positions_with_unpaired_length(rnahybrid_file, chromosome_position_data)
# Optionally, save the output to a file
 with open('final_pos_output.txt', 'w') as out_file:
     out_file.write(final_output)

