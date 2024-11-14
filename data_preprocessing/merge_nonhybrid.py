import pandas as pd

def read_file(file_path):
    columns = ['chromosome', 'strand', 'start1', 'end1', 'start2', 'end2', 'id', 'sequence']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns,index_col=False)
    return df.sort_values(by=['chromosome', 'strand', 'start1', 'start2']).reset_index(drop=True)

def read_nonhybrid(file_path):
    columns = ['chromosome', 'strand', 'start', 'end', 'info']
    df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
    return df.sort_values(by=['chromosome', 'strand', 'start']).reset_index(drop=True)

def merge_and_extend_positions(txt_file, sam_file, output_file):
    regions = read_nonhybrid(txt_file)
    sam_entries = read_file(sam_file)
    
    for index,sam_row in sam_entries.iterrows():
        chrom = sam_row['chromosome']
        strand = sam_row['strand']
        start1 = sam_row['start1']
        end1 = sam_row['end1']
        start2 = sam_row['start2']
        end2 = sam_row['end2']

        # Filter the regions DataFrame to find overlaps on the same chromosome and strand
        overlapping_regions = regions[(regions['chromosome'] == chrom) & 
                                      (regions['strand'] == strand)]
        #print(overlapping_regions)
        if not overlapping_regions.empty:
            overlapping_regions_filtered1 = overlapping_regions[(overlapping_regions['start'] <= end1) & (overlapping_regions['end'] >= start1)]
            if not overlapping_regions_filtered1.empty:
                    # Extend start1 and end1
                sam_entries.at[index, 'start1'] = min(start1, overlapping_regions_filtered1['start'].min())
                sam_entries.at[index, 'end1'] = max(end1, overlapping_regions_filtered1['end'].max())
            
            overlapping_regions_filtered2 = overlapping_regions[(overlapping_regions['start'] <= end2) & (overlapping_regions['end'] >= start2)]
            if not overlapping_regions_filtered2.empty:
                    # Extend start2 and end2
                sam_entries.at[index, 'start2'] = min(start2, overlapping_regions_filtered2['start'].min())
                sam_entries.at[index, 'end2'] = max(end2, overlapping_regions_filtered2['end'].max())

    # Save the updated SAM file entries to the output file
    sam_entries.to_csv(output_file, sep='\t', header=False, index=False)

# Usage
txt_file = 'ADAR_nonhybrid.txt'
sam_file = 'ADAR-irCLASH_concatenated_hybrid_regions.txt'
output_file = 'ADAR_pos.txt'

# To extend "end1 start1"
merge_and_extend_positions(txt_file, sam_file, output_file)

