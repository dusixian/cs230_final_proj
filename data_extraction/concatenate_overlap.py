import pandas as pd
import multiprocessing as mp
from collections import defaultdict, deque


# Function to read the file and store data in a DataFrame
def read_file(file_path):
    columns = ['chromosome', 'strand', 'start1', 'end1', 'start2', 'end2', 'id', 'sequence']
    return pd.read_csv(file_path, sep='\t', header=None, names=columns)

def find_internal_overlaps(df_sorted):
    # Sort by strand and start positions for efficient searching
    
    overlaps = []
    overlapping_indices = set()

    # Using two pointers to check for overlaps
    n = len(df_sorted)
    for i in range(n):
        row1 = df_sorted.iloc[i]
        for j in range(i + 1, n):
            row2 = df_sorted.iloc[j]

            # Break if no overlap is possible
            if row1['strand'] != row2['strand']:
                break
            if row1['end1']<row2['start1']:
                break
            # Check for overlaps
            if (row1['start2'] <= row2['end2'] and row1['end2'] >= row2['start2']):
                overlaps.append((df_sorted.index[i], df_sorted.index[j]))
                overlapping_indices.add(df_sorted.index[i])
                overlapping_indices.add(df_sorted.index[j])
    non_overlaps_df = df_sorted.drop(list(overlapping_indices))
    return overlaps, non_overlaps_df
# Function to find connected components in the overlaps

def find_connected_components(overlaps):
    graph = defaultdict(set)
    for idx1, idx2 in overlaps:
        graph[idx1].add(idx2)
        graph[idx2].add(idx1)
    
    visited = set()
    components = []

    def bfs(start):
        queue = deque([start])
        component = set()
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                component.add(node)
                queue.extend(graph[node] - visited)
        return component

    for node in graph:
        if node not in visited:
            component = bfs(node)
            components.append(component)
    return components

# Function to merge overlapping regions
def merge_overlapping_regions(df, components):
    merged = []
    
    for component in components:
        if len(component) > 1:
            rows = df.loc[component]
            merged_start1 = rows['start1'].min()
            merged_end1 = rows['end1'].max()
            merged_start2 = rows['start2'].min()
            merged_end2 = rows['end2'].max()
            merged_id = '_'.join(rows['id'].astype(str))
            merged_sequence = '_'.join(rows['sequence'].astype(str))
            
            merged.append({
                'chromosome': rows['chromosome'].iloc[0],
                'strand': rows['strand'].iloc[0],
                'start1': merged_start1,
                'end1': merged_end1,
                'start2': merged_start2,
                'end2': merged_end2,
                'id': merged_id,
                'sequence': merged_sequence
            })
            
    merged_df = pd.DataFrame(merged)

    return merged_df


# Function to process each chromosome
def process_chromosome(df1_chromosome):
    # Concatenate the DataFrames
    #df_combined = pd.concat([df1_chromosome.assign(source='df1'), df2_chromosome.assign(source='df2')])
    #df_combined = pd.concat([df1_chromosome.assign(source='df1'), df2_chromosome.assign(source='df2'),df3_chromosome.assign(source='df3')])
    df_sorted = df1_chromosome.sort_values(by=['strand', 'start1', 'start2']).reset_index(drop=True)
    # Find internal overlaps in the combined DataFrame
    internal_overlaps, non_overlaps_df = find_internal_overlaps(df_sorted)

    # Find connected components
    components = find_connected_components(internal_overlaps)

    # Merge overlapping regions
    overlap = merge_overlapping_regions(df_sorted, components)
    merged_internal=pd.concat([overlap, non_overlaps_df], ignore_index=True)
    return merged_internal

# Read the files
#file1 = 'GSM4045981_ADAR3-irCLASH-rep1-hybrid-reads_filtered.txt'
#file2 = 'GSM4045982_ADAR3-irCLASH-rep2-hybrid-reads_filtered.txt'
#file3 = 'GSM4045978_ADAR1-irCLASH-rep3-hybrid-reads.txt'
file = 'ADAR_pos.txt'

#df1 = read_file(file1)
#df2 = read_file(file2)
#df3 = read_file(file3)
df = read_file(file)

# Split the data into chromosomes
#chromosome_dfs_1 = {chrom: df1[df1['chromosome'] == chrom] for chrom in df1['chromosome'].unique()}
#chromosome_dfs_2 = {chrom: df2[df2['chromosome'] == chrom] for chrom in df2['chromosome'].unique()}
#chromosome_dfs_3 = {chrom: df3[df3['chromosome'] == chrom] for chrom in df3['chromosome'].unique()}
chromosome_dfs = {chrom: df[df['chromosome'] == chrom] for chrom in df['chromosome'].unique()}

# Get common chromosomes to ensure processing pairs
#common_chromosomes = set(chromosome_dfs_1.keys()).intersection(set(chromosome_dfs_2.keys()))
#common_chromosomes = set(chromosome_dfs_1.keys()).intersection(
#    set(chromosome_dfs_2.keys()),
#    set(chromosome_dfs_3.keys())
#)
common_chromosomes = set(chromosome_dfs.keys())

# Create a pool of workers to process each chromosome
    
# Create a pool of workers to process each chromosome
with mp.Pool(processes=mp.cpu_count()) as pool:
    results = pool.map(process_chromosome, 
                       [chromosome_dfs[chrom] for chrom in common_chromosomes])

#with mp.Pool(processes=mp.cpu_count()) as pool:
#    results = pool.starmap(process_chromosome, 
#                           [(chromosome_dfs_1[chrom], chromosome_dfs_2[chrom], chromosome_dfs_3[chrom]) 
#                            for chrom in common_chromosomes])
# Combine results
final_df = pd.concat(results, ignore_index=True)

# Save to a file
final_df.to_csv('ADAR_pos_final.txt', sep='\t', index=False, header=False)

