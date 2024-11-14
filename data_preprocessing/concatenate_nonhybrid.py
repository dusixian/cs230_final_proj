import os
import glob

def concatenate_files(output_file, input_folder):
    with open(output_file, "w") as outfile:
        for filename in glob.glob(os.path.join(input_folder, "*.txt")):
            with open(filename, "r") as infile:
                outfile.writelines(infile.readlines())

# Example usage
output_file = "ADAR_nonhybrid.txt"  # Replace with your desired output file path
input_folder = "nonhybrid_ADAR"  # Folder where processed files are saved
concatenate_files(output_file, input_folder)
print(f"Concatenated all files into {output_file}")
