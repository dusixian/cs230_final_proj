#this script is the second half of getting ADAR substrate

#get nonhybrid sequences
bowtie -a -m 3 -x /home/groups/xjgao/irClash/data/fastq/bowtie_index/genome_bowtie_index/genome_bowtie_index -S nonhybrid_Sample_mapped.sam -f input_fasta
python3 get_nonhybird.py
#concatenate all nonhybrid sequences
python3 concatenate_nonhybrid.py #merge samples of same ADAR together
python3 concatenate_overlap.py #check overlapping inside nonhybrids 
#merge  hybrid & nonhybrid
python3 merge_nonhybrid.py
#further extension
python3 concatenate_overlap.py #further check if there's more overlap
#anneal and trim substrate
python3 convert_fasta.py #to get input files for RNAhybrid
RNAhybrid -b 1 -s 3utr_human -t ADAR_arm1.fasta -q ADAR_arm2.fasta >ADAR_RNAhybrid #run RNAhybrid
python3 trim_seq.py #trim unannealed parts
#add annotations
python3 get_rep.py
#get editing levels
perl Query_Editing_Level_fuqiang.pl
#align position to sequence
python3 align_pos_to_seq.py
