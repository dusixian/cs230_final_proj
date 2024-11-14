import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader

# constants
END_TOKEN = "E"
PAD_TOKEN = "P"
MAX_SEQ_LENGTH = 101 # TODO: @haoyu, check whether this is a good value
# file_names = ["ADAR1_seq.txt", "ADAR2_seq.txt", "ADAR3_seq.txt", "Endogenous_ADAR1_seq.txt"]
# ADAR_types = ["ADAR1", "ADAR2", "ADAR3", "Endogenous_ADAR1"]
file_names = ["ADAR1_seq.txt"]
ADAR_types = ["ADAR1"]
vocabulary = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'E': 4, 'P': 5, 'other': 6}
vocab_size = len(vocabulary)

def load_rna_pairs(file_path, ADAR_type):
    data = pd.read_csv(file_path, sep='\t', header=None, 
                       names=['Substrate', 'Arm', 'Chromosome', 'Strand', 'Start', 'End', 
                              'Sequence', 'Location', 'Region', 'RepeatType', 'Source', 'NA'])
    # TODO: @haoyu, plz check the feature names, especially the last 2 columns
    
    pairs = []
    for _, group in data.groupby('Substrate'):
        if len(group) == 2:
            left, right = group[group['Arm'] == 'L'].iloc[0], group[group['Arm'] == 'R'].iloc[0]
            if(len(left['Sequence']) > MAX_SEQ_LENGTH-1 or len(right['Sequence']) > MAX_SEQ_LENGTH-1):
                continue

            pair = {
                "left": {col: left[col] for col in data.columns if col != 'Substrate'},
                "right": {col: right[col] for col in data.columns if col != 'Substrate'},
            }
            pair["left"]["Sequence"] = one_hot_encode(pair["left"]["Sequence"]+END_TOKEN, MAX_SEQ_LENGTH)
            pair["right"]["Sequence"] = one_hot_encode(pair["right"]["Sequence"]+END_TOKEN, MAX_SEQ_LENGTH)
            pair["left"]["ADAR_type"] = ADAR_type
            pair["right"]["ADAR_type"] = ADAR_type
            if random.random() < 0.5:
                pairs.append([pair["left"], pair["right"]])
            else:
                pairs.append([pair["right"], pair["left"]])
    
    return pairs

def one_hot_encode(sequence, max_seq_length):
    indices = [vocabulary.get(char, 6) for char in sequence]
    # Padding if necessary
    
    if len(indices) < max_seq_length:
        indices += [vocabulary['P']] * (max_seq_length - len(indices))
    else:
        indices = indices[:max_seq_length]

    one_hot_seq = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=vocab_size)
    return one_hot_seq.float() 

class RnaPairDataset(Dataset):
    def __init__(self, file_names, ADAR_types, max_seq_length=MAX_SEQ_LENGTH, pad_token=PAD_TOKEN):
        rnn_pairs = []
        for i in range(len(file_names)):
            name = "./data/" + file_names[i]
            pairs = load_rna_pairs(name, ADAR_types[i])
            rnn_pairs.extend(pairs)
        self.rnn_pairs = rnn_pairs
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token

    def __len__(self):
        return len(self.rnn_pairs)

    def __getitem__(self, idx):
        pair = self.rnn_pairs[idx]
        seq1 = pair[0]['Sequence']
        seq2 = pair[1]['Sequence']
        feature1 = [v for k, v in pair[0].items() if k != 'Sequence']
        feature2 = [v for k, v in pair[1].items() if k != 'Sequence']
        if self.augment:
            return [
                # (seq1, seq2, feature1, feature2),
                # (seq2, seq1, feature2, feature1)
                (seq1, seq2),
                (seq2, seq1),
            ]
        else:
            # return (seq1, seq2, feature1, feature2)
            return (seq1, seq2)


def get_dataloaders(file_names=file_names, ADAR_types=ADAR_types, batch_size=32, train_ratio=0.8):
    dataset = RnaPairDataset(file_names, ADAR_types)
    train_size = int(train_ratio * len(dataset))
    dev_size = int((1-train_ratio)/2 * len(dataset))
    test_size = len(dataset) - train_size - dev_size
    train_dataset, dev_dataset, test_dataset = random_split(dataset, [train_size, dev_size, test_size])

    # Enable augmentation only for the training dataset
    train_dataset.dataset.augment = True  # Double the pairs for train set with both directions
    dev_dataset.dataset.augment = False   # Single direction for dev set
    test_dataset.dataset.augment = False  # Single direction for test set

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader, test_loader