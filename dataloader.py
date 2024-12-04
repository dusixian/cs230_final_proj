import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader

# constants
END_TOKEN = "E"
PAD_TOKEN = "P"
START_TOKEN = "S"
MAX_SEQ_LENGTH = 102
# file_names = ["ADAR1_seq.txt", "ADAR2_seq.txt", "ADAR3_seq.txt", "Endogenous_ADAR1_seq.txt"]
# ADAR_types = ["ADAR1", "ADAR2", "ADAR3", "Endogenous_ADAR1"]
file_names = ["ADAR1_seq.txt"]
ADAR_types = ["ADAR1"]
vocabulary = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'E': 4, 'P': 5, 'other': 6, 'S': 7}
Location_vocab = {'Intron': 0, 'Intergenic': 1, 'lncRNA': 2, 'UTR': 3, 
                  'CDS': 4, 'tRNA': 5, 'miRNA': 6, 'rRNA': 7, 'other ncRNAs': 8, 'other': 9}
Chromosome_vocab = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                    '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19,
                    '20': 20, '21': 21, '22': 22, 'X': 23, 'Y': 24, 'MT': 25, 'other': 26}
RepeatType_vocab = {'Alu':0, 'Repetitive non-Alu': 1, 'Nonrepetitive': 2, 'other': 3}
ADAR_type_vocab = {'ADAR1': 0, 'ADAR2': 1, 'ADAR3': 2, 'Endogenous_ADAR1': 3, 'other': 4}

vocab_size = len(vocabulary)
features_name = ['Substrate', 'Arm', 'Chromosome', 'Strand', 'Start', 'End', 
                'Sequence', 'Location', 'Region', 'RepeatType', 'Source', 'NA']
embed_features = ['Chromosome', 'Location', 'RepeatType', 'ADAR_type']

def load_rna_pairs(file_path, ADAR_type, one_hot_encode=True, start_token=False, reverse_left=False):
    data = pd.read_csv(file_path, sep='\t', header=None, 
                       names=features_name)
    # TODO: @haoyu, plz check the feature names, especially the last 2 columns
    
    pairs = []
    for _, group in data.groupby('Substrate'):
        if len(group) == 2:
            left, right = group[group['Arm'] == 'L'].iloc[0], group[group['Arm'] == 'R'].iloc[0]
            seq_len = MAX_SEQ_LENGTH-1
            if start_token:
                seq_len -= 1
            if(len(left['Sequence']) > seq_len or len(right['Sequence']) > seq_len):
                continue

            pair = {
                "left": {col: left[col] for col in data.columns if col != 'Substrate'},
                "right": {col: right[col] for col in data.columns if col != 'Substrate'},
            }
            # reverse the left sequence
            if reverse_left:
                pair["left"]["Sequence"] = pair["left"]["Sequence"][::-1]
            if start_token:
                pair["left"]["Sequence"] = START_TOKEN + pair["left"]["Sequence"]
                pair["right"]["Sequence"] = START_TOKEN + pair["right"]["Sequence"]
            pair["left"]["Sequence"] = pair["left"]["Sequence"] + END_TOKEN
            pair["right"]["Sequence"] = pair["right"]["Sequence"] + END_TOKEN
            pair["left"]["Sequence"] = one_hot_encoder(pair["left"]["Sequence"], MAX_SEQ_LENGTH, one_hot_encode, start_token)
            pair["right"]["Sequence"] = one_hot_encoder(pair["right"]["Sequence"], MAX_SEQ_LENGTH, one_hot_encode, start_token)
            
            # process the features
            pair["left"]["ADAR_type"] = ADAR_type
            pair["right"]["ADAR_type"] = ADAR_type
            if 'UTR' in pair["left"]["Location"]:
                pair["left"]["Location"] = 'UTR'
            if 'UTR' in pair["right"]["Location"]:
                pair["right"]["Location"] = 'UTR'
            for feature in embed_features:
                pair["left"][feature] = globals()[feature + '_vocab'][pair["left"][feature]]
                pair["right"][feature] = globals()[feature + '_vocab'][pair["right"][feature]]
            # 把embed_features整合到一个‘feature‘中
            pair["left"]["feature"] = [pair["left"][feature] for feature in embed_features]
            pair["right"]["feature"] = [pair["right"][feature] for feature in embed_features]

            if random.random() < 0.5:
                pairs.append([pair["left"], pair["right"]])
            else:
                pairs.append([pair["right"], pair["left"]])
    
    return pairs

def one_hot_encoder(sequence, max_seq_length, one_hot_encode=True, start_token=False):
    indices = [vocabulary.get(char, 6) for char in sequence]
    # Padding if necessary
    if len(indices) < max_seq_length:
        indices += [vocabulary['P']] * (max_seq_length - len(indices))
    else:
        indices = indices[:max_seq_length]
    if one_hot_encode:
        if start_token:
            one_hot_seq = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=vocab_size)
        else:
            one_hot_seq = torch.nn.functional.one_hot(torch.tensor(indices), num_classes=vocab_size - 1)
        return one_hot_seq.float() 
    else:
        return torch.tensor(indices).int()

class RnaPairDataset(Dataset):
    def __init__(self, file_names, ADAR_types, max_seq_length=MAX_SEQ_LENGTH, pad_token=PAD_TOKEN, 
                 one_hot_encode=True, start_token=False, reverse_left=False, get_feature=False):
        rnn_pairs = []
        print(file_names)
        for i in range(len(file_names)):
            name = "./data/" + file_names[i]
            pairs = load_rna_pairs(name, ADAR_types[i], one_hot_encode=one_hot_encode, start_token=start_token, reverse_left=reverse_left)
            rnn_pairs.extend(pairs)
        self.rnn_pairs = rnn_pairs
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.get_feature = get_feature

    def __len__(self):
        return len(self.rnn_pairs)

    def __getitem__(self, idx):
        pair = self.rnn_pairs[idx]
        seq1 = pair[0]['Sequence']
        seq2 = pair[1]['Sequence']
        feature1 = pair[0]['feature']
        feature2 = pair[1]['feature']
        # feature1 = {k: pair[0][k] for k in embed_features} 
        # feature2 = {k: pair[1][k] for k in embed_features}
        if self.augment:
            if self.get_feature:
                return [
                    (seq1, feature1, seq2, feature2),
                    (seq2, feature2, seq1, feature1),
                ]
            return [
                (seq1, seq2),
                (seq2, seq1),
            ]
        else:
            if self.get_feature:
                return (seq1, feature1, seq2, feature2)
            return (seq1, seq2)


def get_dataloaders(file_names=file_names, ADAR_types=ADAR_types, batch_size=32, train_ratio=0.8, 
                    one_hot_encode=True, start_token=False, reverse_left=False, get_feature=False):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    dataset = RnaPairDataset(file_names, ADAR_types, 
                             one_hot_encode=one_hot_encode, start_token=start_token, 
                             reverse_left=reverse_left, get_feature=get_feature)
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