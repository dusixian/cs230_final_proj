import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import random
from dataloader import get_dataloaders, MAX_SEQ_LENGTH, vocab_size
from torch.autograd import Variable
import time

class Encoder(nn.Module):
    ''' Sequence to sequence networks consists of Encoder and Decoder modules.
    This class contains the implementation of Encoder module.

    Args:
        input_dim: A integer indicating the size of input dimension.
        emb_dim: A integer indicating the size of embeddings.
        hidden_dim: A integer indicating the hidden dimension of RNN layers.
        n_layers: A integer indicating the number of layers.
        dropout: A float indicating dropout.
    '''
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Linear(input_dim, emb_dim, device=device)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, device=device)  # default is time major
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src is of shape [sentence_length, batch_size], it is time major

        # embedded is of shape [sentence_length, batch_size, embedding_size]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        outputs, (hidden, cell) = self.lstm(embedded)

        # outputs are always from the top hidden layer, if bidirectional outputs are concatenated.
        # outputs shape [sequence_length, batch_size, hidden_dim * num_directions]
        return hidden, cell


class Decoder(nn.Module):
    ''' This class contains the implementation of Decoder Module.

    Args:
        embedding_dim: A integer indicating the embedding size.
        output_dim: A integer indicating the size of output dimension.
        hidden_dim: A integer indicating the hidden size of rnn.
        n_layers: A integer indicating the number of layers in rnn.
        dropout: A float indicating the dropout.
    '''
    def __init__(self, embedding_dim, output_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Linear(output_dim, embedding_dim, device=device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, device=device)
        self.linear = nn.Linear(hidden_dim, output_dim, device=device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input is of shape [batch_size]
        # hidden is of shape [n_layer * num_directions, batch_size, hidden_size]
        # cell is of shape [n_layer * num_directions, batch_size, hidden_size]

        input = input.unsqueeze(1)
        # input shape is [1, batch_size]. reshape is needed rnn expects a rank 3 tensors as input.
        # so reshaping to [1, batch_size] means a batch of batch_size each containing 1 index.

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # embedded is of shape [1, batch_size, embedding_dim]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        predicted = self.linear(output)  # linear expects as rank 2 tensor as input
        # predicted shape is [batch_size, output_dim]

        return predicted, hidden, cell


class Seq2Seq(nn.Module):
    ''' This class contains the implementation of complete sequence to sequence network.
    It uses to encoder to produce the context vectors.
    It uses the decoder to produce the predicted target sentence.

    Args:
        encoder: A Encoder class instance.
        decoder: A Decoder class instance.
    '''
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src is of shape [sequence_len, batch_size]
        # trg is of shape [sequence_len, batch_size]
        # if teacher_forcing_ratio is 0.5 we use ground-truth inputs 50% of time and 50% time we use decoder outputs.

        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # to store the outputs of the decoder
        outputs = torch.zeros(batch_size, max_len+1, trg_vocab_size)

        # context vector, last hidden and cell state of encoder to initialize the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = torch.zeros_like(trg[:, 0, :]).to(device)

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:,t,:] = output.squeeze(1)
            use_teacher_force = random.random() < teacher_forcing_ratio
            # top1 = output.max(1)[1]
            input = (trg[:,t-1,:].squeeze(1) if use_teacher_force else output.squeeze(1))

        # outputs is of shape [sequence_len, batch_size, output_dim]
        return outputs[:,1:,:]

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for seq1, seq2 in train_loader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1, seq2).to(device)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), seq2.reshape(-1, seq2.size(-1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, dev_loader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for seq1, seq2 in dev_loader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), seq2.reshape(-1, seq2.size(-1)))
            total_loss += loss.item()
        print(f'Dev Loss: {total_loss / len(dev_loader):.4f}')

if __name__ == "__main__":
    # Hyperparameters
    input_dim = vocab_size  # One-hot encoded input size
    hidden_dim = 128
    emb_dim = hidden_dim
    dropout = 0.8
    output_dim = vocab_size  # One-hot encoded output size
    num_layers = 2
    num_epochs = 10
    learning_rate = 1e-2
    batch_size = 32

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using ' + device)

    # Load data
    train_loader, dev_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize model, criterion and optimizer
    encoder = Encoder(input_dim, emb_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(emb_dim,output_dim,hidden_dim,num_layers,dropout)
    model = Seq2Seq(encoder,decoder).to(device)
    # weight = torch.tensor([1,1,1,1,1,0.01,1],dtype=torch.float32,requires_grad=False).to(device)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    # save model using dd/mm-hh:mm
    # path = time.strftime("%d-%m-%H:%M") + '.pth'
    # torch.save(model.state_dict(), 'model.pth')

    # Evaluate the model
    evaluate_model(model, dev_loader, criterion)