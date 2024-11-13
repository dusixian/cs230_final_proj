import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders, MAX_SEQ_LENGTH, vocab_size

class RNAPairLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNAPairLSTM, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.input_fc = nn.Linear(input_dim, hidden_dim)  # Add this line

    def forward(self, src, tgt):
        _, (hidden, cell) = self.encoder(src)
        tgt = self.input_fc(tgt)  # Add this line
        outputs, _ = self.decoder(tgt, (hidden, cell))
        outputs = self.fc(outputs)
        return outputs

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for seq1, seq2 in train_loader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            optimizer.zero_grad()
            outputs = model(seq1, seq2)
            loss = criterion(outputs.view(-1, outputs.size(-1)), seq2.view(-1, seq2.size(-1)))
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, dev_loader, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for seq1, seq2 in dev_loader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1, seq2)
            loss = criterion(outputs.view(-1, outputs.size(-1)), seq2.view(-1, seq2.size(-1)))
            total_loss += loss.item()
        print(f'Dev Loss: {total_loss / len(dev_loader):.4f}')

if __name__ == "__main__":
    # Hyperparameters
    input_dim = vocab_size  # One-hot encoded input size
    hidden_dim = 128
    output_dim = vocab_size  # One-hot encoded output size
    num_layers = 2
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 32

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, dev_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize model, criterion and optimizer
    model = RNAPairLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # Evaluate the model
    evaluate_model(model, dev_loader, criterion)