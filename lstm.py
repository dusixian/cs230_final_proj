import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders, MAX_SEQ_LENGTH, vocab_size
from torch.autograd import Variable
import time

class RNAPairLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(RNAPairLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, input): 
        h_0 = Variable(torch.zeros(2*self.num_layers, input.size(0), self.hidden_dim, requires_grad=False).to(device))
        c_0 = Variable(torch.zeros(2*self.num_layers, input.size(0), self.hidden_dim).to(device))

        output, (h_out, _) = self.lstm(input, (h_0, c_0))
        output = self.fc(output)
        
        return output

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for seq1, seq2 in train_loader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            outputs = model(seq1)
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
    output_dim = vocab_size  # One-hot encoded output size
    num_layers = 2
    num_epochs = 100
    learning_rate = 1e-2
    batch_size = 32

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using ' + device)

    # Load data
    train_loader, dev_loader, test_loader = get_dataloaders(batch_size=batch_size)

    # Initialize model, criterion and optimizer
    model = RNAPairLSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    weight = torch.tensor([1,1,1,1,1,0.01,1],dtype=torch.float32,requires_grad=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs)
    # save model using dd/mm-hh:mm
    path = time.strftime("%d-%m-%H:%M") + '.pth'
    # torch.save(model.state_dict(), 'model_test.pth')
    torch.save(model.state_dict(), 'model_test.pth')

    # Evaluate the model
    evaluate_model(model, dev_loader, criterion)