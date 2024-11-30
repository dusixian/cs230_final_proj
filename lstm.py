import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloaders, MAX_SEQ_LENGTH, vocab_size
from torch.autograd import Variable
import time

save_model = False

class RNAPairLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, device='cpu'):
        super(RNAPairLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, input): 
        h_0 = Variable(torch.zeros(2*self.num_layers, input.size(0), self.hidden_dim, requires_grad=False).to(self.device))
        c_0 = Variable(torch.zeros(2*self.num_layers, input.size(0), self.hidden_dim).to(self.device))

        output, (h_out, _) = self.lstm(input, (h_0, c_0))
        output = self.fc(output)
        
        return output


def train_model(model, train_loader, criterion, optimizer, num_epochs, device, time_stamp):
    best_model = None
    best_dev_loss = float('inf')
    best_train_loss = 0
    best_epoch = 0

    loss_arr = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        total_loss = 0
        for seq1, seq2 in train_loader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            seq2_idx = torch.argmax(seq2, dim=-1).to(device)
            outputs = model(seq1)
            outputs = outputs[:, 1:, :]
            seq2_idx = seq2_idx[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), seq2_idx.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
        loss_arr.append(total_loss)

        # eval model
        if (epoch+1) % 10 == 0:
            dev_loss = evaluate_model(model, dev_loader, criterion, device)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_model = model
                best_train_loss = total_loss
                best_epoch = epoch

    if save_model:
        torch.save(best_model.state_dict(), './model/'+time_stamp+'/lstm_model_best.pth')

        import json
        with open('./model/'+time_stamp+'/loss.json', 'w') as f:
            json.dump({'loss': loss_arr, 
                    'best_epoch': best_epoch, 
                    'best_dev_loss': best_dev_loss, 
                    'best_train_loss': best_train_loss}, f, indent=4)
            
    print(f'Best dev loss: {best_dev_loss:.4f}, training loss: {best_train_loss:.4f}')
    print('Training finished')

def evaluate_model(model, dev_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for seq1, seq2 in dev_loader:
            seq1, seq2 = seq1.to(device), seq2.to(device)
            seq2_idx = torch.argmax(seq2, dim=-1).to(device)
            outputs = model(seq1)
            outputs = outputs[:, 1:, :]
            seq2_idx = seq2_idx[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), seq2_idx.reshape(-1))
            total_loss += loss.item()
        print(f'Dev Loss: {total_loss / len(dev_loader):.4f}')
    return total_loss / len(dev_loader)

if __name__ == "__main__":
    # Hyperparameters
    input_dim = vocab_size  # One-hot encoded input size
    hidden_dim = 128
    output_dim = vocab_size  # One-hot encoded output size
    num_layers = 2
    num_epochs = 30
    learning_rate = 1e-2
    batch_size = 32
    time_stamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    if save_model:
        import os
        if not os.path.exists('./model'):
            os.makedirs('./model')
        os.makedirs('./model/'+time_stamp)

        import json
        hyperparameters = {
            'time_stamp': time_stamp,
            'model': 'LSTM',
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }
        with open('./model/'+time_stamp+'/hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f, indent=4)

    # Device configuration
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print('Using ' + device)

    # Load data
    train_loader, dev_loader, test_loader = get_dataloaders(batch_size=batch_size, start_token=True)

    # Initialize model, criterion and optimizer
    model = RNAPairLSTM(input_dim, hidden_dim, output_dim, num_layers, device).to(device)
    weight = torch.tensor([1,1,1,1,2,0.01,1,1],dtype=torch.float32,requires_grad=False).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** epoch)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs, device, time_stamp)