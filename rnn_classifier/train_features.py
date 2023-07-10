import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from feature_dataset import feature_set
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

def collate_fn(batch): #(2)
    sequences, labels = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels)
    return sequences, labels
# Define your GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.smax = nn.Softmax(dim=1)
    
    def forward(self, x):
        _, hidden = self.gru(x)
        hidden = hidden.squeeze(0)  # Remove the batch dimension from the hidden state
        output = self.fc(hidden)
        output = self.smax(output)
        return output

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
print(device)

# Set your hyperparameters
input_size = 512
hidden_size = 128
num_classes = 13
batch_size = 2
learning_rate = 0.001
num_epochs = 50
num_layers = 1
test_split = 0.2  # Define the proportion of data to be used for validation
val_split = 0.2
patience=10

dataset =   feature_set('merge.json')

# Calculate the train-validation split sizes
dataset_size = len(dataset)
test_size = int(test_split * dataset_size)
train_size = dataset_size - test_size

# Split the dataset into train and validation subsets
train_dataset, val_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))

val_size = int(val_split*train_size)
train_size = train_size - val_size

# train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

# Create data loaders for training and validations
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print('Data Lengths:')
print('train, ', len(train_dataset))
print('val, ', len(val_dataset))
# print('test, ', len(test_dataset))

# Create your GRU model instance
model = GRUModel(input_size, hidden_size, num_classes, num_layers).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
# Start training
total_step = len(train_dataloader)
tr_losses = []
val_losses = []
for epoch in range(num_epochs):
    train_loss = 0
    for i, (sequences, labels) in enumerate(tqdm(train_dataloader)):
        sequences = sequences.float()  # Convert sequences to float
        labels = labels.long()  # Convert labels to long
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # Print training loss for every few steps
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')
    avg_train_loss = train_loss/len(train_dataloader)
    tr_losses.append(avg_train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {loss.item():.4f}')
    # Validation
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0
        total_val_samples = 0
        correct_val_samples = 0
        for val_sequences, val_labels in tqdm(val_dataloader):
            val_sequences = val_sequences.float()
            val_labels = val_labels.long()
            val_sequences.to(device)
            val_labels.to(device)
            val_outputs = model(val_sequences)
            val_loss += criterion(val_outputs, val_labels).item()
            
            _, val_predicted = torch.max(val_outputs.data, 1)
            total_val_samples += val_labels.size(0)
            correct_val_samples += (val_predicted == val_labels).sum().item()
        
        val_accuracy = correct_val_samples / total_val_samples
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        print(print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, Validation Accuracy {val_accuracy*100:.2f}%'))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model-2.pth')
            
        if epoch - best_epoch >= patience:
            print(f'Early stopping! No improvement in validation loss for {patience} epochs.')
            print(f'Best loss at Epoch {best_epoch}')
            break
        
    plt.plot(tr_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # Save the plot as an image
    plt.savefig(f'loss_plot.png')
    plt.close()  # Close the figure to release memory