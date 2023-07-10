import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from feature_dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from feature_dataset import feature_set
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def evaluate(model, test_loader, device):
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Get the predicted class labels
            _, predicted = torch.max(output, dim=1)

            # Collect the targets and predictions
            targets.extend(target.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

    targets = np.array(targets)
    predictions = np.array(predictions)

    # Calculate overall accuracy
    accuracy = accuracy_score(targets, predictions)

    # Calculate class-wise accuracies
    class_accuracies = {}
    unique_classes = np.unique(targets)
    for class_label in unique_classes:
        class_indices = np.where(targets == class_label)
        class_accuracy = accuracy_score(targets[class_indices], predictions[class_indices])
        class_accuracies[class_label] = class_accuracy

    # Create confusion matrix
    confusion_mat = confusion_matrix(targets, predictions)

    return accuracy, class_accuracies, confusion_mat

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.smax = nn.Softmax(dim=1)
    
    def forward(self, x):
        _, hidden = self.gru(x[:,:-5,:])
        hidden = hidden.squeeze(0)  # Remove the batch dimension from the hidden state
        output = self.fc(hidden)
        output = self.smax(output)
        return output
    
# Calculate the train-validation split sizes
val_split=0.3
dataset =   feature_set('merge.json')
dataset_size = len(dataset)


val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size

# Split the dataset into train and validation subsets
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))


def collate_fn(batch): #(2)
    sequences, labels = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels)
    return sequences, labels

input_size = 512
hidden_size = 128
num_classes = 13
num_layers=1

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
model_path = 'best_model-2.pth'
device = torch.device("cpu")
model = GRUModel(input_size, hidden_size, num_classes, num_layers).to(device)
model.load_state_dict(torch.load(model_path))

accuracies, class_accuracies, confusion_mat = evaluate(model, val_dataloader, device)

print(accuracies)
print(class_accuracies)
print(confusion_mat)