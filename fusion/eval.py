import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
# from feature_dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from data import merge_features
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def evaluate(model, test_loader, device):
    model.eval()
    targets = []
    predictions = []

    with torch.no_grad():
        for traj, data, target in tqdm(test_loader):
            traj, data, target = traj.to(device), data.to(device), target.to(device)
            output = model(data, traj)

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

class fusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(fusion, self).__init__()
        concat_size = hidden_size + 408
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(concat_size, concat_size//2)
        self.fc2 = nn.Linear(concat_size//2, num_classes)
        self.smax = nn.Softmax(dim=1)
    
    def forward(self, x, y):
        _, hidden = self.gru(x)
        hidden = hidden.squeeze(0)  # Remove the batch dimension from the hidden state
        fused = torch.cat((hidden, y), 1)
        fused = self.fc1(fused)
        fused = self.fc2(fused)
        output = self.smax(fused)
        return output
    
# Calculate the train-validation split sizes
val_split=0.3
dataset =   merge_features('frames.json', 'traj.json')
dataset_size = len(dataset)


val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size

# Split the dataset into train and validation subsets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

def collate_fn(batch): #(2)
    traj, sequences, labels = zip(*batch)
    sequences = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels)
    # traj = torch.tensor(traj)
    traj = torch.stack(list(traj), dim=0)
    return traj, sequences, labels

input_size = 512
hidden_size = 128
num_classes = 13
num_layers=1

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
model_path = 'best_model.pth'
device = torch.device("cpu")
model = fusion(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load(model_path))

accuracies, class_accuracies, confusion_mat = evaluate(model, val_dataloader, device)

print(accuracies)
print(class_accuracies)
print(confusion_mat)