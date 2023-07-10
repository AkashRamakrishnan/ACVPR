import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
# from feature_dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from data import traj_features
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

class classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size//2)
        self.fc2 = nn.Linear(input_size//2, num_classes)
        self.smax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.smax(x)
        return output
    
# Calculate the train-validation split sizes
val_split=0.3
dataset =   traj_features('merged_data.json')
dataset_size = len(dataset)


val_size = int(val_split * dataset_size)
train_size = dataset_size - val_size

# Split the dataset into train and validation subsets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

# def collate_fn(batch): #(2)
#     sequences, labels = zip(*batch)
#     sequences = pad_sequence(sequences, batch_first=True)
#     labels = torch.tensor(labels)
#     return sequences, labels

input_size = 408
hidden_size = 128
num_classes = 13
num_layers=1

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
model_path = 'best_model-2.pth'
device = torch.device("cpu")
model = classifier(input_size, num_classes).to(device)
model.load_state_dict(torch.load(model_path))

accuracies, class_accuracies, confusion_mat = evaluate(model, val_dataloader, device)

print(accuracies)
print(class_accuracies)
print(confusion_mat)