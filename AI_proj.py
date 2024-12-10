import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets import MNIST
from PIL import Image
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.preprocessing import label_binarize
#from models import CNN_for_DeepFake, MLP_for_DeepFake, CNN_LSTM_for_DeepFake
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import csv
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Original transformation
original_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

# Augmented transformation with horizontal flip and random rotation
augmented_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), 
    # Randomly rotate images in the range (-15, 15) degrees
    transforms.RandomRotation(degrees=15),   
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the dataset from the root directory
raw_dataset = datasets.ImageFolder(root='processed_dataset_frame/processed_dataset_frame', transform=None)

# Access a raw image and its label directly
raw_img, label = raw_dataset[0]

# Now apply the transformations to the raw PIL image for demonstration
img_original_tensor = original_transform(raw_img)
img_augmented_tensor = augmented_transform(raw_img)

# Access the raw image and its label from dataset
img, label = raw_dataset[0]

# Define the function to unnormalize and show the image
def show_image(img, title=None, ax=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  
    img = np.clip(img, 0, 1)
    if ax is not None:
        ax.imshow(img)
        if title is not None:
            ax.set_title(title)
        ax.axis('off')
    else:
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Displaying the original image and the augmented image
show_image(img_original_tensor, title='Original Image', ax=axs[0])
show_image(img_augmented_tensor, title='Augmented Image', ax=axs[1])

#plt.show()

# Create the dataloders
root_directory = 'processed_dataset_frame/processed_dataset_frame'
test_directory = 'processed_dataset_frame_test'
batch_size = 16

# Path to the dataset root
dataset_path = 'processed_dataset_frame/processed_dataset_frame'

# Define the image transformations
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms = transforms.Compose([
    transforms.Resize((im_size, im_size)),  # Resize all images to a fixed size
    transforms.ToTensor(),                  # Convert images to PyTorch tensors
    transforms.Normalize(mean, std)         # Normalize the tensor images
])

# Loading the dataset using ImageFolder
dataset = ImageFolder(dataset_path, transform=transforms)

# Splitting the dataset into train and validation subsets
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# Creating data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2)

# Example: Checking class names and some dataset info
print("Classes:", dataset.classes)
print("Number of train samples:", len(train_dataset))
print("Number of validation samples:", len(valid_dataset))

# Configuration
image_dir = 'processed_dataset_frame/processed_dataset_frame'  # Update this path
batch_size = 8

# Define the image transformations
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),  # Resize all images to a fixed size
    transforms.ToTensor(),                  # Convert images to PyTorch tensors
    transforms.Normalize(mean, std)         # Normalize the tensor images
])

def load_image(filepath):
    with Image.open(filepath) as img:
        return transform(img)

def prepare_data(root_dir):
    data, labels = [], []
    categories = {'real': 1, 'fake': 0}

    for category in ['real', 'fake']:
        dir_path = os.path.join(root_dir, category)
        video_dict = {}

        # Collect files by the first four digits of their names
        for file in os.listdir(dir_path):
            if file.endswith('.jpg'):
                video_id = file[:4]  # The first four digits of the filename
                if video_id not in video_dict:
                    video_dict[video_id] = []
                video_dict[video_id].append(os.path.join(dir_path, file))

        # Sort files, load images, and store sequences
        for video_id, video_files in video_dict.items():
            try:
                sorted_files = sorted(
                    video_files,
                    key=lambda x: int(os.path.basename(x).split('_frame')[1].split('.jpg')[0])
                )
            except ValueError as e:
                print(f"Error parsing file name from: {x}")
                continue  # Skip this file or handle it according to your policy

            loaded_images = [load_image(fp) for fp in sorted_files]
            data.append(torch.stack(loaded_images))
            labels.append(categories[category])

            # Print example sequence details
            print(f"Sequence ID: {video_id}")
            print(f"Category: {category}")
            print(f"Number of Frames: {len(sorted_files)}")
            print(f"Sample Frames: {sorted_files[:5]}")  # Print first 5 frame filenames for checking

    # Convert labels to tensor
    Y = torch.tensor(labels)
    
    # Split into train and test sets
    return train_test_split(data, Y, test_size=0.2)

# Usage
X_train, X_test, y_train, y_test = prepare_data(image_dir)


# Define image transformations
im_size = 112
transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),  # Resize all images to a fixed size
    transforms.ToTensor(),                  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(filepath):
    with Image.open(filepath) as img:
        return transform(img)

def prepare_data(root_dir, csv_path):
    sequences, labels, metadata = [], [], []
    categories = {'real': 1, 'fake': 0}

    # Open a CSV file to save the metadata
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['sequence_id', 'category', 'frame_count', 'sample_frames'])

        for category in ['real', 'fake']:
            dir_path = os.path.join(root_dir, category)
            video_dict = {}

            # Collect files by the first four digits of their names
            for file in os.listdir(dir_path):
                if file.endswith('.jpg'):
                    video_id = file[:4]  # The first four digits of the filename
                    if video_id not in video_dict:
                        video_dict[video_id] = []
                    video_dict[video_id].append(os.path.join(dir_path, file))

            # Sort files, load images, and store sequences
            for video_id, video_files in video_dict.items():
                try:
                    sorted_files = sorted(
                        video_files,
                        key=lambda x: int(os.path.basename(x).split('_frame')[1].split('.jpg')[0])
                    )
                except ValueError as e:
                    print(f"Error parsing file name from: {x}")
                    continue  # Skip this file or handle it according to your policy

                loaded_images = [load_image(fp) for fp in sorted_files]
                sequences.append(torch.stack(loaded_images))
                labels.append(categories[category])
                metadata_entry = {
                    'sequence_id': video_id,
                    'category': category,
                    'frame_count': len(sorted_files),
                    'sample_frames': '; '.join(sorted_files[:5])
                }
                metadata.append(metadata_entry)

                # Write to CSV
                writer.writerow([video_id, category, len(sorted_files), '; '.join(sorted_files[:5])])

                # Optionally print the metadata for verification
                print(metadata_entry)

    return sequences, labels, metadata

# Usage: Specify the path where you want to save the CSV
sequences, labels, metadata = prepare_data('processed_dataset_frame/processed_dataset_frame', 'metadata.csv')




class SequenceModel(nn.Module):
    def __init__(self, num_classes):
        super(SequenceModel, self).__init__()
        self.resnext = nn.Sequential(*list(resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT).children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(2048, 512, num_layers=1, batch_first=True)
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x, lengths):
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.resnext(c_in)
        c_out = self.avgpool(c_out)
        c_out = c_out.view(c_out.size(0), -1)
        r_in = c_out.view(batch_size, timesteps, -1)

        # Pack the sequence, process through LSTM, and then unpack
        packed_input = pack_padded_sequence(r_in, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        r_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        r_out = r_out[:, -1, :]  # Get the last timestep outputs

        output = self.fc(r_out)
        return output


def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    # Convert list of sequences where each sequence is a list of tensors to a list of tensor sequences
    sequences = [torch.stack(seq) for seq in sequences]  # Stack each sequence to make 3D tensor
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return sequences_padded, labels, lengths


class FrameSequenceDataset(Dataset):
    def __init__(self, csv_file, transform=None, indices=None):
        self.data = pd.read_csv(csv_file)
        if indices is not None:
            self.data = self.data.iloc[indices]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        frames = row['sample_frames'].split('; ')
        sequence = [self.load_frame(frame) for frame in frames]
        label = int(row['category'] == 'real')
        if self.transform:
            sequence = [self.transform(Image.open(frame).convert('RGB')) for frame in frames]
        return sequence, label, len(sequence)

    def load_frame(self, frame_path):
        img = Image.open(frame_path).convert('RGB')  # Ensure it's always RGB
        return img




# Initialize the Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SequenceModel(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_path='best_model.pth'):
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, labels, lengths in train_loader:
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

        # Validate after every epoch
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        
        # Print training and validation results
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%')

        # Check if the current validation loss is the best we've seen so far
        if val_loss < best_val_loss:
            print(f'Validation loss decreased ({best_val_loss:.6f} --> {val_loss:.6f}). Saving model ...')
            best_val_loss = val_loss
            # Save model state dictionary
            torch.save(model.state_dict(), checkpoint_path)

def validate_model(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, lengths in loader:
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    return val_loss / len(loader), val_accuracy

# Example usage:
data_indices = np.arange(len(pd.read_csv('metadata.csv')))
np.random.shuffle(data_indices)
split = int(0.8 * len(data_indices))  # 80% for training, 20% for validation

train_indices = data_indices[:split]
val_indices = data_indices[split:]

# Prepare Dataset and DataLoader
train_dataset = FrameSequenceDataset('metadata.csv', transform=transform, indices=train_indices)
val_dataset = FrameSequenceDataset('metadata.csv', transform=transform, indices=val_indices)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

# Initialize the Model and Set Device
model = SequenceModel(num_classes=2).to(device)

# Setup Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Path to save the best model
checkpoint_path = 'best_model.pth'

# Start Training
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device, checkpoint_path=checkpoint_path)












# Save the model weights
model_path = 'model_state.pth'
torch.save(model.state_dict(), model_path)
print(f"Saved trained model state to {model_path}")

# Save the losses and accuracies
np.save('train_losses.npy', train_losses)
np.save('validation_losses.npy', validation_losses)
np.save('test_losses.npy', test_losses)
np.save('train_accuracies.npy', train_accuracies)
np.save('validation_accuracies.npy', validation_accuracies)
np.save('test_accuracies.npy', test_accuracies)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.plot(validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracies')
plt.legend()
plt.grid(True)
#plt.show()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.grid(True)
#plt.show()