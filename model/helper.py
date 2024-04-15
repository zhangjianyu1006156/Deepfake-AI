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
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize
from models import CNN_for_DeepFake, MLP_for_DeepFake, CNN_LSTM_for_DeepFake
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Data Loading and Preprocessing ##

# Define original transformation
def get_original_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])

# Define augmented transformation with random adjustments
def get_augmented_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(degrees=15),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

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

def setup_data_loaders(root_dir, test_dir, batch_size, num_workers=2):
    # Define the transformation to be applied to each image
    augmented_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the training and validation dataset from the directory with the specified transform
    train_val_dataset = datasets.ImageFolder(root=root_dir, transform=augmented_transform)
    
    # Load the test dataset
    test_dataset = datasets.ImageFolder(root=test_dir, transform=augmented_transform)

    # Calculate sizes for train and validation sets (using full train_val_dataset)
    train_size = int(0.875 * len(train_val_dataset))  # 70% of the original dataset size, since test is now separate
    val_size = len(train_val_dataset) - train_size

    # Extract labels for stratification of train and validation
    targets = [s[1] for s in train_val_dataset.samples]

    # Split the train_val_dataset into training and validation sets
    train_idx, val_idx = train_test_split(
        range(len(targets)),
        test_size=val_size,
        stratify=targets,
        random_state=42
    )

    # Creating subsets for each split
    train_dataset = Subset(train_val_dataset, train_idx)
    validation_dataset = Subset(train_val_dataset, val_idx)

    # Creating data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, validation_loader, test_loader

## Data Loading and Preprocessing ##


## Model Training and Evaluation ##
def train_and_evaluate(model, train_loader, validation_loader, test_loader, epochs=10, lr=0.0001, early_stop_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize Optimizer, Loss Function, and Learning Rate Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=lr,
                           weight_decay=1e-4, 
                           betas=(0.9, 0.999),
                           eps=1e-8,
                           amsgrad=True)
    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Early stopping initialization
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Initialize lists to store accuracies and losses
    train_accuracies = []
    test_accuracies = []
    validation_accuracies = []
    train_losses = []
    test_losses = []
    validation_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Apply sigmoid to the outputs to get the probability
            probs = torch.sigmoid(outputs)
            # Convert probabilities to predicted class (0 or 1)
            predicted = probs > 0.5
            correct_train += (predicted == labels).type(torch.float).sum().item()
            total_train += labels.size(0)

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss / len(train_loader))

        # Validation phase
        model.eval()
        correct_validation = 0
        total_validation = 0
        validation_loss = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float().view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
      
                # Apply sigmoid to the outputs to get the probability
                probs = torch.sigmoid(outputs)
                # Convert probabilities to predicted class (0 or 1)
                predicted = probs > 0.5
                correct_validation += (predicted == labels).type(torch.float).sum().item()
                total_validation += labels.size(0)

        validation_accuracy = 100 * correct_validation / total_validation
        validation_accuracies.append(validation_accuracy)
        validation_losses.append(validation_loss / len(validation_loader))

        # Learning Rate Scheduler
        scheduler.step(validation_loss / len(validation_loader))
        
        # Early Stopping Check
        if validation_loss / len(validation_loader) < best_val_loss:
            best_val_loss = validation_loss / len(validation_loader)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epochs...")
            if early_stop_counter == early_stop_patience:
                print("Stopping early due to lack of improvement in validation loss.")
                break

        # Evaluate on the test set
        test_loss = 0 
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float().view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Apply sigmoid to the outputs to get the probability
                probs = torch.sigmoid(outputs)
                # Convert probabilities to predicted class (0 or 1)
                predicted = probs > 0.5
                correct_test += (predicted == labels).type(torch.float).sum().item()
                total_test += labels.size(0)

        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss / len(test_loader))

        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Validation Loss: {validation_loss/len(validation_loader):.4f}, "
              f"Test Loss: {test_loss/len(test_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Accuracy: {validation_accuracy:.2f}%, "
              f"Test Accuracy: {test_accuracy:.2f}%")
        print("--------------------------------------------------------------")

    return train_accuracies, test_accuracies, validation_accuracies, train_losses, test_losses, validation_losses
## Model Training and Evaluation ##

## Hyperparameter Search ##
def train_and_evaluate_new(model, train_loader, validation_loader, test_loader, epochs=10, lr=0.0001, save_path='best_model.pt', early_stop_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize Optimizer and Loss Function
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=lr,
                           weight_decay=1e-4, 
                           betas=(0.9, 0.999),
                           eps=1e-8,
                           amsgrad=True)
    # Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Early stopping initialization
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    # Initialize metrics storage
    metrics = {
        'best_validation_accuracy': 0,
        'best_validation_loss': float('inf'),
        'corresponding_test_accuracy': 0,
        'corresponding_test_loss': float('inf'),
        'final_train_accuracy': 0,
        'final_train_loss': float('inf'),
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Apply sigmoid to the outputs to get the probability
            probs = torch.sigmoid(outputs)
            # Convert probabilities to predicted class (0 or 1)
            predicted = probs > 0.5
            correct_train += (predicted == labels).type(torch.float).sum().item()
            total_train += labels.size(0)
        
        # Compute training metrics
        train_accuracy = 100 * correct_train / total_train
        train_loss_avg = train_loss / len(train_loader)

        model.eval()
        validation_loss = 0
        correct_validation = 0
        total_validation = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float().view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                
                # Apply sigmoid to the outputs to get the probability
                probs = torch.sigmoid(outputs)
                # Convert probabilities to predicted class (0 or 1)
                predicted = probs > 0.5
                correct_validation += (predicted == labels).type(torch.float).sum().item()
                total_validation += labels.size(0)

        validation_accuracy = 100 * correct_validation / total_validation
        validation_loss_avg = validation_loss / len(validation_loader)

        # Learning Rate Scheduler
        scheduler.step(validation_loss / len(validation_loader))
        
        # Early Stopping Check
        if validation_loss / len(validation_loader) < best_val_loss:
            best_val_loss = validation_loss / len(validation_loader)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epochs...")
            if early_stop_counter == early_stop_patience:
                print("Stopping early due to lack of improvement in validation loss.")
                break

        # Evaluate on the test set
        test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.float().view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Apply sigmoid to the outputs to get the probability
                probs = torch.sigmoid(outputs)
                # Convert probabilities to predicted class (0 or 1)
                predicted = probs > 0.5
                correct_test += (predicted == labels).type(torch.float).sum().item()
                total_test += labels.size(0)

        test_accuracy = 100 * correct_test / total_test
        test_loss_avg = test_loss / len(test_loader)

        # Update metrics if better validation accuracy is found
        if validation_accuracy > metrics['best_validation_accuracy']:
            metrics.update({
                'best_validation_accuracy': validation_accuracy,
                'best_validation_loss': validation_loss_avg,
                'corresponding_test_accuracy': test_accuracy,
                'corresponding_test_loss': test_loss_avg,
                'save_path': save_path
            })
            torch.save(model.state_dict(), save_path)

        # Always update these metrics after each epoch
        metrics.update({
            'final_train_accuracy': train_accuracy,
            'final_train_loss': train_loss_avg,
        })

        print(f"Epoch {epoch+1}, Train Loss: {train_loss_avg:.4f}, "
              f"Validation Loss: {validation_loss_avg:.4f}, "
              f"Test Loss: {test_loss_avg:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Accuracy: {validation_accuracy:.2f}%, "
              f"Test Accuracy: {test_accuracy:.2f}%")
        print("--------------------------------------------------------------")
    
    return metrics

# Create the dataloders
root_directory = '../processed_dataset_frame/processed_dataset_frame'
test_directory = '../processed_dataset_frame_test'

def random_search_CNN(hyperparameters, num_trials, base_save_path='../models_weight_CNN'):
    train_loader, validation_loader, test_loader = setup_data_loaders(root_directory, test_directory, batch_size=16)
    
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    results = []
    best_validation_accuracy = 0
    best_model_path = ""

    for trial in range(num_trials):
        # Randomly sample hyperparameters
        lr = random.choice(hyperparameters['learning_rate'])
        dropout = random.choice(hyperparameters['dropout_rate'])
        fc_units = random.choice(hyperparameters['fc_units'])
        
        # Define save path for the current trial's best model
        save_path = os.path.join(base_save_path, f'best_model_trial_{trial+1}.pt')
        
        # Initialize and train the model
        print(f"Trial {trial+1}: Training with lr={lr}, dropout={dropout}, fc_units={fc_units}")
        model = CNN_for_DeepFake(dropout_rate=dropout, fc_units=fc_units)
        metrics = train_and_evaluate_new(model, train_loader, validation_loader, test_loader, epochs=20, lr=lr, save_path=save_path)
        
        # Append metrics to results list
        results.append(metrics)
        
        # Update the path of the best model if the current model performence is better
        if metrics['best_validation_accuracy'] > best_validation_accuracy:
            best_validation_accuracy = metrics['best_validation_accuracy']
            best_model_path = metrics['save_path']
    
    print(f"Best model saved at: {best_model_path}")
    return results, best_model_path

def random_search_MLP(hyperparameters, num_trials, base_save_path='../models_weight_MLP'):
    train_loader, validation_loader, test_loader = setup_data_loaders(root_directory, test_directory, batch_size=16)
    
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    results = []
    best_validation_accuracy = 0
    best_model_path = ""

    for trial in range(num_trials):
        # Randomly sample hyperparameters
        lr = random.choice(hyperparameters['learning_rate'])
        dropout = random.choice(hyperparameters['dropout_rate'])
        architecture = random.choice(hyperparameters['architecture'])
        
        # Define save path for the current trial's best model
        save_path = os.path.join(base_save_path, f'best_model_trial_{trial+1}.pt')
        
        # Model initialization
        input_features = 224*224*3  # 224x224 image with 3 channels

        # Initialize and train the model
        print(f"Trial {trial+1}: Training with lr={lr}, dropout={dropout}, architecture={architecture}")
        model = MLP_for_DeepFake(input_features=input_features, dropout_rate=dropout, hidden_units=architecture)
        metrics = train_and_evaluate_new(model, train_loader, validation_loader, test_loader, epochs=20, lr=lr, save_path=save_path)
        
        # Append metrics to results list
        results.append(metrics)
        
        # Update the path of the best model if the current model performence is better
        if metrics['best_validation_accuracy'] > best_validation_accuracy:
            best_validation_accuracy = metrics['best_validation_accuracy']
            best_model_path = metrics['save_path']
    
    print(f"Best model saved at: {best_model_path}")
    return results, best_model_path

def random_search_CNNLSTM(hyperparameters, num_trials, base_save_path='../models_weight_CNNLSTM'):
    train_loader, validation_loader, test_loader = setup_data_loaders(root_directory, test_directory, batch_size=16)
    
    if not os.path.exists(base_save_path):
        os.makedirs(base_save_path)

    results = []
    best_validation_accuracy = 0
    best_model_path = ""

    for trial in range(num_trials):
        # Randomly sample hyperparameters
        lr = random.choice(hyperparameters['learning_rate'])
        dropout = random.choice(hyperparameters['dropout_rate'])
        fc_units = random.choice(hyperparameters['fc_units'])
        lstm_units = random.choice(hyperparameters['lstm_units'])
        num_layers = random.choice(hyperparameters['num_layers'])
        
        # Define save path for the current trial's best model
        save_path = os.path.join(base_save_path, f'best_model_trial_{trial+1}.pt')
        
        # Initialize and train the model
        print(f"Trial {trial+1}: Training with lr={lr}, dropout={dropout}, fc_units={fc_units}, lstm_units={lstm_units}, num_layers={num_layers}")
        model = CNN_LSTM_for_DeepFake(dropout_rate=dropout, fc_units=fc_units, lstm_units=lstm_units, num_layers=num_layers)
        model.to(device)
        metrics = train_and_evaluate_new(model, train_loader, validation_loader, test_loader, epochs=20, lr=lr, save_path=save_path)

        # Append metrics to results list
        results.append(metrics)
        
        # Update the path and info of the best model if the current model's performance is better
        if metrics['best_validation_accuracy'] > best_validation_accuracy:
            best_validation_accuracy = metrics['best_validation_accuracy']
            best_model_info = {
                'path': save_path,
                'lr': lr,
                'dropout': dropout,
                'fc_units': fc_units,
                'lstm_units': lstm_units,
                'num_layers': num_layers,
                'validation_accuracy': best_validation_accuracy
            }
    
    print(f"Best model info: {best_model_info}")
    return results, best_model_info